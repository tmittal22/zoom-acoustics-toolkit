"""
glide.py -- descending spectral ridges ("glides") in the excess spectrogram.

The observation (Sep 2026, analysis4): after acid contact, a band in the EXCESS spectrogram
(each frame divided by the take's own pre-acid mean spectrum) descends smoothly in
frequency over tens to hundreds of seconds, e.g. 6.5 -> 0.9 kHz (-2.84 octaves) in 328 s on
one hydrophone.  It appeared on wetted/contact sensors and in 0 of 14 takes on the air mic,
so it is liquid- or solid-borne.  Its mechanism is NOT established (see STUDENT_GUIDE).

Why a dedicated tracker: argmax inside fixed sub-bands fails, because the ridge descends
through every band edge and the estimate jumps between competing peaks (2.88 -> 3.40 ->
2.41 kHz on a ridge that is monotonic).  track_ridge() follows the local maximum forward in
time under a continuity constraint (+/- tol_frac per step), so it cannot jump features.

Known failure of multi-ridge tracking: seeds started on different bands can slide into the
dominant ridge and all end on it, producing frequency ratios of 1.00 by construction.
track_ridges() therefore reports `merged_with` for any pair of tracks that converge, and
harmonic_ladder() tests harmonics ALONG one measured ridge instead of tracking them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import dsp, masks as M
from .physics import minnaert_R, bond_number, capillary_length


def excess_spectrogram(feat, ch, base_window, t_range=None, f_range=(300.0, 20000.0),
                       base_feat=None):
    """Ratio spectrogram D[t, f] = PSD(t, f) / mean PSD over base_window (unmasked).
    base_feat: take the baseline from ANOTHER take (same sensor, same geometry, same gain),
    for takes that have no quiet stretch of their own.  Frequency grids must match.
    Returns (t, f, D) restricted to f_range and unmasked frames inside t_range."""
    bf = base_feat or feat
    if bf is not feat and not np.array_equal(bf.freqs, feat.freqs):
        raise ValueError("baseline take has a different frequency grid")
    base, n = bf.psd_mean(ch, *base_window)
    if base is None:
        raise ValueError(f"{feat.take} Ch{ch}: no unmasked frames in baseline {base_window}")
    t = feat.t_psd
    f = feat.freqs
    fm = (f >= f_range[0]) & (f <= min(f_range[1], f[-1]))
    t0, t1 = t_range or (0.0, feat.duration_s)
    tm = (t >= t0) & (t <= t1) & ~M.mask_array(t, feat.mask_windows)
    idx = np.flatnonzero(tm)
    P = feat.psd(ch)
    D = np.empty((idx.size, int(fm.sum())))
    j0 = np.flatnonzero(fm)
    for i in range(0, idx.size, 2000):
        blk = np.asarray(P[idx[i:i + 2000]][:, j0[0]:j0[-1] + 1], float)
        D[i:i + 2000] = blk / np.maximum(base[fm][None, :], 1e-30)
    return t[idx], f[fm], D


def _follow(t, f, D, seed_hz, tol_frac, tol_hz, smooth, seed_s):
    cur = float(seed_hz)
    sm = None
    out = []
    for i in range(len(t)):
        tol = cur * tol_frac + tol_hz
        m = (f > cur - tol) & (f < cur + tol)
        if m.sum() < 3:
            continue
        row = D[i, m]
        pk = float(f[m][int(np.argmax(row))])
        sm = pk if sm is None else smooth * sm + (1 - smooth) * pk
        cur = sm
        contrast = float(dsp.db(row.max()) - dsp.db(np.median(row)))
        out.append((float(t[i]), sm, float(row.max()), contrast))
    return pd.DataFrame(out, columns=["t_s", "f_hz", "peak_ratio", "contrast_db"])


def track_ridge(feat, ch, base_window, t0, t1=None, seed=(4000.0, 12000.0), seed_s=8.0,
                tol_frac=0.12, tol_hz=120.0, smooth=0.82, f_range=(300.0, 20000.0),
                seed_hz=None, base_feat=None):
    """Track ONE ridge forward in time from t0.

    Seed: the strongest bin of the mean excess spectrum over the first `seed_s` seconds
    inside the `seed` frequency window (or `seed_hz` if given).  Each later frame searches
    only within +/- (tol_frac * f + tol_hz) of the current (smoothed) ridge frequency;
    the new value is smoothed with a single-pole filter  f <- s f + (1-s) f_peak.
    The smoother LAGS the ridge by  tau = psd_dt * s / (1 - s)  (1.14 s for the defaults),
    so on a ridge falling at rate r = |d ln f / dt| the tracked frequency is high by ~ r*tau
    (0.7 % for the Sep-2026 glide, 0.6 %/s; tested in test_glide_catalogue.py).

    Returns a DataFrame t_s, f_hz, peak_ratio (excess ratio at the peak, linear) and
    contrast_db (peak minus median of the search window, dB).  A tracker ALWAYS returns a
    path, even through pure noise, and a noise path drifts (on the demo air mic, which has
    no ridge, it "glides" -0.36 octaves).  Only a path whose contrast is well above the
    noise value (a few dB) is a ridge: see summarize().
    """
    t, f, D = excess_spectrogram(feat, ch, base_window, (t0, t1 or feat.duration_s), f_range,
                                 base_feat)
    if len(t) < 20:
        return pd.DataFrame(columns=["t_s", "f_hz", "peak_ratio", "contrast_db"])
    if seed_hz is None:
        n0 = max(1, int(round(seed_s / feat.psd_dt)))
        s0 = D[:n0].mean(axis=0)
        m0 = (f >= seed[0]) & (f <= seed[1])
        if not m0.any():
            raise ValueError("seed window outside the frequency range")
        seed_hz = float(f[m0][np.argmax(s0[m0])])
    return _follow(t, f, D, seed_hz, tol_frac, tol_hz, smooth, seed_s)


def track_ridges(feat, ch, base_window, t0, t1=None, n=3, seed=(700.0, 14000.0), seed_s=8.0,
                 min_sep=1.25, merge_tol=0.05, base_feat=None, **kw):
    """Up to n ridges from the n strongest early-excess peaks at least `min_sep` apart in
    frequency ratio.  A pair of tracks whose frequencies end within merge_tol (fractional)
    of each other is flagged in `merged_with`: they have collapsed onto one feature and must
    not be reported as two (or as a harmonic pair)."""
    t, f, D = excess_spectrogram(feat, ch, base_window, (t0, t1 or feat.duration_s),
                                 kw.pop("f_range", (300.0, 20000.0)), base_feat)
    n0 = max(1, int(round(seed_s / feat.psd_dt)))
    s0 = D[:n0].mean(axis=0)
    seeds = []
    for i in np.argsort(s0)[::-1]:
        fi = float(f[i])
        if not seed[0] <= fi <= seed[1]:
            continue
        if any(max(fi, s) / min(fi, s) < min_sep for s in seeds):
            continue
        seeds.append(fi)
        if len(seeds) >= n:
            break
    tracks = []
    for s in sorted(seeds):
        tr = _follow(t, f, D, s, kw.get("tol_frac", 0.12), kw.get("tol_hz", 120.0),
                     kw.get("smooth", 0.82), seed_s)
        tracks.append(dict(seed_hz=s, track=tr, merged_with=[]))
    for i, a in enumerate(tracks):
        for j, b in enumerate(tracks):
            if i < j and len(a["track"]) and len(b["track"]):
                fa, fb = a["track"].f_hz.iloc[-1], b["track"].f_hz.iloc[-1]
                if abs(fa - fb) / max(fa, fb) < merge_tol:
                    a["merged_with"].append(j)
                    b["merged_with"].append(i)
    return tracks


def summarize(track, glide_octaves=-0.5, edge_s=3.0, min_contrast_db=6.0, lock_s=10.0):
    """Glide summary of one track.

    f_start / f_end are MEDIANS over the first / last `edge_s` seconds (default 3 s) of the
    LOCKED part of the track (rolling `lock_s` median contrast > min_contrast_db), not single
    frames.  Keep edge_s short: a real ridge can fall fast at first (take 260910_016 Ch1
    drops 6.5 -> 4.2 kHz in ~10 s), and a long edge window then understates the glide.  A seed can
    sit on another feature and the path can wander through noise before it finds the ridge:
    on the demo contact channel the seed took a 9 kHz rattle and the path crept down for
    12 s at ~3 dB contrast before locking at 58 s, which a whole-track start turned into a
    spurious -0.81 octaves.  t_lock_s and frac_locked report this.
    Also: octaves = log2(f_end/f_start), median ridge contrast and `significant` (a path
    through noise sits at ~2-4 dB), and the Minnaert radius ONLY with its Bond number and a
    free-bubble feasibility flag."""
    if track is None or len(track) < 5:
        return None
    t_all = track.t_s.to_numpy()
    con_all = track.contrast_db.to_numpy() if "contrast_db" in track else np.full(len(track), 99.0)
    # LOCKED frames: rolling median contrast (window edge_s) above min_contrast_db
    dt = float(np.median(np.diff(t_all))) if len(t_all) > 1 else 1.0
    w = max(1, int(round(lock_s / dt)))
    roll = pd.Series(con_all).rolling(w, center=True, min_periods=1).median().to_numpy()
    locked = roll > min_contrast_db
    frac_locked = float(locked.mean())
    use = locked if locked.sum() >= 5 else np.ones(len(t_all), bool)
    t = t_all[use]
    fr = track.f_hz.to_numpy()[use]
    a = t <= t[0] + edge_s
    b = t >= t[-1] - edge_s
    f0, f1 = float(np.median(fr[a])), float(np.median(fr[b]))
    con = float(np.median(con_all))
    octv = float(np.log2(f1 / f0))
    # f0, f1 are medians over edge windows, so they belong to the windows' median TIMES
    dur_min = max((np.median(t[b]) - np.median(t[a])) / 60, 1e-9)
    R0, R1 = float(minnaert_R(f0)), float(minnaert_R(f1))
    return dict(t_start_s=float(t[0]), t_end_s=float(t[-1]), frac_locked=frac_locked,
                t_lock_s=float(t_all[locked][0]) if locked.any() else np.nan,
                f_start_hz=f0, f_end_hz=f1,
                octaves=octv, octaves_per_min=octv / dur_min,
                median_contrast_db=con, significant=bool(con > min_contrast_db),
                glides=bool(octv < glide_octaves and con > min_contrast_db),
                minnaert_R_start_mm=R0 * 1e3, minnaert_R_end_mm=R1 * 1e3,
                bond_end=float(bond_number(R1)),
                free_bubble_feasible_at_end=bool(bond_number(R1) < 1.0),
                note="Minnaert radius shown only as a feasibility test; Bond >= 1 means "
                     f"larger than the {capillary_length()*1e3:.2f} mm capillary length, "
                     "not a free sphere. Do not quote it as a bubble size.")


def harmonic_ladder(feat, ch, base_window, track, mults=(0.5, 1, 1.5, 2, 2.5, 3, 4, 5),
                    halfwidth=0.04, side=(0.10, 0.20), base_feat=None):
    """Is there a line at m * f1(t) along a measured ridge f1(t)?

    For each multiple m and frame: peak excess (dB) within |f/(m f1) - 1| < halfwidth minus
    the median excess (dB) in the sidebands side[0] < |f/(m f1) - 1| < side[1].
    Non-integer multiples (1.5, 2.5) are CONTROLS: their contrast is what noise gives.
      * integer m >= 2 well above the controls -> harmonics of one oscillator;
      * m = 0.5 well above the controls -> the tracked ridge is itself a HARMONIC, and the
        fundamental sits at half its frequency (re-seed lower).
    Returns one row per m: median contrast, fraction of frames > 3 dB, control flag."""
    t, f, D = excess_spectrogram(feat, ch, base_window, (track.t_s.min(), track.t_s.max()),
                                 (50.0, feat.freqs[-1]), base_feat)
    f1 = np.interp(t, track.t_s, track.f_hz)
    Ddb = dsp.db(D)
    rows = []
    for m in mults:
        c = []
        for i in range(len(t)):
            fm = m * f1[i]
            if fm * (1 + side[1]) >= f[-1] or fm * (1 - side[1]) <= f[0]:
                continue
            x = f / fm - 1
            on = np.abs(x) < halfwidth
            sb = (np.abs(x) > side[0]) & (np.abs(x) < side[1])
            if on.sum() >= 1 and sb.sum() >= 3:
                c.append(Ddb[i, on].max() - np.median(Ddb[i, sb]))
        c = np.asarray(c)
        rows.append(dict(multiple=m, control=bool(m > 1 and abs(m - round(m)) > 0.25),
                         n_frames=int(c.size),
                         median_contrast_dB=float(np.median(c)) if c.size else np.nan,
                         frac_above_3dB=float(np.mean(c > 3)) if c.size else np.nan))
    return pd.DataFrame(rows)


# ================================================================ interpretation
def interpret(track, depths_m=(0.005, 0.01, 0.02, 0.04), smooth_s=10.0):
    """What each candidate mechanism would REQUIRE to produce this ridge (THEORY section 12,
    docs/GLIDE_INTERPRETATION.md).  Nothing here is a measurement of the mechanism; it turns
    one observed f(t) into the parameter each hypothesis needs, so impossible readings show.

    Columns per frame:
      R_minnaert_mm     radius of a FREE bubble ringing at f (hypothesis A: one growing bubble)
      dRdt_um_per_s     growth rate that bubble would need (from the smoothed track)
      bond              (R / l_c)^2; >= 1 means hypothesis A is not self-consistent
      R_wall_mm         same, for a bubble touching a rigid wall: the wall lowers its frequency
                        by 0.816 (Strasberg 1953), so the observed f needs a SMALLER bubble
      beta_h{d}mm       void fraction a bubbly LAYER of thickness d would need (hypothesis B,
                        quarter-wave mode); NaN = impossible at any beta <= 2 %
    """
    from .physics import minnaert_R, bond_number, wall_factor, beta_from_layer_mode
    t = track.t_s.to_numpy(float)
    fr = track.f_hz.to_numpy(float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    w = max(1, int(round(smooth_s / dt)))
    fs = pd.Series(fr).rolling(w, center=True, min_periods=1).median().to_numpy()
    R = np.atleast_1d(minnaert_R(fs))
    # wall: f_wall = f_free(R) * 0.816 -> the free-bubble frequency is f / 0.816
    Rw = np.atleast_1d(minnaert_R(fs / wall_factor(1.0, 1.0)))
    dRdt = np.gradient(R, t) if len(t) > 2 else np.full(len(t), np.nan)
    out = pd.DataFrame(dict(t_s=t, f_hz=fs, R_minnaert_mm=R * 1e3, dRdt_um_per_s=dRdt * 1e6,
                            bond=bond_number(R), R_wall_mm=Rw * 1e3))
    for d in depths_m:
        out[f"beta_h{d*1e3:g}mm"] = beta_from_layer_mode(fs, d)
    return out


def interpretation_summary(interp):
    """One line per hypothesis: range of the required parameter and whether it stays
    physical over the whole track."""
    rows = [dict(hypothesis="A: one free bubble growing",
                 requires=f"R {interp.R_minnaert_mm.min():.2f}-{interp.R_minnaert_mm.max():.2f} mm, "
                          f"median dR/dt {np.nanmedian(interp.dRdt_um_per_s):.1f} um/s",
                 physical=bool((interp.bond < 1).all()),
                 note=f"Bond max {interp.bond.max():.2f} (>= 1 impossible as a free sphere)"),
            dict(hypothesis="A': bubble touching a wall",
                 requires=f"R {interp.R_wall_mm.min():.2f}-{interp.R_wall_mm.max():.2f} mm",
                 physical=bool((interp.R_wall_mm / 1e3 / 2.727e-3 < 1).all()),
                 note="a wall lowers f (x0.816 touching), so the SAME f needs a bubble 0.816x "
                      "smaller; physical = stays below the capillary length")]
    for c in [c for c in interp.columns if c.startswith("beta_h")]:
        b = interp[c]
        rows.append(dict(hypothesis=f"B: bubbly layer, thickness {c[6:]}",
                         requires=(f"void fraction {np.nanmin(b):.1e}-{np.nanmax(b):.1e}"
                                   if b.notna().any() else "no beta <= 2 % works"),
                         physical=bool(b.notna().all()),
                         note="beta must RISE as f falls; check against gas production"))
    return pd.DataFrame(rows)
