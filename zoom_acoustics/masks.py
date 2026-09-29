"""
masks.py -- time windows to EXCLUDE from passive statistics.

Why: an active chirp, a speaker, a pour, a knock, somebody talking next to the air mic.
If you sent active chirps (the Sep-2026 sessions had a 1 s chirp every 10 s, 40-70 dB above
the reaction), every passive level, spectrum and event rate must be computed with the chirp
windows removed, or you are measuring the loudspeaker.

A mask is a list of (t_start_s, t_end_s) pairs in seconds from the start of the take.
Sources, combined by windows_for_take():
  1. config  takes: <take>: masks: [[t0, t1], ...]         (hand-entered)
  2. config  masks: periodic: {period_s, first_s, pad_before_s, pad_after_s}
             (optionally per take under takes: <take>: periodic: {...})
  3. find_periodic_bursts() on a reference channel, when the chirp phase is unknown.
"""
from __future__ import annotations

import numpy as np


def _edges(windows):
    w = merge(windows)
    if not w:
        return np.zeros(0), np.zeros(0)
    a, b = np.array(w, float).T
    return a, b


def mask_array(t, windows):
    """Boolean array, True where t falls INSIDE any window.  O(N log W): the windows are
    merged and each t is located with searchsorted (a 24 h take has ~8600 chirp windows)."""
    t = np.asarray(t, float)
    a, b = _edges(windows)
    if not a.size:
        return np.zeros(t.shape, bool)
    k = np.searchsorted(a, t, "right") - 1
    return (k >= 0) & (t < b[np.maximum(k, 0)])


def covered_before(x, windows):
    """Total masked time in (-inf, x) for each x (vectorised).  Masked time in [x0, x1) is
    covered_before(x1) - covered_before(x0)."""
    x = np.asarray(x, float)
    a, b = _edges(windows)
    if not a.size:
        return np.zeros(x.shape)
    ln = b - a
    cum = np.r_[0.0, np.cumsum(ln)]
    k = np.searchsorted(a, x, "right") - 1
    kk = np.maximum(k, 0)
    part = np.clip(x - a[kk], 0.0, ln[kk])
    return np.where(k >= 0, cum[kk] + part, 0.0)


def periodic_windows(first_s, period_s, duration_s, pad_before_s=0.25, pad_after_s=1.4):
    """Windows around a periodic source (e.g. a chirp every period_s seconds).
    Default pads: 0.25 s before to 1.4 s after each start = a 1 s sweep + 0.4 s ringing."""
    if not period_s or period_s <= 0:
        return []
    k0 = int(np.floor((0 - first_s) / period_s))
    starts = first_s + period_s * np.arange(k0, int(np.ceil((duration_s - first_s) / period_s)) + 1)
    return [(float(s - pad_before_s), float(s + pad_after_s)) for s in starts
            if s + pad_after_s > 0 and s - pad_before_s < duration_s]


def merge(windows):
    """Sort and merge overlapping windows."""
    w = sorted((float(a), float(b)) for a, b in windows or [] if b > a)
    out = []
    for a, b in w:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def windows_for_take(cfg, take_name, duration_s):
    """All configured mask windows for one take, merged."""
    wins = []
    opts = cfg.take_opts(take_name) if cfg is not None else {}
    wins += [tuple(w) for w in opts.get("masks", []) or []]
    per = opts.get("periodic") or ((cfg.get("masks") or {}).get("periodic") if cfg else None)
    if per:
        wins += periodic_windows(per.get("first_s", 0.0), per["period_s"], duration_s,
                                 per.get("pad_before_s", 0.25), per.get("pad_after_s", 1.4))
    return merge(wins)


def live_seconds(t0, t1, windows):
    """Seconds of [t0, t1) not covered by windows (t0, t1 may be arrays)."""
    t0 = np.asarray(t0, float)
    t1 = np.asarray(t1, float)
    live = (t1 - t0) - (covered_before(t1, windows) - covered_before(t0, windows))
    live = np.maximum(live, 0.0)
    return float(live) if live.ndim == 0 else live


def detector_dead_windows(windows, lta_s, duration_s):
    """Time in which the STA/LTA ratio is NaN, so no trigger can fire (THEORY section 4):
    the first lta_s/2 of the take (the LTA needs more than half its window filled), every
    mask, and lta_s/2 after the end of every mask at least lta_s/2 long (the LTA re-fills).
    Shorter masks leave the LTA more than half valid and add no extra dead time."""
    h = 0.5 * float(lta_s)
    out = [(0.0, h)]
    for a, b in merge(windows):
        out.append((a, min(b + h, duration_s)) if (b - a) >= h else (a, b))
    return merge(out)


def find_periodic_bursts(t, level, period_s=None, min_dur_s=0.3, max_dur_s=3.0,
                         thresholds=(32, 20, 12, 8, 5, 3), tol_s=0.05):
    """Locate a periodic loud source (chirp train) in a level series.

    Run it on the channel that hears the source best (the air mic, usually).  The
    threshold (x median) is SWEPT and the one giving the most grid inliers wins, because a
    fixed threshold fails once the reaction raises the median.  If `period_s` is known (from
    a signal-generator log) pass it: fitting the period from the audio is how an earlier
    analysis once got 4.81 s for a true 10.00 s.

    Returns dict(first_s, period_s, n_inliers, n_candidates, residual_ms) or None.
    """
    t = np.asarray(t, float)
    e = np.asarray(level, float)
    dt = float(np.median(np.diff(t)))
    sm = max(1, int(round(0.05 / dt)))        # 50 ms boxcar; a no-op when dt >= 50 ms
    es = np.convolve(np.nan_to_num(e), np.ones(sm) / sm, mode="same")
    pos = es[es > 0]
    if not pos.size:
        return None
    med = float(np.median(pos))
    best = None
    for thr in thresholds:
        hot = es > thr * med
        d = np.diff(np.r_[0, hot.astype(int), 0])
        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        dur = (en - st) * dt
        cand = t[st[(dur >= min_dur_s) & (dur <= max_dur_s)]]
        if len(cand) < 4:
            continue
        gaps = np.diff(cand)
        gaps = gaps[gaps > 2 * min_dur_s]
        per = period_s if period_s else (float(np.median(gaps)) if gaps.size else None)
        if not per:
            continue
        k = np.round((cand - cand[0]) / per)
        phase = float(np.median(cand - per * k))
        res = cand - (phase + per * k)
        inl = np.abs(res - np.median(res)) < tol_s
        if best is None or inl.sum() > best[0]:
            best = (int(inl.sum()), cand, per, phase)
    if best is None or best[0] < 3:
        return None
    n_in, cand, per, phase = best
    k = np.round((cand - phase) / per)
    res = cand - (phase + per * k)
    good = np.abs(res - np.median(res)) < tol_s
    if good.sum() >= 3 and period_s is None:
        A = np.vstack([np.ones(good.sum()), k[good]]).T
        (phase, per), *_ = np.linalg.lstsq(A, cand[good], rcond=None)
    elif good.sum() >= 1:
        phase = float(np.median(cand[good] - per * k[good]))
    res = cand - (phase + per * k)
    good = np.abs(res) < tol_s
    first = float(phase - per * np.floor(phase / per))
    return dict(first_s=first, period_s=float(per), n_inliers=int(good.sum()),
                n_candidates=int(len(cand)),
                residual_ms=float(np.std(res[good]) * 1e3) if good.sum() > 1 else np.nan)
