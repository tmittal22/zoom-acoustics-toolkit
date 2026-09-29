"""
catalogue.py -- from triggers to an event catalogue: waveforms, per-event features,
temporal clustering, cross-channel coincidence, and waveform clustering.

Three different meanings of "clustering", kept separate on purpose:

1. TEMPORAL clustering -- are events bunched in time more than a Poisson process?
   interevent_stats(): CV of inter-event gaps (1 for Poisson), KS test against an
   exponential with the same mean, Fano factor of counts in windows (1 for Poisson), and
   the fraction of gaps shorter than 2x the dead time (one source firing more than once).
   Gaps that straddle a masked window are REMOVED first: in Sep 2026 about 1 % of gaps
   were holes cut by the chirp mask and carried 60-92 % of the variance (CV 2.58 -> 1.24).
   burst_sweep(): number of bursts as a function of the gap that separates them.  If the
   gap distribution has no natural break, the burst count depends on that choice (38 -> 5
   bursts for gaps of 1 -> 4 s in one Sep-2026 run), and no single value may be quoted.

2. CROSS-CHANNEL coincidence -- coincidence(): fraction of events on channel A with an
   event on B within +/- win, compared with the same count after time-shifting B by
   several seconds (the chance level).  Only the excess over chance means anything.

3. WAVEFORM clustering -- similar waveforms, e.g. a rattling fitting that repeats.
   xcorr_matrix(): peak normalised cross-correlation over lag for every pair (lag search
   matters: identical sources arrive with trigger jitter).  cluster_waveforms():
   average-linkage hierarchical clustering on 1 - rho, cut at rho_min, returned as ranked
   CANDIDATE clusters.  feature_clusters(): Ward clustering of standardised log-features
   (descriptive only; check stability, see STUDENT_GUIDE).

   CAUTION: simple damped sinusoids of similar frequency correlate strongly with each
   other whatever their source.  Correlation clusters of bubbles are expected.  A
   mechanical repeater stands out only if it is clearly MORE similar than those (on the
   demo contact channel: rho 0.94 vs <= 0.87 for bubble clusters; on a hydrophone where the
   rattle is weaker it ties with them at 0.88).  Also check f_peak_hz: a cluster sitting on
   an apparatus line (5.6 kHz in the Sep-2026 rig) is the line, not a source.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as sg, stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from . import dsp, masks as M


# ================================================================ 1. temporal
def live_gaps(t, windows=None):
    """Inter-event gaps with every gap that straddles a masked window removed.
    Returns (gaps, n_removed)."""
    t = np.sort(np.asarray(t, float))
    if t.size < 2:
        return np.zeros(0), 0
    a, b = t[:-1], t[1:]
    g = b - a
    cov = M.covered_before(b, windows or []) - M.covered_before(a, windows or [])
    keep = (cov <= 0) & (g > 0)
    return g[keep], int((cov > 0).sum())


def interevent_stats(t, windows=None, dead_s=1.5e-3, fano_windows=(1.0, 5.0), t_range=None,
                     n_min=100):
    """Temporal clustering statistics for one event list (seconds).  Poisson: CV = 1,
    Fano = 1, KS p not small.  Returns a dict; `reportable` is False below n_min events."""
    t = np.sort(np.asarray(t, float))
    if t_range is not None:
        t = t[(t >= t_range[0]) & (t < t_range[1])]
    out = dict(n=int(t.size), reportable=bool(t.size >= n_min))
    g, nrem = live_gaps(t, windows)
    if g.size < 10:
        return out
    mu, sd = float(g.mean()), float(g.std(ddof=1))
    ks = stats.kstest(g, "expon", args=(0.0, mu))
    out.update(n_gaps=int(g.size), n_gaps_removed_straddling_mask=nrem, mean_gap_s=mu,
               cv=sd / mu, ks_D=float(ks.statistic), ks_p=float(ks.pvalue),
               frac_gaps_below_2x_dead=float(np.mean(g < 2 * dead_s)))
    lo, hi = (t_range or (t[0], t[-1]))
    for w in fano_windows:
        edges = np.arange(lo, hi + w, w)
        if edges.size < 4:
            continue
        c = np.histogram(t, edges)[0].astype(float)
        live = M.live_seconds(edges[:-1], edges[1:], windows or [])
        full = live >= 0.999 * w                     # only windows with no masked time
        if full.sum() >= 3 and c[full].mean() > 0:
            out[f"fano_{w:g}s"] = float(c[full].var(ddof=1) / c[full].mean())
    return out


def burst_sweep(t, gaps=(0.05, 0.1, 0.2, 0.5, 1, 2, 5), windows=None):
    """Number of bursts (runs of events separated by < gap) for each gap value.
    A plateau means a natural burst scale; a steady decline means there is none."""
    g, _ = live_gaps(t, windows)
    return pd.DataFrame(dict(gap_s=list(gaps), n_bursts=[int(1 + (g > x).sum()) for x in gaps]))


# ================================================================ 2. coincidence
def coincidence(ta, tb, win_s=3e-3, shifts_s=(-7.3, -5.1, -3.7, 3.3, 4.9, 7.7)):
    """Fraction of A events with a B event within +/- win_s, the chance fraction from
    time-shifted B, and the excess (obs - chance)/(1 - chance).  Also the lag
    distribution (B minus A) of coincident pairs."""
    ta = np.sort(np.asarray(ta, float))
    tb = np.sort(np.asarray(tb, float))
    if ta.size < 10 or tb.size < 10:
        return dict(n_a=int(ta.size), n_b=int(tb.size), reportable=False)

    def nearest(tb_):
        i = np.searchsorted(tb_, ta)
        lo = tb_[np.clip(i - 1, 0, tb_.size - 1)] - ta
        hi = tb_[np.clip(i, 0, tb_.size - 1)] - ta
        return np.where(np.abs(lo) < np.abs(hi), lo, hi)

    d = nearest(tb)
    hit = np.abs(d) <= win_s
    f_obs = float(hit.mean())
    f_null = float(np.mean([np.mean(np.abs(nearest(tb + s)) <= win_s) for s in shifts_s]))
    return dict(n_a=int(ta.size), n_b=int(tb.size), reportable=True, win_s=win_s,
                frac_obs=f_obs, frac_chance=f_null,
                excess=float((f_obs - f_null) / max(1 - f_null, 1e-12)),
                lag_median_ms=float(np.median(d[hit]) * 1e3) if hit.any() else np.nan,
                lags_s=d[hit])


# ================================================================ 3. waveforms
def extract_waveforms(take, times_s, ch, pre_s=0.002, post_s=0.010, highpass_hz=None,
                      max_events=2000, rng=0):
    """Raw waveform snippets around trigger times from the WAV (not the cache).

    Returns (W[n, m] float64, times_used, sr).  Events closer than pre_s to the start or
    post_s to the end are dropped.  With more than max_events, a random subset is taken
    (reproducible via rng) -- reading 10^5 snippets from a 2 GB file is slow and rarely
    needed.  A trigger time is the centre of the 0.5 ms envelope sample that fired, so the
    onset sits within +/- 0.25 ms of index pre_s*sr."""
    t = np.sort(np.asarray(times_s, float))
    t = t[(t >= pre_s) & (t <= take.duration_s - post_s)]
    if t.size > max_events:
        t = np.sort(np.random.default_rng(rng).choice(t, max_events, replace=False))
    sr = take.sr
    n0, n1 = int(round(pre_s * sr)), int(round(post_s * sr))
    W = np.zeros((t.size, n0 + n1))
    sos = dsp.bandpass_sos(highpass_hz, sr / 2, sr) if highpass_hz else None
    for i, ti in enumerate(t):
        pad = 0.005 if sos is not None else 0.0              # filter warm-up
        x = take.read(ti - pre_s - pad, ti + post_s, [ch])[:, 0]
        if sos is not None:
            x = sg.sosfiltfilt(sos, x)[int(round(pad * sr)):]
        x = x[:n0 + n1]
        W[i, :x.size] = x - x[:max(1, n0 // 2)].mean()
    return W, t, sr


def waveform_features(W, sr, pre_s=0.002):
    """Per-event features from snippets (rows of W):
    peak_amp, energy (sum x^2 / sr), rise_ms (10 % -> peak of the Hilbert envelope),
    decay_ms (e-folding time from a log-linear fit to the envelope after the peak),
    f_peak_hz and f_centroid_hz (Hann-windowed spectrum), bandwidth_hz (spectral std),
    q_est = pi * f_peak * decay (for a damped sinusoid, Q = pi f tau), and peak2_rel_db:
    the strongest spectral peak more than 20 % away from f_peak, relative to it.  A free
    bubble rings in ONE mode (peak2 far below 0 dB); a mechanical rattle or a structure mode
    is usually multi-modal (peak2 within ~10 dB)."""
    env = np.abs(sg.hilbert(W, axis=1))
    n0 = int(round(pre_s * sr))
    f = np.fft.rfftfreq(W.shape[1], 1 / sr)
    S = np.abs(np.fft.rfft(W * np.hanning(W.shape[1]), axis=1)) ** 2
    rows = []
    for w, e, s in zip(W, env, S):
        ip = int(np.argmax(e))
        pk = float(e[ip])
        base = float(np.median(e[:max(2, n0 // 2)]))
        pre = np.flatnonzero(e[:ip + 1] < max(0.1 * pk, 2 * base))
        i0 = int(pre[-1]) if pre.size else 0
        tail = e[ip:]
        stop = np.flatnonzero(tail < max(0.05 * pk, 2 * base))
        nt = int(stop[0]) if stop.size else len(tail)
        decay = np.nan
        if nt >= 5:
            sl = np.polyfit(np.arange(nt) / sr, np.log(tail[:nt] + 1e-30), 1)[0]
            decay = -1.0 / sl if sl < 0 else np.nan
        tot = s.sum() + 1e-30
        fc = float((f * s).sum() / tot)
        jp = int(np.argmax(s))
        fpk = float(f[jp])
        away = (np.abs(f / max(fpk, 1e-9) - 1) > 0.2) & (f > 0)
        p2 = float(10 * np.log10(s[away].max() / max(s[jp], 1e-30))) if away.any() else np.nan
        rows.append(dict(peak_amp=pk, energy=float((w ** 2).sum() / sr),
                         rise_ms=(ip - i0) / sr * 1e3, decay_ms=decay * 1e3,
                         f_peak_hz=fpk, f_centroid_hz=fc,
                         bandwidth_hz=float(np.sqrt(((f - fc) ** 2 * s).sum() / tot)),
                         q_est=float(np.pi * fpk * decay) if np.isfinite(decay) else np.nan,
                peak2_rel_db=p2,
                         snr_db=float(20 * np.log10(pk / max(base, 1e-30)))))
    return pd.DataFrame(rows)


def xcorr_matrix(W, max_lag=None):
    """Peak |normalised cross-correlation| over lag for every pair of rows (FFT based).
    max_lag (samples) limits the lag search; None = any lag."""
    w = W - W.mean(axis=1, keepdims=True)
    w /= np.sqrt((w ** 2).sum(axis=1, keepdims=True)) + 1e-30
    n = w.shape[1]
    nf = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(w, nf, axis=1)
    C = np.empty((len(w), len(w)))
    for i in range(len(w)):
        cc = np.fft.irfft(F[i][None, :] * np.conj(F), nf, axis=1)
        if max_lag is not None:
            cc = np.concatenate([cc[:, :max_lag + 1], cc[:, -max_lag:]], axis=1)
        C[i] = np.abs(cc).max(axis=1)
    np.fill_diagonal(C, 1.0)
    return np.clip((C + C.T) / 2, 0, 1)


def cluster_waveforms(C, rho_min=0.8, min_size=3, features=None, times=None, f_tol=0.05):
    """Average-linkage clustering on distance 1 - rho, cut at 1 - rho_min.

    What this can and cannot do (tested on the demo, docs/VALIDATION.md C1):
    damped sinusoids of similar frequency correlate strongly WHATEVER their source, so a
    population of unrelated bubbles forms many tight clusters (rho 0.82-0.88 on the demo).
    Correlation alone therefore cannot tell a repeating mechanical source from bubbles, and
    any null built from the data is biased because a cluster is by construction the most
    mutually similar subset.  So the output is a RANKED list of CANDIDATES, with the numbers
    that do discriminate physically:
      median_rho     internal similarity (a true repeater is near the noise ceiling)
      f_peak_hz      cluster median peak frequency           (needs `features`)
      peak2_rel_db   second spectral peak vs main, median   (needs `features`); an
                     isolated single-mode ringdown is far below 0 dB, a multi-mode source
                     near it -- but overlapping events, apparatus tones and glides in the
                     snippet also raise it (demo: bubbles -5.7 dB, rattle -4.5 dB on Ch1),
                     so it discriminates only for clean, isolated events
      null_rho       median rho to NON-members of matching f_peak (informational; biased low)
      gap_cv         CV of member inter-event gaps (needs `times`); regular source << 1
    Look at the gallery (plots.plot_waveform_gallery) before calling anything a repeater.

    Returns (labels, per-cluster summary sorted by median_rho, median pairwise rho)."""
    n = len(C)
    if n < 2:
        return np.ones(n, int), pd.DataFrame(), np.nan
    D = np.clip(1 - C, 0, 1)
    np.fill_diagonal(D, 0)
    lab = fcluster(linkage(squareform(D, checks=False), "average"), 1 - rho_min,
                   criterion="distance")
    med_all = float(np.median(C[np.triu_indices(n, 1)]))
    fp = features["f_peak_hz"].to_numpy(float) if features is not None else None
    p2 = features["peak2_rel_db"].to_numpy(float) if features is not None else None
    tt = None if times is None else np.asarray(times, float)
    rows = []
    for k in np.unique(lab):
        m = np.flatnonzero(lab == k)
        if m.size < min_size:
            continue
        inner = C[np.ix_(m, m)][np.triu_indices(m.size, 1)]
        row = dict(cluster=int(k), size=int(m.size), median_rho=float(np.median(inner)))
        if fp is not None:
            fm = float(np.median(fp[m]))
            others = np.setdiff1d(np.flatnonzero(np.abs(fp / fm - 1) < f_tol), m)
            row.update(f_peak_hz=fm, peak2_rel_db=float(np.nanmedian(p2[m])),
                       null_rho=float(np.median(C[np.ix_(m, others)])) if others.size >= 3
                       else np.nan)
        if tt is not None:
            g = np.diff(np.sort(tt[m]))
            row["gap_cv"] = float(g.std(ddof=1) / g.mean()) if g.size > 1 and g.mean() > 0 else np.nan
        rows.append(row)
    summ = (pd.DataFrame(rows).sort_values("median_rho", ascending=False).reset_index(drop=True)
            if rows else pd.DataFrame())
    return lab, summ, med_all


def feature_clusters(features: pd.DataFrame, cols=("f_peak_hz", "decay_ms", "energy",
                                                   "bandwidth_hz"), n_clusters=3, log=True):
    """Ward clustering of standardised (log) features.  Returns (labels, PCA scores[n, 2],
    explained variance ratio).  DESCRIPTIVE: k is your choice; re-run with other k and
    feature sets and only trust groups that persist."""
    X = features[list(cols)].to_numpy(float)
    ok = np.all(np.isfinite(X), axis=1) & (np.all(X > 0, axis=1) if log else True)
    Z = np.log10(X[ok]) if log else X[ok]
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    lab = np.zeros(len(X), int)
    if ok.sum() > n_clusters:
        lab[ok] = fcluster(linkage(Z, "ward"), n_clusters, criterion="maxclust")
    U, s, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    sc = np.full((len(X), 2), np.nan)
    sc[ok] = U[:, :2] * s[:2]
    return lab, sc, (s ** 2 / (s ** 2).sum())[:2]
