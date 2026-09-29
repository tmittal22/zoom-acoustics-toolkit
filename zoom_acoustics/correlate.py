"""
correlate.py -- relate acoustic quantities to pH (or any logger) over time.

Acoustic quantities (all on the lab clock, across every take, gaps between takes kept):
    band_timeline(features, ch, f_lo, f_hi, dt)   level in ANY band you choose after
                                                  processing, integrated from the stored PSD
                                                  (rectangle rule); linear FS^2
    excess(series, baseline_value)                subtract a background level (linear power)
    rate_timeline / level_timeline (timeline.py)  trigger rate, processed band levels
pH quantities:
    ph_rate(ph, smooth_s)                         dpH/dt [pH units per minute], smoothed
Correlation:
    correlate(x, y, dt_s, lags)                   Pearson + Spearman r at the best lag, the
                                                  lag scan, and an EFFECTIVE sample size that
                                                  accounts for autocorrelation
    band_scan(features, ch, target, bands, ...)   which frequency band tracks pH best:
                                                  r for every band x lag (heat map)

WHY n_eff.  Two smooth time series (a decaying sound level and a rising pH) correlate
strongly whatever the mechanism, and consecutive samples are not independent.  For two
series with lag-1 autocorrelations r1, r2 the effective number of independent pairs is
    n_eff = n (1 - r1 r2) / (1 + r1 r2)
(Bretherton et al. 1999, J. Climate 12).  p-values here use n_eff, not n.  With
n_eff < ~10 nothing is significant, however high r is: say so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import dsp


# ---------------------------------------------------------------- acoustic series
def band_level_from_psd(feat, ch, f_lo, f_hi, dt=None):
    """Mean square [FS^2] in [f_lo, f_hi) for every unmasked PSD frame of one take,
    integrated from the stored spectrogram (no re-processing needed).  Optionally averaged
    into dt-second bins (linear power).  Returns a Series indexed by take time [s]."""
    from .masks import mask_array
    f = feat.freqs
    m = (f >= f_lo) & (f < f_hi)
    if not m.any():
        raise ValueError(f"no PSD bins in {f_lo}-{f_hi} Hz (resolution {f[1]-f[0]:.1f} Hz)")
    df = float(f[1] - f[0])
    P = feat.psd(ch)
    j0, j1 = np.flatnonzero(m)[[0, -1]]
    v = np.empty(P.shape[0])
    for i in range(0, P.shape[0], 4000):
        v[i:i + 4000] = np.asarray(P[i:i + 4000, j0:j1 + 1], float).sum(axis=1) * df
    t = feat.t_psd
    v[mask_array(t, feat.mask_windows)] = np.nan
    s = pd.Series(v, index=t)
    if dt and dt > feat.psd_dt:
        b = (t // dt).astype(int)
        s = s.groupby(b).mean()
        s.index = (s.index + 0.5) * dt
    return s


def band_timeline(features, ch, f_lo, f_hi, dt=10.0):
    """band_level_from_psd for every take that has channel ch, on the lab clock, with a NaN
    after each take so gaps are not bridged."""
    parts = []
    for f in features:
        if int(ch) not in f.channels:
            continue
        s = band_level_from_psd(f, ch, f_lo, f_hi, dt)
        parts.append(pd.Series(s.to_numpy(), index=f.abs_time(s.index.to_numpy())))
        parts.append(pd.Series([np.nan], index=f.abs_time([f.duration_s + dt])))
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts).sort_index().rename(f"Ch{ch} {f_lo:g}-{f_hi:g} Hz")


def excess(series, baseline_ms):
    """Linear-power excess over a background mean square, clipped at a tiny positive value
    so it can be shown in dB."""
    return (series - float(baseline_ms)).clip(lower=1e-30)


def ph_rate(ph, col="pH", smooth_s=60.0):
    """dpH/dt [pH per minute] by a centred least-squares slope over +/- smooth_s/2 around each
    sample (robust to uneven sampling).  pH CHANGES are often the better thing to compare with
    a sound level or event rate than pH itself: a reaction rate is a rate."""
    t = (ph.index - ph.index[0]).total_seconds().to_numpy()
    y = ph[col].to_numpy(float)
    h = smooth_s / 2
    out = np.full(len(t), np.nan)
    for i in range(len(t)):
        m = (t >= t[i] - h) & (t <= t[i] + h) & np.isfinite(y)
        if m.sum() >= 3 and np.ptp(t[m]) > 0:
            out[i] = np.polyfit(t[m], y[m], 1)[0] * 60.0
    return pd.Series(out, index=ph.index, name=f"d{col}/dt [per min]")


# ---------------------------------------------------------------- correlation
def _grid(x, y, dt_s):
    rule = f"{int(round(dt_s * 1000))}ms"
    xs = x.resample(rule).mean()
    ys = y.resample(rule).mean()
    idx = xs.index.union(ys.index)
    return xs.reindex(idx), ys.reindex(idx)


def _lag1(v):
    v = np.asarray(v, float)
    a, b = v[:-1], v[1:]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4:
        return 0.0
    return float(np.corrcoef(a[m], b[m])[0, 1])


def n_effective(x, y):
    """Bretherton et al. (1999) effective number of independent pairs for two AR(1)-like
    series: n (1 - r1 r2)/(1 + r1 r2); never more than n, never less than 2."""
    n = int(np.sum(np.isfinite(x) & np.isfinite(y)))
    r1, r2 = _lag1(x), _lag1(y)
    ne = n * (1 - r1 * r2) / (1 + r1 * r2) if (1 + r1 * r2) > 0 else n
    return int(max(2, min(n, np.floor(ne)))), r1, r2


def correlate(x, y, dt_s=10.0, max_lag_s=0.0, x_in_db=True, differences=False):
    """Correlate an acoustic series x (linear power if x_in_db, converted to dB) with y (pH,
    dpH/dt, ...), both Series on datetime indices, averaged onto a common dt_s grid.

    y(t + lag) vs x(t) for lags in [-max_lag_s, max_lag_s]; positive lag = y lags x.
    differences=True correlates the step-to-step CHANGES (x[i+1]-x[i] vs y[i+1]-y[i]).  That
    removes any trend the two series share (e.g. both simply follow one decaying reaction),
    which is what makes a single monotonic take unable to establish a relation (n_eff ~ 2).
    Returns dict with the best-|r| lag, Pearson r and slope (y per dB) there, Spearman rho,
    n, n_eff, and a p-value from n_eff; plus the full lag scan (DataFrame)."""
    xx = dsp.db(x) if x_in_db else x
    xs, ys = _grid(xx, y, dt_s)
    if differences:
        xs, ys = xs.diff(), ys.diff()
    rows = []
    kmax = int(max_lag_s // dt_s)
    for k in range(-kmax, kmax + 1):
        yy = ys.shift(-k)
        m = xs.notna() & yy.notna()
        if m.sum() < 5:
            rows.append((k * dt_s, np.nan, np.nan, int(m.sum())))
            continue
        rows.append((k * dt_s, float(np.corrcoef(xs[m], yy[m])[0, 1]),
                     float(stats.spearmanr(xs[m], yy[m]).statistic), int(m.sum())))
    scan = pd.DataFrame(rows, columns=["lag_s", "pearson_r", "spearman_rho", "n"])
    if scan.pearson_r.notna().sum() == 0:
        return dict(n=0), scan
    j = int(scan.pearson_r.abs().idxmax())
    k = int(round(scan.lag_s[j] / dt_s))
    yy = ys.shift(-k)
    m = xs.notna() & yy.notna()
    ne, r1, r2 = n_effective(xs[m].to_numpy(), yy[m].to_numpy())
    r = scan.pearson_r[j]
    tstat = r * np.sqrt(max(ne - 2, 1) / max(1 - r * r, 1e-12))
    p = float(2 * stats.t.sf(abs(tstat), max(ne - 2, 1)))
    slope = float(np.polyfit(xs[m], yy[m], 1)[0])
    return dict(lag_s=float(scan.lag_s[j]), pearson_r=float(r), spearman_rho=float(scan.spearman_rho[j]),
                slope_y_per_dB=slope, n=int(m.sum()), n_eff=ne, lag1_autocorr_x=r1,
                lag1_autocorr_y=r2, p_value_neff=p,
                note="n_eff < 10: not significant whatever r is" if ne < 10 else ""), scan


def log_bands(f_lo=500.0, f_hi=20000.0, per_octave=3):
    """Contiguous logarithmic bands (default third-octave) as [(lo, hi), ...]."""
    n = int(np.ceil(np.log2(f_hi / f_lo) * per_octave))
    e = f_lo * 2 ** (np.arange(n + 1) / per_octave)
    return list(zip(e[:-1], np.minimum(e[1:], f_hi)))


def band_scan(features, ch, target, bands=None, dt_s=10.0, max_lag_s=0.0, baseline=None):
    """Correlate `target` (a Series: pH, dpH/dt, ...) with the level in each of many narrow
    bands (default third-octave 0.5-20 kHz) of channel ch.  baseline=(feat, (t0, t1)) turns
    every band into an EXCESS level first.  Returns a DataFrame per band: f_lo, f_hi, best
    lag, r, rho, slope, n, n_eff, p.  Many bands are tested, so expect ~5 % of them to pass
    p < 0.05 by chance: look for a coherent range of bands, not one."""
    rows = []
    for lo, hi in bands or log_bands():
        s = band_timeline(features, ch, lo, hi, dt_s)
        if s.notna().sum() < 5:
            continue
        if baseline is not None:
            bf, win = baseline
            b = band_level_from_psd(bf, ch, lo, hi)
            b0 = float(b[(b.index >= win[0]) & (b.index < win[1])].mean())
            s = excess(s, b0)
        res, _ = correlate(s, target, dt_s, max_lag_s)
        if res.get("n", 0):
            rows.append(dict(f_lo=lo, f_hi=hi, f_centre=np.sqrt(lo * hi), **res))
    return pd.DataFrame(rows)


def event_count_timeline(features, ch, band="audio", bin_s=10.0, min_env_db=None):
    """Trigger RATE per live second on the lab clock, optionally only triggers louder than
    min_env_db (a crude energy class): lets you correlate 'big events' separately."""
    parts = []
    for f in features:
        if int(ch) not in f.channels or band not in f.meta["detect_bands"]:
            continue
        ev = f.events
        if min_env_db is not None:
            ev = ev[ev.env_db >= min_env_db]
        t, r, _, _ = f.event_rate(ch, band, bin_s, events=ev)
        parts.append(pd.Series(r, index=f.abs_time(t)))
        parts.append(pd.Series([np.nan], index=f.abs_time([f.duration_s + bin_s])))
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts).sort_index().rename(f"Ch{ch} {band} trigger rate")
