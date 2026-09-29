"""
dsp.py -- signal-processing primitives.  Every definition here is written out in THEORY.md.

Levels are mean squares of the recorded sample value, in FS^2 (FS = digital full scale).
There is no acoustic calibration, so only RATIOS within one channel are physical.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sg


# ---------------------------------------------------------------- units
def db(x, floor=1e-30):
    """10 log10 of a mean-square quantity -> dB re 1 FS^2.  NaN stays NaN; a pandas
    Series stays a Series (same index)."""
    v = 10.0 * np.log10(np.maximum(np.asarray(x, float), floor))
    if hasattr(x, "index") and hasattr(x, "to_numpy"):
        return type(x)(v, index=x.index, name=getattr(x, "name", None))
    return v


def undb(x):
    return 10.0 ** (np.asarray(x, float) / 10.0)


# ---------------------------------------------------------------- filters
def bandpass_sos(lo, hi, sr, order=4):
    """Butterworth band-pass as second-order sections.

    SOS, not (b, a): a 4th-order band-pass at 20 Hz with a 96 kHz Nyquist has normalised
    edge 2e-4 and is numerically unusable in transfer-function form.
    If lo <= 0 it becomes a low-pass; if hi >= 0.95 Nyquist it becomes a high-pass.
    """
    nyq = sr / 2.0
    lo_n = lo / nyq
    hi_n = hi / nyq
    if lo_n <= 0 and hi_n >= 0.95:
        return None                                   # broadband: no filter
    if lo_n <= 0:
        return sg.butter(order, hi_n, btype="low", output="sos")
    if hi_n >= 0.95:
        return sg.butter(order, lo_n, btype="high", output="sos")
    return sg.butter(order, [lo_n, hi_n], btype="band", output="sos")


def usable_bands(bands, sr):
    """Drop bands that start above 0.95 Nyquist.  A band whose upper edge is at or above
    0.95 Nyquist becomes a HIGH-PASS (see bandpass_sos), so its stored upper edge is set to
    Nyquist: the edges recorded in the bundle are the edges of the filter actually used.
    Returns [(name, lo, hi)] and a list of human-readable notes about what changed."""
    nyq = sr / 2.0
    out, notes = [], []
    for name, lo, hi in bands:
        if lo >= 0.95 * nyq:
            notes.append(f"band {name} ({lo:g}-{hi:g} Hz) dropped: above Nyquist ({nyq:g} Hz)")
            continue
        if hi >= 0.95 * nyq and hi != nyq:
            notes.append(f"band {name}: upper edge {hi:g} Hz is at/above 0.95 Nyquist, so it is "
                         f"a high-pass from {lo:g} Hz to Nyquist ({nyq:g} Hz)")
            hi = nyq
        out.append((name, float(lo), float(hi)))
    return out, notes


# ---------------------------------------------------------------- spectra
def band_power(P, f, lo, hi):
    """Integrate a one-sided PSD [FS^2/Hz] over [lo, hi) -> mean square [FS^2].

    RECTANGLE rule on purpose.  A Welch PSD is a per-bin density and the discrete Parseval
    relation is mean(x^2) = sum_k P_k df.  The trapezoid rule halves the two end bins and
    under-integrates by ~1/N of the band: 0.2 dB for a 21-bin band.  (Found and fixed twice
    in the earlier Sep-2026 pipelines.)
    """
    f = np.asarray(f, float)
    m = (f >= lo) & (f < hi)
    if not m.any():
        return np.zeros(np.shape(P)[:-1]) if np.ndim(P) > 1 else 0.0
    df = float(np.median(np.diff(f)))
    return np.sum(np.asarray(P)[..., m], axis=-1) * df


def auto_nperseg(sr, target_df=23.4):
    """Power-of-two segment length giving a bin width close to target_df Hz
    (8192 at 192 kHz, 4096 at 96 kHz, 2048 at 48 kHz)."""
    return int(2 ** np.round(np.log2(sr / target_df)))


def line_q(f, p, f_lo, f_hi):
    """Peak frequency and half-power bandwidth Q of the strongest line in [f_lo, f_hi).
    `q_resolved` is False when the half-power width is under 3 bins: then Q is a property
    of the FFT length, not of the line."""
    f = np.asarray(f, float)
    m = (f >= f_lo) & (f < f_hi)
    if m.sum() < 5:
        return None
    ff, pp = f[m], np.asarray(p, float)[m]
    j = int(np.argmax(pp))
    half = pp[j] / 2.0
    lo = j
    while lo > 0 and pp[lo] > half:
        lo -= 1
    hi = j
    while hi < len(pp) - 1 and pp[hi] > half:
        hi += 1
    bw = float(ff[hi] - ff[lo])
    df = float(np.median(np.diff(ff)))
    return dict(f_peak_hz=float(ff[j]), bw_hz=bw, Q=(ff[j] / bw if bw > 0 else np.nan),
                df_hz=df, q_resolved=bool(bw > 3 * df))


# ---------------------------------------------------------------- running windows
def boxcount(x, n):
    """Running SUM over n samples of a 0/1 array, 'valid' alignment: out[k] = sum x[k:k+n].
    Exact integers (cumsum of 0/1 values in float64 is exact below 2**53)."""
    x = np.asarray(x, float)
    n = int(n)
    if n <= 1:
        return x.copy()
    c = np.cumsum(np.r_[0.0, x])
    return c[n:] - c[:-n]


def find_onset(t, level, base_window, rise_db=6.0, hold_s=10.0, frac=0.8, t_min=None):
    """First time the level rises `rise_db` above the median of `base_window` AND at least
    `frac` of the next `hold_s` seconds stay above it.

    The hold requirement is what rejects a single impulse (dropping the sample, a knock on
    the bench); a bare threshold crossing fires on those.  NaN samples (masked) are neither
    above nor below: the fraction is over valid samples only, and at least half of the hold
    window must be valid.

    Returns (t_onset or None, baseline mean square).
    """
    t = np.asarray(t, float)
    y = np.asarray(level, float)
    ok = np.isfinite(y)
    bm = ok & (t >= base_window[0]) & (t < base_window[1])
    if bm.sum() < 3:
        return None, np.nan
    base = float(np.median(y[bm]))
    if base <= 0:
        return None, base
    dt = float(np.median(np.diff(t)))
    hold = max(1, int(round(hold_s / dt)))
    if hold >= len(y):
        return None, base
    above = np.where(ok, db(np.where(ok, y, base) / base) > rise_db, False).astype(float)
    n_ok = boxcount(ok.astype(float), hold)
    n_above = boxcount(above, hold)
    good = (n_ok > 0.5 * hold) & (n_above > frac * n_ok) & (above[:len(n_ok)] > 0)
    if t_min is not None:
        good &= t[:len(n_ok)] >= t_min
    else:
        good &= t[:len(n_ok)] >= base_window[1]
    idx = np.flatnonzero(good)
    return (float(t[idx[0]]) if idx.size else None), base


# ---------------------------------------------------------------- STA/LTA
class StaLta:
    """Streaming STA/LTA trigger on a mean-square envelope e_k sampled every `dt` seconds.

        STA_k = mean of the finite values in e[k-na+1 .. k]      (na = sta_s/dt)
        LTA_k = mean of the finite values in e[k-nl+1 .. k]      (nl = lta_s/dt)
        r_k   = STA_k / LTA_k,  NaN if either window is at most half finite

    BOTH windows END at sample k.  The Sep-2026 pipeline once truncated the two running
    means to a common start index instead, which compared sample k with the 60 ms AFTER it
    and made every trigger 59.5 ms late; tests/test_events.py keeps that bug out.

    Trigger logic (hysteresis): the detector is ARMED; it FIRES at the first k with
    r_k > on and at least `dead_s` since the previous trigger; it then stays DISARMED until
    r drops below `off`.  One excursion above `on` therefore gives one trigger, however long
    it lasts.

    Feed blocks with process(); the result is identical to one call on the whole array
    (tested).  NaN samples in e (masked time) are skipped and count as dead time.
    """

    def __init__(self, dt, sta_s=5e-4, lta_s=6e-2, on=8.0, off=2.0, dead_s=1.5e-3):
        self.dt = float(dt)
        self.na = max(1, int(round(sta_s / dt)))
        self.nl = max(self.na + 1, int(round(lta_s / dt)))
        self.on, self.off = float(on), float(off)
        self.dead = max(1, int(round(dead_s / dt)))
        self.params = dict(dt=self.dt, sta_s=self.na * dt, lta_s=self.nl * dt, on=on, off=off,
                           dead_s=self.dead * dt)
        self._tail = np.full(self.nl - 1, np.nan)     # previous nl-1 samples
        self._n_seen = 0                              # global index of next sample
        self._armed = True
        self._last = -(10 ** 15)
        self.n_valid = 0                              # samples with a finite ratio

    def ratio(self, e):
        """Ratio for the samples of e, using and updating the carried tail."""
        e = np.asarray(e, float)
        ext = np.r_[self._tail, e]
        ok = np.isfinite(ext)
        v = np.where(ok, ext, 0.0)
        cs = np.r_[0.0, np.cumsum(v)]
        ck = np.r_[0.0, np.cumsum(ok)]
        end = np.arange(self.nl - 1, len(ext)) + 1           # window end (exclusive) in cs
        s_sta = cs[end] - cs[end - self.na]
        k_sta = ck[end] - ck[end - self.na]
        s_lta = cs[end] - cs[end - self.nl]
        k_lta = ck[end] - ck[end - self.nl]
        with np.errstate(invalid="ignore", divide="ignore"):
            sta = np.where(k_sta > 0.5 * self.na, s_sta / np.maximum(k_sta, 1), np.nan)
            lta = np.where(k_lta > 0.5 * self.nl, s_lta / np.maximum(k_lta, 1), np.nan)
            r = sta / lta
        r[~np.isfinite(e)] = np.nan
        self._tail = ext[-(self.nl - 1):] if self.nl > 1 else np.zeros(0)
        return r

    def process(self, e):
        """Consume one block of envelope samples.  Returns (global_indices, ratio_at_trigger)."""
        r = self.ratio(e)
        base = self._n_seen
        self._n_seen += len(r)
        fin = np.isfinite(r)
        self.n_valid += int(fin.sum())
        on_idx = np.flatnonzero(fin & (r > self.on))
        off_idx = np.flatnonzero(fin & (r < self.off))
        trig = []
        pos = 0
        n = len(r)
        while pos < n:
            if self._armed:
                earliest = max(pos, self._last + self.dead - base)
                j = np.searchsorted(on_idx, earliest)
                if j >= len(on_idx):
                    break
                k = int(on_idx[j])
                trig.append(k)
                self._last = base + k
                self._armed = False
                pos = k + 1
            else:
                j = np.searchsorted(off_idx, pos)
                if j >= len(off_idx):
                    break
                self._armed = True
                pos = int(off_idx[j]) + 1
        trig = np.asarray(trig, int)
        return base + trig, (r[trig] if trig.size else np.zeros(0))


def detect(e, dt, chunk=1_000_000, **kw):
    """Run StaLta over a whole envelope array.  Returns (t_trigger_s, ratio, live_s).
    Trigger time is the CENTRE of the envelope sample, (k + 0.5) * dt."""
    d = StaLta(dt, **kw)
    ks, rs = [], []
    for i in range(0, len(e), chunk):
        k, r = d.process(e[i:i + chunk])
        ks.append(k)
        rs.append(r)
    k = np.concatenate(ks) if ks else np.zeros(0, int)
    r = np.concatenate(rs) if rs else np.zeros(0)
    return (k + 0.5) * dt, r, d.n_valid * dt
