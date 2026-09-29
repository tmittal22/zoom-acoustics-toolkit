"""STA/LTA detector: timing, hysteresis, block invariance, recovery of known events.

Two controls reproduce defects of the Sep-2026 analysis4 detector (a4_passive.py) and
must be REJECTED by the same assertions that the new detector passes:
  * mis-aligned running means (every trigger 59.5 ms late), and
  * re-arming immediately after a trigger, so the `off` threshold does nothing and one
    long excursion fires once per dead time.
"""
import numpy as np

from zoom_acoustics import dsp

DT = 5e-4
KW = dict(sta_s=5e-4, lta_s=6e-2, on=8.0, off=2.0, dead_s=1.5e-3)


def _noise(n, seed=0):
    """Mean square of 24 raw Gaussian samples (a 0.5 ms envelope sample at 48 kHz):
    chi^2_24 / 24 = Gamma(12, 1/12), unit mean."""
    return np.random.default_rng(seed).gamma(12.0, 1.0 / 12.0, n)


# ------------------------------------------------ controls (old a4 behaviour, reproduced)
def _a4_ratio_misaligned(e, na=1, nl=120):
    """analysis4 before its 2026-09-14 fix: both running means truncated to a common index,
    labelled with the LONG window's end."""
    c = np.cumsum(np.r_[0.0, e])
    sta = (c[na:] - c[:-na]) / na
    lta = (c[nl:] - c[:-nl]) / nl
    n = min(len(sta), len(lta))
    return np.arange(nl - 1, nl - 1 + n), sta[:n] / lta[:n]


def _a4_trigger(idx_r, r, on=8.0, off=2.0, dead=3):
    """analysis4 trigger_from_ratio: `armed` is reset to True right after firing."""
    trig, last, armed = [], -10 ** 9, True
    for i in np.flatnonzero(r > on):
        if i - last < dead or not armed:
            continue
        trig.append(idx_r[i]); last = i; armed = False
        armed = True
    return np.array(trig)


# ------------------------------------------------ tests
def test_impulse_timing_exact():
    e = _noise(4000)
    k_true = 900                                   # 450.25 ms (sample centre)
    e[k_true] += 500.0
    t, r, live = dsp.detect(e, DT, **KW)
    assert len(t) == 1
    assert abs(t[0] - (k_true + 0.5) * DT) < 1e-12
    # control: the misaligned version reports the impulse 59.5 ms late
    idx, rr = _a4_ratio_misaligned(e)
    k_bad = idx[np.argmax(rr)]
    assert (k_bad - k_true) * DT > 0.059


def test_one_trigger_per_sustained_excursion():
    """A 10 ms burst holding the ratio above `on` is one event."""
    e = _noise(6000, 1)
    e[2000:2020] += 200.0
    t, _, _ = dsp.detect(e, DT, **KW)
    assert len(t) == 1 and abs(t[0] - 2000.5 * DT) < 1e-12
    # control: immediate re-arm fires every dead time inside the burst
    c = np.cumsum(np.r_[0.0, e])
    lta = np.r_[np.full(119, np.nan), (c[120:] - c[:-120]) / 120]
    bad = _a4_trigger(np.arange(len(e)), e / lta)
    assert len(bad) >= 5


def test_block_invariance():
    """Feeding the detector in arbitrary blocks gives exactly the one-shot result."""
    rng = np.random.default_rng(5)
    e = _noise(200_000, 2)
    ks = rng.choice(len(e), 300, replace=False)
    e[ks] += rng.uniform(20, 400, ks.size)
    e[50_000:52_000] = np.nan                      # a masked stretch
    t_all, r_all, live_all = dsp.detect(e, DT, chunk=len(e), **KW)
    d = dsp.StaLta(DT, **KW)
    got = []
    i = 0
    for step in rng.integers(1, 30_000, 200):
        k, _ = d.process(e[i:i + step])
        got.append(k)
        i += step
        if i >= len(e):
            break
    if i < len(e):
        got.append(d.process(e[i:])[0])
    t_blk = (np.concatenate(got) + 0.5) * DT
    assert np.array_equal(np.round(t_blk / DT), np.round(t_all / DT))
    assert d.n_valid * DT == live_all
    assert len(t_all) > 250


def test_masked_samples_never_trigger():
    e = _noise(10_000, 3)
    e[3000:3100] += 1000.0
    e[2990:3200] = np.nan
    t, _, live = dsp.detect(e, DT, **KW)
    assert not np.any((t > 2990 * DT) & (t < 3200 * DT))
    assert live < 10_000 * DT


def test_recovery_of_poisson_events():
    """Known impulsive events on noise, rate 20/s, well separated from the 60 ms LTA:
    >= 95 % found within 1 envelope sample, false triggers < 2 %."""
    rng = np.random.default_rng(7)
    n = int(120 / DT)
    e = _noise(n, 4)
    gaps = rng.exponential(1 / 20.0, 4000)
    tt = np.cumsum(gaps + 0.004)
    tt = tt[tt < 119]
    k = (tt / DT).astype(int)
    e[k] += rng.uniform(30, 300, k.size)
    t, _, _ = dsp.detect(e, DT, **KW)
    kt = np.round(t / DT - 0.5).astype(int)
    hit = np.isin(k, kt) | np.isin(k, kt - 1) | np.isin(k, kt + 1)
    false = ~(np.isin(kt, k) | np.isin(kt, k + 1) | np.isin(kt, k - 1))
    assert hit.mean() > 0.95, hit.mean()
    assert false.mean() < 0.02, false.mean()
