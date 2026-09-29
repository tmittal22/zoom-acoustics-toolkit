"""Band integration, filters, onset.  Each gating test has a wrong-implementation control."""
import numpy as np
from scipy import signal as sg

from zoom_acoustics import dsp

SR = 48000


def _white_psd(seed=0, n=SR * 20, sigma=0.1):
    x = np.random.default_rng(seed).normal(0, sigma, n)
    f, P = sg.welch(x, SR, nperseg=2048, detrend=False)
    return f, P, sigma


def test_band_power_parseval_narrow_band():
    """White noise of variance s^2 has PSD s^2/(fs/2); a band [lo, hi) of N bins holds
    s^2 * N df / (fs/2).  The rectangle rule must hit it to < 0.05 dB for a 21-bin band.
    CONTROL: the trapezoid rule on the same bins misses by ~1/N (0.2 dB) and must FAIL."""
    f, P, s = _white_psd()
    df = f[1] - f[0]
    lo, hi = 5000.0, 5000.0 + 21 * df
    nbin = int(np.sum((f >= lo) & (f < hi)))
    expect = s ** 2 * nbin * df / (SR / 2)
    got = dsp.band_power(P, f, lo, hi)
    err = abs(dsp.db(got) - dsp.db(expect))
    assert err < 0.05, err
    m = (f >= lo) & (f < hi)
    trap = np.trapezoid(P[m], f[m])
    assert abs(dsp.db(trap) - dsp.db(expect)) > 0.15       # the control really is wrong


def test_band_power_sine():
    """A sine of amplitude A has mean square A^2/2 (Hann-windowed leakage stays in band)."""
    t = np.arange(SR * 5) / SR
    A = 0.3
    x = A * np.sin(2 * np.pi * 5612.0 * t)
    f, P = sg.welch(x, SR, nperseg=2048, detrend=False)
    got = dsp.band_power(P, f, 5400, 5900)
    assert abs(got / (A ** 2 / 2) - 1) < 0.01


def test_streaming_filter_equals_whole_signal():
    """Carrying SOS state across blocks reproduces the one-shot filter to rounding.
    CONTROL: resetting the state every block leaves a transient at each block start."""
    x = np.random.default_rng(3).normal(0, 1, SR * 3)
    sos = dsp.bandpass_sos(20, 1200, SR)
    whole = sg.sosfilt(sos, x)
    zi = np.zeros((sos.shape[0], 2))
    parts, reset = [], []
    for i in range(0, len(x), 7001):
        y, zi = sg.sosfilt(sos, x[i:i + 7001], zi=zi)
        parts.append(y)
        reset.append(sg.sosfilt(sos, x[i:i + 7001]))
    assert np.max(np.abs(np.concatenate(parts) - whole)) < 1e-12
    assert np.max(np.abs(np.concatenate(reset) - whole)) > 1e-3


def test_usable_bands_nyquist():
    b, notes = dsp.usable_bands([("a", 100, 1000), ("b", 20000, 30000), ("c", 30000, 40000)], 48000)
    assert [x[0] for x in b] == ["a", "b"] and b[1][2] == 24000.0   # high-pass to Nyquist
    assert dsp.bandpass_sos(b[1][1], b[1][2], 48000) is not None
    assert len(notes) == 2


def test_onset_rejects_single_impulse():
    """A 1-sample spike at 20 s must not be an onset; a sustained +10 dB step at 50 s is."""
    t = np.arange(0, 100, 0.25) + 0.125
    y = np.ones_like(t)
    y[t == 20.125] = 1000.0
    y[t >= 50] = 10.0
    on, base = dsp.find_onset(t, y, (1, 15), rise_db=6, hold_s=10)
    assert base == 1.0 and abs(on - 50.125) < 1e-9
    # control: with no hold requirement the spike is (wrongly) picked
    on2, _ = dsp.find_onset(t, y, (1, 15), rise_db=6, hold_s=0.25, frac=0.0)
    assert abs(on2 - 20.125) < 1e-9


def test_line_q_resolution_flag():
    f = np.arange(0, 10000, 23.4)
    p = np.exp(-0.5 * ((f - 5600) / 5.0) ** 2)             # narrower than a bin
    r = dsp.line_q(f, p, 5000, 6000)
    assert not r["q_resolved"]
