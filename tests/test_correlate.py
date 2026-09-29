"""Correlation tools: effective sample size, lag recovery, narrow-band levels from the PSD."""
import numpy as np
import pandas as pd

import zoom_acoustics as za
from zoom_acoustics import correlate as R


def test_n_eff_white_vs_smooth():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=500), rng.normal(size=500)
    ne, _, _ = R.n_effective(x, y)
    assert ne > 450                                        # white noise: n_eff ~ n
    xs = pd.Series(x).rolling(20, min_periods=1).mean().to_numpy()
    ys = pd.Series(y).rolling(20, min_periods=1).mean().to_numpy()
    ne2, r1, r2 = R.n_effective(xs, ys)
    assert ne2 < 60 and r1 > 0.9                           # smooth: ~10x fewer independent pairs


def test_spurious_correlation_of_two_trends_is_not_significant():
    """Two unrelated smooth trends correlate strongly; the n_eff p-value must not be tiny.
    CONTROL: the naive p-value with n = 200 would be ~1e-50."""
    idx = pd.date_range("2026-01-01", periods=200, freq="10s")
    rng = np.random.default_rng(1)
    a = pd.Series(np.cumsum(rng.normal(size=200)), index=idx)
    b = pd.Series(np.cumsum(rng.normal(size=200)), index=idx)
    res, _ = R.correlate(10 ** (a / 10), b, dt_s=10)
    from scipy import stats
    naive = stats.pearsonr(a, b).pvalue
    assert res["n_eff"] < 40 and res["p_value_neff"] > 100 * naive


def test_correlate_recovers_lag_and_slope():
    idx = pd.date_range("2026-01-01", periods=600, freq="10s")
    rng = np.random.default_rng(2)
    level_db = pd.Series(np.convolve(rng.normal(size=600), np.ones(8) / 8, "same") * 10 - 60, index=idx)
    ph = (-0.1 * level_db).shift(3) + rng.normal(0, 0.02, 600)      # pH lags level by 30 s
    res, scan = R.correlate(10 ** (level_db / 10), ph.dropna(), dt_s=10, max_lag_s=100)
    assert res["lag_s"] == 30.0 and abs(res["slope_y_per_dB"] + 0.1) < 0.01 and res["pearson_r"] < -0.95


def test_ph_rate():
    idx = pd.date_range("2026-01-01", periods=100, freq="15s")
    ph = pd.DataFrame({"pH": 1 + 0.02 * np.arange(100) / 4}, index=idx)      # 0.02 per minute
    r = R.ph_rate(ph, smooth_s=90)
    assert np.allclose(r.dropna(), 0.02, atol=1e-9)


def test_band_level_from_psd_matches_tone(tmp_path, cfg_factory):
    from conftest import write_array
    sr = 24000
    t = np.arange(sr * 6) / sr
    x = 0.2 * np.sin(2 * np.pi * 3000 * t)
    (tmp_path / "audio").mkdir()
    write_array(tmp_path / "audio" / "B_001.WAV", x[:, None].astype(np.float32), sr)
    cfg = cfg_factory(dsp=dict(bands=[["audio", 1000, 10000]], psd_nperseg=1024))
    take, = za.discover_takes(cfg.data_dir, verbose=False)
    za.process_take(take, cfg, verbose=False)
    f = za.load_features(cfg, "B_001")
    s = R.band_level_from_psd(f, 1, 2800, 3200)
    assert abs(np.median(s) / 0.02 - 1) < 0.01            # A^2/2 = 0.02 FS^2
    s0 = R.band_level_from_psd(f, 1, 5000, 6000)
    assert np.median(s0) < 1e-6
