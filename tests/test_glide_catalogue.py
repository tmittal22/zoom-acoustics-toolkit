"""Glide tracker and event-catalogue tools against synthetic signals with known answers."""
import datetime as dt

import numpy as np
import pytest

import zoom_acoustics as za
from zoom_acoustics import catalogue as K, glide as G
from conftest import write_array

SR = 24000


def _glide_take(tmp_path, cfg_factory, glide=True, dur=90.0):
    """Noise + (optionally) a tone falling 4 kHz -> 1 kHz exponentially from t = 20 s
    to the end, with its 2nd harmonic.  True octaves over [t0, t1]: log2(f(t1)/f(t0))."""
    n = int(SR * dur)
    t = np.arange(n) / SR
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1e-3, n)
    Tg = dur - 20.0
    u = np.clip((t - 20) / Tg, 0, 1)
    lr = np.log(0.25)
    ph = 2 * np.pi * 4000 * Tg * (np.exp(u * lr) - 1) / lr
    if glide:
        x += np.where(t > 20, 3e-3 * np.sin(ph) + 1.5e-3 * np.sin(2 * ph), 0)
    (tmp_path / "audio").mkdir(exist_ok=True)
    name = "G_001" if glide else "N_001"
    write_array(tmp_path / "audio" / f"{name}.WAV", x[:, None].astype(np.float32), SR)
    cfg = cfg_factory(dsp=dict(bands=[["audio", 500, 11000]], psd_nperseg=1024))
    take = za.find_take(za.discover_takes(cfg.data_dir, verbose=False), name)
    za.process_take(take, cfg, verbose=False)
    return za.load_features(cfg, name)


def f_true(t, dur=90.0):
    return 4000 * 0.25 ** np.clip((np.asarray(t) - 20) / (dur - 20), 0, 1)


def test_ridge_tracker_recovers_known_glide(tmp_path, cfg_factory):
    f = _glide_take(tmp_path, cfg_factory)
    tr = G.track_ridge(f, 1, (2, 18), 22, seed=(2000, 6000), f_range=(300, 11000))
    # the single-pole smoother lags by tau = dt s/(1-s); compare with the lagged truth
    tau = f.psd_dt * 0.82 / 0.18
    err_lag = np.median(np.abs(tr.f_hz / f_true(tr.t_s - tau) - 1))
    err_raw = np.median(np.abs(tr.f_hz / f_true(tr.t_s) - 1))
    assert err_lag < 0.01 < err_raw, (err_lag, err_raw)
    s = G.summarize(tr)
    # f_start/f_end are medians over 3 s edge windows (centred 1.5 s in), lagged by tau
    true_oct = np.log2(f_true(s["t_end_s"] - 1.5 - tau) / f_true(s["t_start_s"] + 1.5 - tau))
    assert abs(s["octaves"] - true_oct) < 0.03 and s["significant"], (s["octaves"], true_oct)
    assert not s["free_bubble_feasible_at_end"] or s["bond_end"] < 1


def test_argmax_in_fixed_bands_fails_control(tmp_path, cfg_factory):
    """CONTROL: the naive method (argmax inside a fixed sub-band) jumps when the ridge
    leaves the band.  Its median error must be far above the tracker's."""
    f = _glide_take(tmp_path, cfg_factory)
    t, fr, D = G.excess_spectrogram(f, 1, (2, 18), (22, 90), (2000, 4000))
    naive = fr[np.argmax(D, axis=1)]
    err = np.median(np.abs(naive / f_true(t) - 1))
    assert err > 0.1, err


def test_no_ridge_is_not_significant(tmp_path, cfg_factory):
    """A tracker run on pure noise still returns a path; summarize must flag it."""
    f = _glide_take(tmp_path, cfg_factory, glide=False)
    tr = G.track_ridge(f, 1, (2, 18), 22, seed=(2000, 6000), f_range=(300, 11000))
    s = G.summarize(tr)
    assert not s["significant"] and not s["glides"]


def test_harmonic_ladder_sees_2nd_harmonic_and_detects_harmonic_lock(tmp_path, cfg_factory):
    f = _glide_take(tmp_path, cfg_factory)
    tr = G.track_ridge(f, 1, (2, 18), 22, seed=(2000, 6000), f_range=(300, 11000))
    lad = G.harmonic_ladder(f, 1, (2, 18), tr).set_index("multiple")
    ctrl = lad[lad.control].median_contrast_dB.max()
    assert lad.loc[2.0, "median_contrast_dB"] > ctrl + 6
    assert lad.loc[0.5, "median_contrast_dB"] < ctrl + 3
    # seed on the harmonic: the ladder must now show a line at 0.5x
    tr2 = G.track_ridge(f, 1, (2, 18), 22, seed=(6500, 9000), f_range=(300, 11000))
    assert np.median(tr2.f_hz / f_true(tr2.t_s)) == pytest.approx(2.0, rel=0.03)
    lad2 = G.harmonic_ladder(f, 1, (2, 18), tr2).set_index("multiple")
    assert lad2.loc[0.5, "median_contrast_dB"] > lad2[lad2.control].median_contrast_dB.max() + 6


# ---------------------------------------------------------------- catalogue
def test_interevent_poisson_and_mask_straddling():
    rng = np.random.default_rng(1)
    t = np.cumsum(rng.exponential(0.05, 20000))
    s = K.interevent_stats(t)
    assert abs(s["cv"] - 1) < 0.03 and 0.9 < s["fano_1s"] < 1.1
    # mask 1 s every 10 s: gaps across the holes must be removed, CV stays ~1
    win = [(a, a + 1.0) for a in np.arange(5, t[-1], 10)]
    tm = t[~za.masks.mask_array(t, win)]
    s2 = K.interevent_stats(tm, windows=win)
    assert abs(s2["cv"] - 1) < 0.05
    s_bad = K.interevent_stats(tm)                       # control: holes counted as gaps
    assert s_bad["cv"] > s2["cv"] + 0.3


def test_interevent_detects_bursts():
    rng = np.random.default_rng(2)
    centres = np.cumsum(rng.exponential(1.0, 500))
    t = np.sort(np.concatenate([c + rng.exponential(0.01, 10) for c in centres]))
    s = K.interevent_stats(t)
    assert s["cv"] > 2 and s["fano_1s"] > 3
    sw = K.burst_sweep(t, gaps=(0.1, 0.3))
    # bursts merge when two centres are closer than the gap (plus burst length)
    expect = 1 + np.sum(np.diff(centres) > 0.1 + 0.05)
    assert abs(sw.n_bursts.iloc[0] - expect) / expect < 0.05, (sw.n_bursts.iloc[0], expect)


def test_coincidence_excess_over_chance():
    rng = np.random.default_rng(3)
    ta = np.sort(rng.uniform(0, 1000, 3000))
    shared = ta[rng.random(ta.size) < 0.4] + 0.3e-3            # 40 % seen on B, 0.3 ms later
    tb = np.sort(np.concatenate([shared, rng.uniform(0, 1000, 3000)]))
    r = K.coincidence(ta, tb, win_s=2e-3)
    assert abs(r["excess"] - 0.4) < 0.05 and abs(r["lag_median_ms"] - 0.3) < 0.1
    r0 = K.coincidence(ta, np.sort(rng.uniform(0, 1000, 3000)), win_s=2e-3)
    assert abs(r0["excess"]) < 0.03


def test_waveform_features_and_repeater_cluster():
    sr = 48000
    tt = np.arange(int(0.012 * sr)) / sr
    rng = np.random.default_rng(4)
    W = []
    for i in range(60):                                   # 60 bubbles: Q = 10, random f
        fb = rng.uniform(2000, 8000)
        W.append(np.r_[np.zeros(96), np.exp(-np.pi * fb * tt / 10) * np.sin(2 * np.pi * fb * tt)][:tt.size])
    rat = np.exp(-tt / 6e-4) * (np.sin(2 * np.pi * 9000 * tt) + 0.6 * np.sin(2 * np.pi * 13100 * tt + 1))
    for i in range(12):                                   # 12 identical rattles, jittered
        W.append(np.roll(np.r_[np.zeros(96), rat][:tt.size], rng.integers(-5, 5)))
    W = np.array(W) + rng.normal(0, 0.01, (72, tt.size))
    ft = K.waveform_features(W, sr)
    q = ft.q_est[:60].median()
    assert 8 < q < 12, q                                  # Q = 10 put in
    C = K.xcorr_matrix(W)
    lab, summ, med = K.cluster_waveforms(C, rho_min=0.9, features=ft)
    top = summ.iloc[0]
    members = np.flatnonzero(lab == top.cluster)
    assert set(members) == set(range(60, 72)), members
