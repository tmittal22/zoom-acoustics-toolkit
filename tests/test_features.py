"""Feature bundles: known-signal levels, PSD, block invariance, channel subsets, caching."""
import datetime as dt

import numpy as np
from scipy import signal as sg

import zoom_acoustics as za
from zoom_acoustics import dsp
from conftest import write_array, START

SR = 48000


def _tone_take(tmp_path, nch=2, dur=6.0, split=None):
    t = np.arange(int(SR * dur)) / SR
    x = np.zeros((t.size, nch))
    x[:, 0] = 0.2 * np.sin(2 * np.pi * 5612.0 * t)                 # 'rig' band tone
    if nch > 1:
        x[:, 1] = np.where(t >= 3.0, 0.05, 0.005) * np.sin(2 * np.pi * 3000.0 * t)  # +20 dB step
    (tmp_path / "audio").mkdir(exist_ok=True)
    if split:
        write_array(tmp_path / "audio" / "T_001_0001.WAV", x[:split], SR)
        write_array(tmp_path / "audio" / "T_001_0002.WAV", x[split:], SR,
                    start=START + dt.timedelta(seconds=round(split / SR)))
    else:
        write_array(tmp_path / "audio" / "T_001.WAV", x, SR)
    return x


def test_levels_and_psd_of_known_tones(tmp_path, cfg_factory):
    _tone_take(tmp_path)
    cfg = cfg_factory()
    take, = za.discover_takes(cfg.data_dir)
    za.process_take(take, cfg, verbose=False)
    f = za.load_features(cfg, take.name)
    # Ch1: 0.2 amplitude tone inside 'rig' and 'audio' -> mean square 0.02
    rig = f.level(1, "rig")[4:-4]
    assert abs(np.median(rig) / 0.02 - 1) < 0.01
    assert abs(np.median(f.level(1, "broadband")) / 0.02 - 1) < 1e-3
    P, n = f.psd_mean(1, 1, 5)
    assert abs(dsp.band_power(P, f.freqs, 5400, 5900) / 0.02 - 1) < 0.01
    # Ch2: 20 dB step at 3.0 s in the audio band
    t, y, _ = f.level_series(2, "audio")
    step = dsp.db(np.median(y[t > 4])) - dsp.db(np.median(y[(t > 1) & (t < 2.5)]))
    assert abs(step - 20.0) < 0.1
    on, _ = f.onset(2, "audio", base_window=(0.5, 2.5), hold_s=1.0)
    assert abs(on - 3.0) <= f.level_dt


def test_split_take_equals_single_file(tmp_path, cfg_factory):
    """Levels and PSD from a take split mid-block match the unsplit take exactly-ish."""
    a = tmp_path / "a"; b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    _tone_take(a)
    _tone_take(b, split=SR * 2 + 12345)
    fa = fb = None
    for root in (a, b):
        cfg = cfg_factory(data_dir=str(root / "audio"), cache_dir=str(root / "cache"))
        take, = za.discover_takes(cfg.data_dir)
        za.process_take(take, cfg, verbose=False)
        if root is a:
            fa = za.load_features(cfg, take.name)
        else:
            fb = za.load_features(cfg, take.name)
    assert np.allclose(fa._lev, fb._lev, rtol=1e-6, atol=1e-12)
    assert np.allclose(fa._psd, fb._psd, rtol=1e-6, atol=1e-14)


def test_block_size_does_not_matter(tmp_path, cfg_factory):
    """block_s = 1 s and 0.5 s give the same level series (filter state carried)."""
    _tone_take(tmp_path)
    out = []
    for bs in (1.0, 0.5):
        cfg = cfg_factory(cache_dir=str(tmp_path / f"cache{bs}"), dsp=dict(block_s=bs))
        take, = za.discover_takes(cfg.data_dir)
        za.process_take(take, cfg, verbose=False)
        out.append(za.load_features(cfg, take.name)._lev[:])
    assert np.max(np.abs(out[0] - out[1])) < 1e-9


def test_channel_subset_and_cache_reuse(tmp_path, cfg_factory, capsys):
    _tone_take(tmp_path, nch=4)
    cfg = cfg_factory(takes={"T_001": {"use_channels": [2, 4]}})
    take, = za.discover_takes(cfg.data_dir)
    za.process_take(take, cfg, verbose=False)
    f = za.load_features(cfg, take.name)
    assert f.channels == [2, 4]
    za.process_take(take, cfg, verbose=True)
    assert "up to date" in capsys.readouterr().out


def test_mask_windows_remove_time(tmp_path, cfg_factory):
    _tone_take(tmp_path)
    cfg = cfg_factory(takes={"T_001": {"masks": [[1.0, 2.0]]}})
    take, = za.discover_takes(cfg.data_dir)
    za.process_take(take, cfg, verbose=False)
    f = za.load_features(cfg, take.name)
    t, y, n = f.level_series(1, "rig")
    assert np.all(np.isnan(y[(t >= 1.0) & (t < 2.0)]))
    assert np.all(np.isfinite(y[(t > 2.2) & (t < 5)]))
    _, _, _, live = f.event_rate(1, "audio", bin_s=1.0)
    assert abs(live[1]) < 1e-9 and abs(live[3] - 1.0) < 1e-9
