"""Regression tests for the findings of the independent code review (2026-09-29).
Each test reproduces the failing scenario the reviewer found and asserts the fixed behaviour."""
import datetime as dt
import time

import numpy as np
import pandas as pd
import pytest

import zoom_acoustics as za
from zoom_acoustics import io, masks, ph, dsp
from conftest import write_array, START

SR = 8000


# ---------------------------------------------------------------- pH parsing (findings 1-3, 7, 12)
def test_ambiguous_day_month_is_refused(tmp_path):
    """dd/mm crossing the 12th -> 13th used to be read half month-first, half day-first and
    silently reordered."""
    p = tmp_path / "a.csv"
    p.write_text("Date,Time,pH\n12/09/2026,23:59:30,1.0\n12/09/2026,23:59:45,1.1\n"
                 "13/09/2026,00:00:00,1.2\n")
    with pytest.raises(ValueError, match="ambiguous"):
        ph.load_ph(p)
    d = ph.load_ph(p, dayfirst=True)
    assert list(d.index) == [pd.Timestamp("2026-09-12 23:59:30"), pd.Timestamp("2026-09-12 23:59:45"),
                             pd.Timestamp("2026-09-13 00:00:00")]
    assert list(d.pH) == [1.0, 1.1, 1.2]


def test_timezone_aware_keeps_wall_clock(tmp_path):
    p = tmp_path / "b.csv"
    p.write_text("timestamp,pH\n2026-09-10T15:33:20-04:00,2.0\n2026-09-10T15:33:30-04:00,2.1\n")
    d = ph.load_ph(p)
    assert d.index[0] == pd.Timestamp("2026-09-10 15:33:20")


def test_time_only_column_needs_a_date(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("time,pH\n23:59:50,1.0\n00:00:10,1.1\n")
    with pytest.raises(ValueError, match="times of day only"):
        ph.load_ph(p)
    d = ph.load_ph(p, base_date="2026-10-01")
    assert list(d.index) == [pd.Timestamp("2026-10-01 23:59:50"), pd.Timestamp("2026-10-02 00:00:10")]


def test_elapsed_start_is_lab_time_offset_not_applied_twice(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("t_min,pH\n0,7\n1,6\n")
    d = ph.load_ph(p, elapsed_start="2026-10-01 10:00:00", clock_offset_s=-3600)
    assert d.index[0] == pd.Timestamp("2026-10-01 10:00:00")
    d2 = ph.load_ph(p, elapsed_start="2026-10-01 10:00:00", elapsed_unit="minutes")
    assert d2.index[1] == pd.Timestamp("2026-10-01 10:01:00")


def test_decimal_comma(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("timestamp;pH;Temp\n2026-10-01 10:00:00;7,01;21,5\n2026-10-01 10:00:10;6,95;21,6\n")
    d = ph.load_ph(p)
    assert np.allclose(d.pH, [7.01, 6.95]) and np.allclose(d.Temp, [21.5, 21.6])


# ---------------------------------------------------------------- grouping (findings 4-6, 13)
def _tone(n=SR, ch=1):
    return np.random.default_rng(0).normal(0, 0.01, (n, ch)).astype(np.float32)


def test_four_digit_take_numbers_are_not_split_parts(tmp_path):
    write_array(tmp_path / "EXP_0011.WAV", _tone(), SR, start=START)
    write_array(tmp_path / "EXP_0012.WAV", _tone(), SR, start=START + dt.timedelta(hours=2))
    write_array(tmp_path / "260910_0011.WAV", _tone(), SR, start=START + dt.timedelta(hours=3))
    names = sorted(t.name for t in io.discover_takes(tmp_path, verbose=False))
    assert names == ["260910_0011", "EXP_0011", "EXP_0012"]


def test_contiguous_split_parts_still_join(tmp_path):
    write_array(tmp_path / "T_008_0001.WAV", _tone(SR * 2), SR, start=START)
    write_array(tmp_path / "T_008_0002.WAV", _tone(SR), SR, start=START + dt.timedelta(seconds=2))
    take, = io.discover_takes(tmp_path, verbose=False)
    assert take.name == "T_008" and len(take.parts) == 2 and take.duration_s == 3.0


def test_mono_split_with_part_before_track(tmp_path):
    for part, t0 in ((1, 0), (2, 1)):
        for tr in (1, 2):
            write_array(tmp_path / f"ZOOM0001_{part:04d}_Tr{tr}.WAV", _tone(), SR,
                        start=START + dt.timedelta(seconds=t0))
    take, = io.discover_takes(tmp_path, verbose=False)
    assert take.name == "ZOOM0001" and take.channel_numbers == [1, 2] and len(take.parts) == 2


def test_poly_next_to_mix_file_is_one_based(tmp_path):
    write_array(tmp_path / "260910_011.WAV", _tone(ch=4), SR)
    write_array(tmp_path / "260910_011_LR.WAV", _tone(ch=2), SR)
    take, = io.discover_takes(tmp_path, verbose=False)
    assert take.channel_numbers == [1, 2, 3, 4]


def test_same_name_in_two_folders(tmp_path, cfg_factory):
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        write_array(tmp_path / sub / "ZOOM0001.WAV", _tone(SR * 2), SR)
    takes = io.discover_takes(tmp_path, recursive=True, verbose=False)
    assert sorted(t.name for t in takes) == ["a__ZOOM0001", "b__ZOOM0001"]
    cfg = cfg_factory(data_dir=str(tmp_path))
    za.process_all(takes, cfg, verbose=False)
    assert all((cfg.cache_path / t.name / "meta.json").exists() for t in takes)


def test_mono_track_names_from_ixml(tmp_path):
    from zoom_acoustics.demo import write_bwf
    x = _tone()
    for tr in (1, 2):
        # iXML lists all tracks, as the recorder writes it for every mono file
        write_bwf(tmp_path / f"S_001_Tr{tr}.WAV", SR, 1, SR, lambda f0, n: x[f0:f0 + n], START,
                  track_names=["HydA", "HydB"])
    take, = io.discover_takes(tmp_path, verbose=False)
    assert take.track_names == ["HydA", "HydB"]


# ---------------------------------------------------------------- masks and live time (8, 9, 14)
def test_mask_array_matches_naive_and_is_fast():
    rng = np.random.default_rng(1)
    w = [(a, a + rng.uniform(0.1, 3)) for a in np.sort(rng.uniform(0, 1000, 500))]
    t = rng.uniform(-5, 1005, 20000)
    naive = np.zeros(t.size, bool)
    for a, b in w:
        naive |= (t >= a) & (t < b)
    assert np.array_equal(masks.mask_array(t, w), naive)
    naive_live = 1000 - sum(max(0, min(b, 1000) - max(a, 0)) for a, b in masks.merge(w))
    assert abs(masks.live_seconds(0, 1000, w) - naive_live) < 1e-9
    per = masks.periodic_windows(1.0, 10.0, 6 * 3600)                 # 6 h of chirps
    tt = np.arange(0, 6 * 3600, 5e-4)
    t0 = time.time()
    masks.mask_array(tt, per)
    masks.live_seconds(np.arange(0, 21600, 5.0), np.arange(5.0, 21605, 5.0), per)
    assert time.time() - t0 < 2.0


def test_event_rate_live_time_matches_detector(tmp_path, cfg_factory):
    x = np.random.default_rng(2).normal(0, 0.01, (48000 * 8, 1)).astype(np.float32)
    (tmp_path / "audio").mkdir()
    write_array(tmp_path / "audio" / "L_001.WAV", x, 48000)
    cfg = cfg_factory(takes={"L_001": {"masks": [[2.0, 3.0], [5.0, 5.01]]}})
    take, = za.discover_takes(cfg.data_dir, verbose=False)
    za.process_take(take, cfg, verbose=False)
    f = za.load_features(cfg, take.name)
    _, _, _, live = f.event_rate(1, "audio", bin_s=1.0)
    assert abs(live.sum() - f.meta["live_s"]["1:audio"]) < 3 * f.env_dt


def test_summary_honours_added_masks(tmp_path, cfg_factory):
    x = np.random.default_rng(3).normal(0, 0.01, (48000 * 6, 1)).astype(np.float32)
    x[48000 * 3:48000 * 3 + 20] += 0.5                               # one transient at 3 s
    (tmp_path / "audio").mkdir()
    write_array(tmp_path / "audio" / "M_001.WAV", x, 48000)
    cfg = cfg_factory()
    take, = za.discover_takes(cfg.data_dir, verbose=False)
    za.process_take(take, cfg, verbose=False)
    f = za.load_features(cfg, take.name)
    n0 = f.summary().n_trig_audio.iloc[0]
    f.add_masks([(2.9, 3.1)])
    assert f.summary().n_trig_audio.iloc[0] == n0 - 1


# ---------------------------------------------------------------- DSP plan (10, 11, 15)
def test_band_edges_are_the_filters_edges(tmp_path, cfg_factory):
    x = np.random.default_rng(4).normal(0, 0.01, (48000 * 2, 1)).astype(np.float32)
    (tmp_path / "audio").mkdir()
    write_array(tmp_path / "audio" / "B_001.WAV", x, 48000)
    cfg = cfg_factory(dsp=dict(bands=[["audio", 1200, 24000]]))
    take, = za.discover_takes(cfg.data_dir, verbose=False)
    p = za.plan(take, cfg)
    assert p["bands"] == [("audio", 1200.0, 24000.0)]


@pytest.mark.parametrize("sr,psd_dt", [(44100, 0.25), (44100, 0.37), (96000, 0.25)])
def test_block_s_is_honoured(tmp_path, cfg_factory, sr, psd_dt):
    x = np.zeros((sr * 2, 1), np.float32)
    (tmp_path / "audio").mkdir(exist_ok=True)
    write_array(tmp_path / "audio" / f"S{sr}.WAV", x, sr)
    cfg = cfg_factory(dsp=dict(block_s=10.0, psd_dt=psd_dt))
    take, = za.discover_takes(cfg.data_dir, verbose=False)
    p = za.plan(take, cfg)
    assert 0.5 * 10 * sr <= p["block"] <= 1.5 * 10 * sr
    assert p["block"] % p["level_hop"] == 0 and p["block"] % p["env_hop"] == 0 \
        and p["block"] % p["psd_hop"] == 0


def test_nonfinite_samples_are_reported(tmp_path, cfg_factory):
    x = np.random.default_rng(5).normal(0, 0.01, (48000 * 2, 1)).astype(np.float32)
    x[1000:1010] = np.nan
    (tmp_path / "audio").mkdir()
    write_array(tmp_path / "audio" / "N_001.WAV", x, 48000)
    cfg = cfg_factory()
    take, = za.discover_takes(cfg.data_dir, verbose=False)
    za.process_take(take, cfg, verbose=False)
    f = za.load_features(cfg, take.name)
    assert f.meta["n_nonfinite"] == [10]
    assert any("NaN" in n for n in f.meta["notes"])
