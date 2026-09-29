"""Glide-interpretation physics against closed forms, and the view functions against
synthetic signals."""
import numpy as np
import pytest

import zoom_acoustics as za
from zoom_acoustics import physics as P, glide as G, views as V, plots as zp


def test_wood_limits_and_values():
    # beta -> 0 recovers water; the gas-dominated limit c^2 -> kappa P0 / (rho beta (1-beta))
    assert abs(P.wood_sound_speed(0.0) - P.C_W) < 1e-9
    b = 1e-2
    approx = np.sqrt(P.GAMMA_CO2 * P.P_ATM / (P.RHO_W * b * (1 - b)))
    assert abs(P.wood_sound_speed(b) / approx - 1) < 0.02
    # the values quoted in the docs
    assert abs(P.wood_sound_speed(1e-4) - 909) < 2 and abs(P.wood_sound_speed(1e-3) - 354) < 2
    # control: the old analysis4 docstring said 1e-4 gives ~300 m/s; it does not
    assert P.wood_sound_speed(1e-4) > 600


def test_layer_mode_inversion_roundtrip_and_limits():
    b = np.array([1e-5, 1e-4, 1e-3, 1e-2])
    f = P.layer_mode(b, 0.02)
    assert np.allclose(P.beta_from_layer_mode(f, 0.02), b, rtol=1e-6)
    assert abs(P.layer_mode(0.0, 0.02) - P.C_W / 0.08) < 1e-6
    assert np.isnan(P.beta_from_layer_mode(P.C_W / 0.08 * 1.1, 0.02))   # above bubble-free mode
    assert np.isnan(P.beta_from_layer_mode(10.0, 0.02))                   # needs > 2 % gas


def test_wall_and_fritz():
    assert abs(P.wall_factor(1.0, 1.0) - np.sqrt(2 / 3)) < 1e-12
    assert abs(P.wall_factor(1.0, 1e9) - 1) < 1e-9
    d = 1e-3
    R = P.fritz_departure_radius(d)
    assert abs(4 / 3 * np.pi * R ** 3 * P.RHO_W * 9.81 - np.pi * d * P.SIGMA_W) < 1e-12
    assert abs(P.capillary_length() - 2.727e-3) < 1e-5


def test_interpret_growing_bubble_bond_flag():
    """A ridge f(t) from a bubble growing linearly 0.5 -> 3.5 mm: interpret must recover
    R(t) and dR/dt and flag the Bond > 1 part."""
    import pandas as pd
    t = np.arange(0, 300, 0.25)
    R = 0.5e-3 + 1e-5 * t
    tr = pd.DataFrame(dict(t_s=t, f_hz=P.minnaert_f0(R), contrast_db=20.0))
    it = G.interpret(tr, smooth_s=0.25)
    assert np.allclose(it.R_minnaert_mm, R * 1e3, rtol=1e-6)
    assert abs(np.median(it.dRdt_um_per_s) - 10.0) < 0.01
    summ = G.interpretation_summary(it)
    assert not summ.physical.iloc[0]           # free bubble exceeds the capillary length
    assert (it.R_wall_mm < it.R_minnaert_mm).all()


@pytest.fixture
def demo_like(tmp_path, cfg_factory):
    from conftest import write_array
    sr = 24000
    t = np.arange(sr * 40) / sr
    x = np.random.default_rng(0).normal(0, 1e-3, t.size)
    x += np.where(t > 15, 5e-3 * np.sin(2 * np.pi * 3000 * t), 0)          # tone from 15 s
    (tmp_path / "audio").mkdir()
    write_array(tmp_path / "audio" / "V_001.WAV", x[:, None].astype(np.float32), sr)
    cfg = cfg_factory(dsp=dict(bands=[["low", 20, 1200], ["audio", 1200, 11000]], psd_nperseg=1024),
                      channels={1: {"name": "H", "sensor": "hydrophone", "pa_per_fs": 10.0}})
    take, = za.discover_takes(cfg.data_dir, verbose=False)
    za.process_take(take, cfg, verbose=False)
    return cfg, take, za.load_features(cfg, "V_001")


def test_difference_image_and_calibration(demo_like):
    cfg, take, f = demo_like
    te, fe, D, lab, sym = zp.spectrogram_db(f, 1, background=(2, 12))
    fc = 0.5 * (fe[:-1] + fe[1:])
    tc = 0.5 * (te[:-1] + te[1:])
    row = np.argmin(abs(fc - 3000))
    # background part ~ 0 dB, tone part strongly positive
    assert abs(np.nanmedian(D[:, tc < 12])) < 1.0
    assert np.nanmedian(D[row, tc > 20]) > 30 and sym
    # calibration: pa_per_fs = 10 -> +20 dB + 120 dB re 1 uPa
    _, _, Dfs, _, _ = zp.spectrogram_db(f, 1)
    _, _, Dpa, lab_pa, _ = zp.spectrogram_db(f, 1, units="pa")
    assert np.allclose(Dpa - Dfs, 140.0) and "µPa" in lab_pa


def test_suggest_zooms_and_overview(demo_like):
    cfg, take, f = demo_like
    z = V.suggest_zooms(f, 1, baseline=(2, 12), fmax=11000)
    names = [d["name"] for d in z]
    assert "onset" in names
    on = next(d for d in z if d["name"] == "onset")
    assert on["t0"] < 15 < on["t1"]
    fig = V.plot_overview_zoom(f, 1, z, background=(2, 12), take=take, fmax=11000)
    assert len(fig.axes) >= 3 + len(z)
