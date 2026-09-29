"""pH loading/alignment and bubble physics."""
import numpy as np
import pandas as pd
import pytest

from zoom_acoustics import ph, physics


def test_load_formats(tmp_path):
    t = pd.date_range("2026-10-01 10:00:00", periods=6, freq="30s")
    vals = [7.0, 6.5, 3.0, 2.0, 1.5, 1.4]
    # 1. single timestamp column
    pd.DataFrame({"Timestamp": t.strftime("%Y-%m-%d %H:%M:%S"), "pH": vals}).to_csv(tmp_path / "a.csv", index=False)
    # 2. meter style Date + Time, semicolon separated, extra column
    pd.DataFrame({"Date": t.strftime("%d/%m/%Y"), "Time": t.strftime("%H:%M:%S"),
                  "pH": vals, "Temp (C)": 21.0}).to_csv(tmp_path / "b.csv", index=False, sep=";")
    # 3. elapsed minutes
    pd.DataFrame({"t_min": np.arange(6) * 0.5, "pH value": vals}).to_csv(tmp_path / "c.csv", index=False)
    # 4. excel
    pd.DataFrame({"time": t, "pH": vals}).to_excel(tmp_path / "d.xlsx", index=False)
    a = ph.load_ph(tmp_path / "a.csv")
    b = ph.load_ph(tmp_path / "b.csv", dayfirst=True)
    c = ph.load_ph(tmp_path / "c.csv", elapsed_start="2026-10-01 10:00:00")
    d = ph.load_ph(tmp_path / "d.xlsx")
    for df in (a, b, c, d):
        assert list(df.index) == list(t), df.index
        assert np.allclose(df["pH"], vals)
    assert "Temp (C)" in b.columns
    e = ph.load_ph(tmp_path / "a.csv", clock_offset_s=-90)
    assert e.index[0] == t[0] - pd.Timedelta(seconds=90)
    with pytest.raises(ValueError):
        ph.load_ph(tmp_path / "c.csv")                    # elapsed without a start


def test_acoustic_at_linear_power_mean():
    """Window mean is taken in linear power: two samples at 0 dB and 10 dB average to
    10 log10(5.5) = 7.40 dB, NOT 5 dB."""
    idx = pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:00:01"])
    s = pd.Series([1.0, 10.0], index=idx)
    v = ph.acoustic_at(pd.to_datetime(["2026-01-01 00:00:00.5"]), s, window_s=4)
    assert abs(v.iloc[0] - 10 * np.log10(5.5)) < 1e-9
    none = ph.acoustic_at(pd.to_datetime(["2026-01-01 01:00:00"]), s, window_s=4)
    assert np.isnan(none.iloc[0])


def test_lagged_correlation_recovers_known_lag():
    idx = pd.date_range("2026-01-01", periods=600, freq="10s")
    x = pd.Series(np.random.default_rng(0).normal(size=600), index=idx).rolling(5, min_periods=1).mean()
    y = x.shift(6)                                         # y lags x by 60 s
    c = ph.lagged_correlation(x, y, dt_s=10, max_lag_s=200)
    assert c.lag_s.iloc[c.r.idxmax()] == 60.0


def test_minnaert_limits():
    R = 3e-2
    closed = np.sqrt(3 * physics.GAMMA_CO2 * physics.P_ATM / physics.RHO_W) / (2 * np.pi * R)
    assert abs(physics.minnaert_f0(R) / closed - 1) < 1e-4
    f = np.logspace(2, np.log10(9e4), 30)
    assert np.max(np.abs(physics.minnaert_f0(physics.minnaert_R(f)) / f - 1)) < 1e-12
    assert abs(physics.minnaert_f0(1e-2) * 1e-2 - 3.17) < 0.02
    # control: forgetting the factor 3 (kappa instead of 3 kappa) is rejected by the limit
    wrong = np.sqrt(physics.GAMMA_CO2 * physics.P_ATM / physics.RHO_W) / (2 * np.pi * R)
    assert abs(wrong / closed - 1) > 0.4
