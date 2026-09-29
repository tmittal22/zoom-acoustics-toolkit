"""
ph.py -- load a pH (or any other scalar) time series and put it on the acoustic time axis.

Accepted inputs: .csv / .txt / .tsv / .xlsx / .xls, with EITHER
  * one timestamp column               e.g. "2026-09-10 11:33:20"
  * separate date and time columns     e.g. "10/09/2026", "11:33:20"  (common meter export)
  * an elapsed-time column + a start   e.g. "t_min" = 0, 0.5, 1.0 ... and elapsed_start
Columns are auto-detected when not named; pass names explicitly if detection guesses wrong.

CLOCKS.  Two clocks are involved: the recorder's (bext start time in the WAV) and the pH
logger's.  Neither is guaranteed to be right.  The convention here:
    lab time = recorder time + recorder_clock_offset_s       (config, top level)
    lab time = pH-logger time + ph.clock_offset_s            (config, ph section)
Measure the offsets once per session (clap / tap on the beaker at a noted wall-clock time,
or photograph both displays together) and write them into the config.
"""
from __future__ import annotations

import datetime as _dtmod
import re
from pathlib import Path

import numpy as np
import pandas as pd

_TIME_NAMES = ("timestamp", "datetime", "date_time", "date time", "time", "clock")
_ELAPSED_RE = re.compile(r"(elapsed|^t$|^t[_ (]|seconds|minutes|hours|^sec|^min|^hr)", re.I)
_UNIT_RE = {"s": re.compile(r"(^|[_ (\[])(s|sec|secs|seconds)([_ )\]]|$)", re.I),
            "min": re.compile(r"(^|[_ (\[])(m|min|mins|minutes)([_ )\]]|$)", re.I),
            "h": re.compile(r"(^|[_ (\[])(h|hr|hrs|hours)([_ )\]]|$)", re.I)}
_UNIT_S = {"s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
           "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
           "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0}


def _read_table(path, sheet=None):
    path = Path(path)
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, sheet_name=sheet or 0)
    # sniff the delimiter (comma, semicolon, tab, whitespace)
    return pd.read_csv(path, sep=None, engine="python", comment="#")


def _find_ph_col(df):
    for c in df.columns:
        if str(c).strip().lower() in ("ph", "ph value", "ph_value", "ph (-)", "ph[-]"):
            return c
    for c in df.columns:
        if re.search(r"(^|[^a-z])ph([^a-z]|$)", str(c).lower()):
            return c
    raise ValueError(f"no pH column found in {list(df.columns)}; pass ph_col=")


def _looks_like_datetime(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s):
        return False
    v = pd.to_datetime(s.astype(str).head(20), errors="coerce", format="mixed")
    return bool(v.notna().mean() > 0.8)


_TIME_ONLY = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?\s*$")
_ISO = re.compile(r"^\s*\d{4}-\d{2}-\d{2}([ T]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?)?"
                  r"\s*(Z|[+-]\d{2}:?\d{2})?\s*$")
_DECIMAL_COMMA = re.compile(r"^\s*-?\d+,\d+\s*$")


def _is_time_only(s: pd.Series) -> bool:
    v = s.dropna().head(50)
    if not len(v):
        return False
    if all(isinstance(x, _dtmod.time) for x in v):
        return True
    return bool(v.astype(str).str.match(_TIME_ONLY).mean() > 0.8)


def _parse_datetimes(txt: pd.Series, time_format=None, dayfirst=None) -> pd.Series:
    """Parse date-time strings WITHOUT per-row guessing of day/month order.

    format="mixed" decides each row independently, so a dd/mm log crossing from the 12th
    to the 13th is read partly month-first and partly day-first and silently reordered.
    Here, unless time_format or dayfirst is given, the column is parsed both ways; if the
    two readings differ on any row the order is ambiguous and ValueError asks you to set
    ph: dayfirst (or time_format) in the config.
    """
    if time_format:
        return pd.to_datetime(txt, format=time_format)
    if txt.astype(str).str.match(_ISO).all():          # YYYY-MM-DD... is unambiguous
        return pd.to_datetime(txt, format="ISO8601")
    if dayfirst is not None:
        return pd.to_datetime(txt, format="mixed", dayfirst=bool(dayfirst))
    a = pd.to_datetime(txt, format="mixed", dayfirst=False, errors="coerce")
    b = pd.to_datetime(txt, format="mixed", dayfirst=True, errors="coerce")
    ok_a, ok_b = a.notna().all(), b.notna().all()
    if ok_a and ok_b and not a.equals(b):
        raise ValueError("date order is ambiguous (e.g. 12/09 could be 12 Sep or 9 Dec): set "
                         "ph: dayfirst: true/false (or time_format) in the config")
    if ok_a:
        return a
    if ok_b:
        return b
    bad = txt[a.isna() & b.isna()].head(3).tolist()
    raise ValueError(f"cannot parse times such as {bad}; set ph: time_format")


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric columns, accepting decimal commas ('7,01') from European exports."""
    out = {}
    for c in df.columns:
        col = df[c]
        if col.dtype == object or pd.api.types.is_string_dtype(col):
            sv = col.dropna().astype(str)
            if len(sv) and sv.str.match(_DECIMAL_COMMA).mean() > 0.8:
                col = col.astype(str).str.replace(",", ".", regex=False)
        out[c] = pd.to_numeric(col, errors="coerce")
    return pd.DataFrame(out, index=df.index).dropna(axis=1, how="all")


def load_ph(path, time_col=None, ph_col=None, date_col=None, time_format=None,
            elapsed_unit=None, elapsed_start=None, clock_offset_s=0.0, sheet=None,
            dayfirst=None, base_date=None, value_name="pH", **_ignored) -> pd.DataFrame:
    """Return a DataFrame indexed by lab-clock datetime with column `value_name` plus every
    other numeric column of the file (temperature, conductivity...).

    Time sources, in order: `date_col` + `time_col`; one timestamp column; a time-of-day
    column + `base_date`; an elapsed-time column + `elapsed_start` (a LAB-clock datetime,
    so clock_offset_s is NOT applied to it).  Time-zone-aware stamps keep their wall-clock
    time (the recorder clock is a naive wall clock too).  The result is sorted; a warning is
    printed if the file was not in time order."""
    df = _read_table(path, sheet)
    df.columns = [str(c).strip() for c in df.columns]
    ph_col = ph_col or _find_ph_col(df)

    if time_col is None:
        low = {c.lower(): c for c in df.columns}
        if date_col is None and "date" in low and "time" in low:
            date_col, time_col = low["date"], low["time"]
        else:
            cands = [c for c in df.columns if c != ph_col and c != date_col]
            named = [c for c in cands if c.lower() in _TIME_NAMES or c.lower().startswith("time")]
            ela = [c for c in cands if _ELAPSED_RE.search(c) and pd.api.types.is_numeric_dtype(df[c])]
            dtc = [c for c in cands if _looks_like_datetime(df[c]) or _is_time_only(df[c])]
            time_col = (named[0] if named and (_looks_like_datetime(df[named[0]])
                                               or _is_time_only(df[named[0]])) else
                        dtc[0] if dtc else ela[0] if ela else named[0] if named else None)
        if time_col is None:
            raise ValueError(f"no time column found in {list(df.columns)}; pass time_col=")

    s = df[time_col]
    apply_offset = True
    if date_col is not None:
        d = df[date_col]
        dtxt = (d.dt.strftime("%Y-%m-%d") if pd.api.types.is_datetime64_any_dtype(d)
                else d.map(lambda x: x.strftime("%Y-%m-%d") if isinstance(x, (_dtmod.date,))
                           else str(x)).str.strip())
        ttxt = s.map(lambda x: x.strftime("%H:%M:%S.%f") if isinstance(x, _dtmod.time)
                     else str(x)).str.strip()
        t = _parse_datetimes(dtxt + " " + ttxt, time_format, dayfirst)
    elif _is_time_only(s):
        if base_date is None:
            raise ValueError(f"column {time_col!r} holds times of day only; set ph: date_col "
                             f"(a date column) or ph: base_date: YYYY-MM-DD")
        ttxt = s.map(lambda x: x.strftime("%H:%M:%S.%f") if isinstance(x, _dtmod.time)
                     else str(x)).str.strip()
        t = pd.to_datetime(str(pd.Timestamp(base_date).date()) + " " + ttxt, format="mixed")
        if (t.diff().dt.total_seconds() < -43200).any():
            print("note: time of day wraps past midnight; later rows moved to the next day")
            t = t + pd.to_timedelta((t.diff().dt.total_seconds() < -43200).cumsum(), unit="D")
    elif pd.api.types.is_numeric_dtype(s) and not _looks_like_datetime(s):
        if elapsed_start is None:
            raise ValueError(f"column {time_col!r} is elapsed time; pass elapsed_start= "
                             f"(the lab-clock datetime of t = 0)")
        unit = elapsed_unit
        if unit is None:
            unit = next((u for u, rx in _UNIT_RE.items() if rx.search(time_col)), None)
            if unit is None:
                raise ValueError(f"cannot tell the unit of {time_col!r}; pass elapsed_unit="
                                 f"'s', 'min' or 'h'")
        if unit not in _UNIT_S:
            raise ValueError(f"elapsed_unit {unit!r} not one of {sorted(_UNIT_S)}")
        t = pd.to_datetime(elapsed_start) + pd.to_timedelta(s.astype(float) * _UNIT_S[unit], unit="s")
        apply_offset = False                      # elapsed_start is already lab time
    elif pd.api.types.is_datetime64_any_dtype(s):
        t = pd.Series(s)
    else:
        t = _parse_datetimes(s.astype(str), time_format, dayfirst)
    t = pd.Series(pd.DatetimeIndex(t))
    if t.dt.tz is not None:
        t = t.dt.tz_localize(None)                # keep the wall-clock time, drop the zone
    if apply_offset:
        t = t + pd.to_timedelta(float(clock_offset_s), unit="s")

    out = _numeric(df.drop(columns=[c for c in (time_col, date_col) if c is not None]))
    if ph_col not in out.columns:
        raise ValueError(f"column {ph_col!r} has no numeric values (first raw values: "
                         f"{df[ph_col].head(3).tolist()})")
    out = out.rename(columns={ph_col: value_name})
    out.index = pd.DatetimeIndex(t.to_numpy(), name="time")
    out = out[out[value_name].notna()]
    if not out.index.is_monotonic_increasing:
        print(f"note: {Path(path).name} is not in time order; sorted "
              f"(check dayfirst / time_format if this is unexpected)")
        out = out.sort_index()
    return out


def load_ph_from_config(cfg) -> pd.DataFrame:
    p = dict(cfg["ph"])
    if not p.get("file"):
        raise ValueError("config has no ph: file:")
    return load_ph(p.pop("file"), **p)


# ---------------------------------------------------------------- alignment
def acoustic_at(times, acoustic: pd.Series, window_s=10.0, how="mean_db"):
    """Value of an acoustic series (linear mean square, datetime index) around each time in
    `times`: the mean over [t - window/2, t + window/2) in LINEAR power, returned in dB.
    NaN where the window holds no acoustic data (between takes)."""
    from .dsp import db
    a = acoustic.dropna()
    ta = a.index.values.astype("datetime64[ns]").astype(np.int64)
    va = a.to_numpy(float)
    cs = np.r_[0.0, np.cumsum(va)]
    tq = pd.DatetimeIndex(times).values.astype("datetime64[ns]").astype(np.int64)
    h = int(window_s * 1e9 / 2)
    i0 = np.searchsorted(ta, tq - h, "left")
    i1 = np.searchsorted(ta, tq + h, "left")
    n = i1 - i0
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n > 0, (cs[i1] - cs[i0]) / np.maximum(n, 1), np.nan)
    out = db(m) if how == "mean_db" else m
    return pd.Series(np.where(n > 0, out, np.nan), index=pd.DatetimeIndex(times), name=acoustic.name)


def align(ph: pd.DataFrame, acoustic: pd.Series, window_s=10.0, value="pH") -> pd.DataFrame:
    """One row per pH sample that has acoustic data within +/- window_s/2:
    columns pH, acoustic_dB, n."""
    a = acoustic_at(ph.index, acoustic, window_s)
    df = pd.DataFrame({value: ph[value].to_numpy(), "acoustic_dB": a.to_numpy()}, index=ph.index)
    return df.dropna()


def lagged_correlation(x: pd.Series, y: pd.Series, dt_s=10.0, max_lag_s=600.0, method="spearman"):
    """Correlation of y(t + lag) with x(t) on a common regular grid, for lags in
    [-max_lag, +max_lag].  Positive lag = y LAGS x.  Returns DataFrame(lag_s, r, n).

    Both series are resampled by time-averaging onto dt_s bins (no interpolation across
    gaps).  With autocorrelated series the effective sample size is much smaller than n:
    do not read a p-value off this without accounting for that.
    """
    rule = f"{int(round(dt_s * 1000))}ms"
    xs = x.resample(rule).mean()
    ys = y.resample(rule).mean()
    idx = xs.index.union(ys.index)
    xs, ys = xs.reindex(idx), ys.reindex(idx)
    rows = []
    for k in range(-int(max_lag_s // dt_s), int(max_lag_s // dt_s) + 1):
        yy = ys.shift(-k)
        m = xs.notna() & yy.notna()
        if m.sum() < 5:
            rows.append((k * dt_s, np.nan, int(m.sum())))
            continue
        r = xs[m].corr(yy[m], method=method)
        rows.append((k * dt_s, r, int(m.sum())))
    return pd.DataFrame(rows, columns=["lag_s", "r", "n"])
