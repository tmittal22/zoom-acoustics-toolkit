"""
timeline.py -- put many takes on one lab-clock time axis ("how does it change over hours
and days").  Each take contributes its own masked, binned series; gaps between takes stay
gaps (NaN), they are never interpolated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import dsp


def level_timeline(features, ch, band="audio", dt=10.0, name=None) -> pd.Series:
    """Linear mean square [FS^2] of one channel/band, binned to dt seconds, indexed by
    lab-clock datetime, across every take that HAS that channel.  Takes without the
    channel are skipped (a 2-channel day and a 4-channel day can share one plot)."""
    parts = []
    for f in features:
        if int(ch) not in f.channels or band not in f.bands:
            continue
        if f.start is None:
            raise ValueError(f"{f.take}: no start time in WAV header; cannot place on timeline")
        t, y, _ = f.level_series(ch, band, dt=dt)
        s = pd.Series(y, index=f.abs_time(t))
        parts.append(s)
        # an explicit NaN just after each take keeps line plots from bridging the gap
        parts.append(pd.Series([np.nan], index=f.abs_time([f.duration_s + dt])))
    if not parts:
        return pd.Series(dtype=float, name=name or f"Ch{ch} {band}")
    s = pd.concat(parts).sort_index()
    s.name = name or f"Ch{ch} {band}"
    return s


def rate_timeline(features, ch, band="audio", bin_s=10.0) -> pd.Series:
    """Live-time-normalised trigger rate [1/s] across takes on the lab clock."""
    parts = []
    for f in features:
        if int(ch) not in f.channels or band not in f.meta["detect_bands"]:
            continue
        t, r, _, _ = f.event_rate(ch, band, bin_s)
        parts.append(pd.Series(r, index=f.abs_time(t)))
        parts.append(pd.Series([np.nan], index=f.abs_time([f.duration_s + bin_s])))
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts).sort_index().rename(f"Ch{ch} {band} rate")


def takes_overview(features) -> pd.DataFrame:
    rows = []
    for f in features:
        rows.append(dict(take=f.take, start=f.start,
                         end=(f.start + pd.Timedelta(seconds=f.duration_s)) if f.start else None,
                         duration_min=f.duration_s / 60, sr=f.sr,
                         channels=",".join(map(str, f.channels)),
                         n_triggers=len(f.events)))
    return pd.DataFrame(rows)


def window_table(features, windows, bands=("audio",), relative_to_first=True) -> pd.DataFrame:
    """Band level in named windows for every take and channel, e.g.
        windows = {"before": (5, 25), "after": (60, 120)}   (seconds from take start)
    Returns long-format DataFrame: take, channel, label, band, window, level_dB, and for
    every window after the first, delta_dB relative to the first window (same channel)."""
    rows = []
    names = list(windows)
    for f in features:
        for c in f.channels:
            ref = {}
            for b in bands:
                if b not in f.bands:
                    continue
                for w in names:
                    a, z = windows[w]
                    v = dsp.db(f.band_level_in_window(c, b, a, z))
                    ref.setdefault(b, v)
                    rows.append(dict(take=f.take, channel=c, label=f.label(c),
                                     sensor=f.info(c)["sensor"], band=b, window=w,
                                     level_dB=float(v),
                                     delta_dB=float(v - ref[b]) if relative_to_first else np.nan))
    return pd.DataFrame(rows)
