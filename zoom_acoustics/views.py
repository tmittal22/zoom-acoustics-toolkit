"""
views.py -- "show me everything, then zoom in where it matters".

    zooms = za.views.suggest_zooms(f, ch=1, baseline=(3, 25))     # from the analysis
    fig = za.views.plot_overview_zoom(f, ch=1, zooms=zooms)        # full range + zoom panels
    fig = za.views.plot_overview_zoom(f, ch=1, zooms=zooms, background=(3, 25))   # difference

A zoom is a dict  {name, t0, t1, f0, f1, why}  (seconds, Hz).  Write your own list to zoom
anywhere by hand; suggest_zooms only proposes windows found by the analysis:

  baseline   the quiet reference window you give (what "nothing happening" looks like)
  onset      10 s before to 30 s after the sustained level rise (dsp.find_onset)
  loudest    the 30 s with the highest masked audio-band level
  glide      the time span and frequency range of a SIGNIFICANT tracked ridge (glide.py)
  transient  +/- 0.25 s around the strongest single trigger after the onset (detector);
             kind="raw": drawn from the RAW audio (waveform + fine STFT), because a 0.25 s
             spectrogram frame is longer than the event itself

Colour/level range: `clim=(lo, hi)` is applied to the overview and every zoom, so the same
colour means the same level everywhere (pass the same clim to other takes to compare them).
`units="pa"` shows dB re 1 uPa^2/Hz for channels with a pa_per_fs calibration.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from . import dsp
from .plots import spectrogram_db, INK, PALETTE, _marks


def suggest_zooms(feat, ch, baseline=(3.0, 25.0), band="audio", glide_seed=(4000.0, 12000.0),
                  fmax=None, include=("baseline", "onset", "loudest", "glide", "transient")):
    fmax = fmax or min(24000.0, feat.sr / 2)
    out = []
    dur = feat.duration_s
    if "baseline" in include:
        out.append(dict(name="baseline", t0=baseline[0], t1=baseline[1], f0=0.0, f1=fmax,
                        why="quiet reference window"))
    on, _ = feat.onset(ch, band, base_window=baseline)
    if "onset" in include and on is not None:
        out.append(dict(name="onset", t0=max(0.0, on - 10), t1=min(dur, on + 30), f0=0.0, f1=fmax,
                        why=f"sustained +6 dB rise at {on:.1f} s"))
    if "loudest" in include:
        t, y, _ = feat.level_series(ch, band, dt=1.0)
        w = 30
        if len(y) > w:
            yy = np.nan_to_num(y)
            c = np.convolve(yy, np.ones(w), "valid")
            i = int(np.argmax(c))
            out.append(dict(name="loudest", t0=float(t[i] - 0.5), t1=float(t[i] - 0.5 + w),
                            f0=0.0, f1=fmax, why="30 s with the highest audio-band level"))
    if "glide" in include:
        try:
            from .glide import track_ridge, summarize
            t0 = (on or baseline[1]) + 6.0
            tr = track_ridge(feat, ch, baseline, t0, seed=glide_seed)
            s = summarize(tr)
            if s and s["significant"]:
                lo = min(s["f_start_hz"], s["f_end_hz"], tr.f_hz.min())
                hi = max(s["f_start_hz"], s["f_end_hz"], tr.f_hz.max())
                out.append(dict(name="glide", t0=s["t_start_s"], t1=s["t_end_s"], f0=0.6 * lo,
                                f1=min(fmax, 1.6 * hi), why=f"ridge {s['octaves']:+.2f} oct, "
                                f"contrast {s['median_contrast_db']:.0f} dB", track=tr))
        except ValueError:
            pass
    if "transient" in include and band in feat.meta["detect_bands"]:
        ev = feat.events
        ev = ev[(ev.channel == ch) & (ev.band == band) & (ev.t_s > (on or 0))]
        if len(ev):
            tt = float(ev.loc[ev.env_db.idxmax(), "t_s"])
            out.append(dict(name="transient", t0=max(0, tt - 0.05), t1=min(dur, tt + 0.20),
                            f0=0.0, f1=fmax, why=f"strongest trigger at {tt:.3f} s", kind="raw"))
    return out


def plot_overview_zoom(feat, ch, zooms=None, fmax=None, clim=None, pct=(2, 99.5), units="fs",
                       background=None, background_feat=None, diff="ratio", log_f=False,
                       markers=None, ncols=3, hide_masked=True, take=None):
    """Top: the WHOLE take (spectrogram, or difference image if `background` is given) with
    every zoom drawn as a box; masked time blank if hide_masked.  Middle: band levels for time
    context.  Below: one panel per zoom on the SAME colour scale.  Zooms with kind="raw"
    (single events) are recomputed from the raw audio at high resolution; pass `take`
    (from discover_takes) for those, otherwise they are skipped."""
    fmax = fmax or min(24000.0, feat.sr / 2)
    zooms = zooms if zooms is not None else suggest_zooms(feat, ch, fmax=fmax)
    nz = len(zooms)
    nrow_z = int(np.ceil(nz / ncols)) if nz else 0
    fig = plt.figure(figsize=(12, 4.6 + 2.6 * nrow_z), constrained_layout=True)
    gs = fig.add_gridspec(2 + nrow_z, ncols, height_ratios=[2.2, 0.8] + [2.0] * nrow_z)

    te, fe, D, lab, sym = spectrogram_db(feat, ch, None, fmax, 1.0 if log_f else 0.0, units,
                                         background, background_feat, diff,
                                         hide_masked=hide_masked)
    fin = D[np.isfinite(D)]
    lo, hi = clim or (np.percentile(fin, pct) if fin.size else (0, 1))
    if sym and clim is None:
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m
    cmap = "RdBu_r" if sym else "magma"
    ax0 = fig.add_subplot(gs[0, :])
    pm = ax0.pcolormesh(te, fe / 1e3, D, vmin=lo, vmax=hi, cmap=cmap, shading="flat", rasterized=True)
    fig.colorbar(pm, ax=ax0, pad=0.01, label=lab)
    ax0.set_ylabel("frequency [kHz]")
    ax0.set_title(f"{feat.take}  {feat.label(ch)}: full range, {feat.duration_s:.0f} s"
                  + (f", difference vs {background[0]:g}-{background[1]:g} s" if background else ""),
                  loc="left")
    ax0.grid(False)
    if log_f:
        ax0.set_yscale("log")
    for i, z in enumerate(zooms):
        col = PALETTE[i % len(PALETTE)]
        ax0.add_patch(Rectangle((z["t0"], z["f0"] / 1e3), z["t1"] - z["t0"], (z["f1"] - z["f0"]) / 1e3,
                                fill=False, ec=col, lw=1.6))
        ax0.text(z["t0"], z["f1"] / 1e3 * (1 - 0.09 * (i % 4)), f" {i+1} {z['name']}", color=col,
                 fontsize=8, va="top", fontweight="bold",
                 bbox=dict(facecolor="white", alpha=0.7, lw=0, pad=0.5))
    _marks(ax0, markers)

    # level strip for time context
    ax1 = fig.add_subplot(gs[1, :], sharex=ax0)
    for j, b in enumerate([b for b in ("low", "audio", "rig") if b in feat.bands]):
        t, y, _ = feat.level_series(ch, b, dt=1.0)
        ax1.plot(t, dsp.db(y), color=PALETTE[j], lw=0.9, label=b)
    ax1.set_ylabel("level\n[dB re FS²]")
    ax1.legend(loc="upper right", ncol=3)
    ax1.set_xlabel("time from take start [s]")
    for i, z in enumerate(zooms):
        ax1.axvspan(z["t0"], z["t1"], color=PALETTE[i % len(PALETTE)], alpha=0.12, lw=0)

    for i, z in enumerate(zooms):
        ax = fig.add_subplot(gs[2 + i // ncols, i % ncols])
        col = PALETTE[i % len(PALETTE)]
        if z.get("kind") == "raw":
            if take is None:
                ax.text(0.5, 0.5, "raw zoom: pass take=", ha="center", transform=ax.transAxes)
                ax.set_title(f"{i+1} {z['name']}", loc="left", fontsize=8, color=col)
                continue
            raw_zoom_axes(ax, take, ch, z["t0"], z["t1"], z["f1"])
        else:
            tz, fz, Dz, _, _ = spectrogram_db(feat, ch, (z["t0"], z["t1"]), z["f1"], z["f0"], units,
                                              background, background_feat, diff, max_cols=600,
                                              max_rows=500, hide_masked=hide_masked)
            ax.pcolormesh(tz, fz / 1e3, Dz, vmin=lo, vmax=hi, cmap=cmap, shading="flat",
                          rasterized=True)
            if "track" in z:
                tr = z["track"]
                m = (tr.t_s >= z["t0"]) & (tr.t_s <= z["t1"])
                ax.plot(tr.t_s[m], tr.f_hz[m] / 1e3, color=INK, lw=1.2)
            ax.set_xlim(z["t0"], z["t1"])
            ax.set_ylabel("kHz")
            if log_f and z["f0"] > 0:
                ax.set_yscale("log")
        ax.set_title(f"{i+1} {z['name']}: {z['why']}", loc="left", fontsize=8, color=col)
        ax.set_xlabel("s")
        ax.grid(False)
    return fig


def raw_zoom_axes(ax, take, ch, t0, t1, fmax=None, nperseg=256):
    """Fine STFT (nperseg samples, 90 % overlap) of the RAW audio in [t0, t1) with the
    high-passed waveform overlaid (white, arbitrary scale): the view for single events."""
    from scipy import signal as sg
    x = take.read(t0, t1, [ch])[:, 0]
    sr = take.sr
    f, t, S = sg.spectrogram(x, sr, nperseg=nperseg, noverlap=int(0.9 * nperseg), scaling="density")
    fm = f <= (fmax or sr / 2)
    D = dsp.db(S[fm])
    lo, hi = np.percentile(D, [5, 99.8])
    ax.pcolormesh(t0 + t, f[fm] / 1e3, D, vmin=lo, vmax=hi, cmap="magma", shading="nearest",
                  rasterized=True)
    sos = dsp.bandpass_sos(300, sr / 2, sr)
    y = sg.sosfiltfilt(sos, x) if len(x) > 50 else x
    ftop = (fmax or sr / 2) / 1e3
    ax.plot(t0 + np.arange(len(y)) / sr, ftop * (0.5 + 0.45 * y / (np.abs(y).max() + 1e-30)),
            color="w", lw=0.4, alpha=0.8)
    ax.set_xlim(t0, t1)
    ax.set_ylabel("kHz (raw STFT)")
    return ax


def plot_raw_zoom(take, ch, t0, t1, fmax=None, nperseg=256, cfg=None):
    """Stand-alone high-resolution view of a short stretch of raw audio (one event)."""
    fig, ax = plt.subplots(figsize=(8, 3.4), constrained_layout=True)
    raw_zoom_axes(ax, take, ch, t0, t1, fmax, nperseg)
    lab = cfg.channel_label(ch, take.name) if cfg else f"Ch{ch}"
    ax.set_title(f"{take.name} {lab}: raw audio {t0:.3f}-{t1:.3f} s, STFT {nperseg} samples "
                 f"({nperseg / take.sr * 1e3:.2f} ms); white = waveform", loc="left", fontsize=9)
    ax.set_xlabel("time from take start [s]")
    return fig
