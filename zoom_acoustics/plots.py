"""
plots.py -- standard figures.  Every function returns the matplotlib Figure so you can
tweak it, and works with any number of channels (1 to 8).

Conventions
  * One colour per CHANNEL NUMBER, always the same (Ch1 blue, Ch2 orange, ...), so a
    figure with channels 1 and 4 does not repaint Ch4 as the second colour.
  * Levels are dB re 1 FS^2 (uncalibrated).  Never compare absolute dB between different
    sensors; compare changes within one channel.
  * Masked time (chirps etc.) is shaded grey.
  * Two quantities with different units never share a y-axis: pH gets its own panel.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from . import dsp, masks as M
from .physics import band_radius_label

# categorical palette, validated for colour-vision deficiency (adjacent-pair dE >= 9)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#2b2b2b"
MASK_KW = dict(color="0.85", alpha=0.6, lw=0, zorder=0)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6,
    "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False, "lines.linewidth": 1.2,
})


def ch_color(ch):
    return PALETTE[(int(ch) - 1) % len(PALETTE)]


def band_color(i):
    return PALETTE[i % len(PALETTE)]


def _axes(n, height=1.9, width=10.0, sharex=True):
    fig, axs = plt.subplots(n, 1, figsize=(width, 0.8 + height * n), sharex=sharex,
                            squeeze=False, constrained_layout=True)
    return fig, axs[:, 0]


def _shade(ax, windows, t_range=None, max_n=2000):
    w = M.merge(windows)
    if t_range:
        w = [(a, b) for a, b in w if b > t_range[0] and a < t_range[1]]
    for a, b in w[:max_n]:
        ax.axvspan(a, b, **MASK_KW)


def _marks(ax, markers):
    for i, (label, t) in enumerate((markers or {}).items()):
        ax.axvline(t, color=INK, lw=0.8, ls="--")
        ax.text(t, 1.0 - 0.1 * (i % 3), f" {label}", transform=ax.get_xaxis_transform(),
                fontsize=7, va="top", ha="left", color=INK)


def _chs(feat, channels):
    return [int(c) for c in (channels or feat.channels) if int(c) in feat.channels]


def save(fig, cfg_or_dir, name, formats=("png",)):
    """Save into cfg.figures_path (or a directory); returns list of paths."""
    from pathlib import Path
    d = Path(cfg_or_dir.figures_path if hasattr(cfg_or_dir, "figures_path") else cfg_or_dir)
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for f in formats:
        p = d / f"{name}.{f}"
        fig.savefig(p, bbox_inches="tight")
        out.append(p)
    return out


# ---------------------------------------------------------------- raw waveforms
def plot_waveforms(take, t0, t1, channels=None, cfg=None, highpass_hz=None):
    """Raw samples of a short window, one panel per channel (reads the WAV directly)."""
    chans = list(channels or take.channel_numbers)
    x = take.read(t0, t1, chans)
    if highpass_hz:
        sos = dsp.bandpass_sos(highpass_hz, take.sr / 2, take.sr)
        from scipy.signal import sosfiltfilt
        x = sosfiltfilt(sos, x, axis=0)
    t = t0 + np.arange(x.shape[0]) / take.sr
    fig, axs = _axes(len(chans), 1.5)
    for ax, c, k in zip(axs, chans, range(len(chans))):
        ax.plot(t, x[:, k], color=ch_color(c), lw=0.5)
        lab = cfg.channel_label(c, take.name) if cfg else f"Ch{c}"
        ax.set_title(lab, loc="left")
        ax.set_ylabel("sample [FS]")
    axs[-1].set_xlabel("time from take start [s]")
    fig.suptitle(f"{take.name}: raw waveform {t0:.3f}-{t1:.3f} s"
                 + (f" (high-pass {highpass_hz:g} Hz)" if highpass_hz else ""), fontsize=10)
    return fig


# ---------------------------------------------------------------- spectrogram
def spectrogram_image(feat, ch, t_range=None, fmax=None, fmin=0.0, max_cols=1500, max_rows=700):
    """Time- and frequency-decimated spectrogram in LINEAR power for display.
    Returns (t_edges, f_edges, P[rows, cols])."""
    t = feat.t_psd
    f = feat.freqs
    t0, t1 = t_range or (0, feat.duration_s)
    it = np.flatnonzero((t >= t0) & (t < t1))
    jf = np.flatnonzero((f >= fmin) & (f <= (fmax or f[-1])))
    if not it.size or not jf.size:
        raise ValueError("empty time or frequency range")
    kt = max(1, int(np.ceil(it.size / max_cols)))
    kf = max(1, int(np.ceil(jf.size / max_rows)))
    ncol = it.size // kt
    nrow = jf.size // kf
    P = feat.psd(ch)
    img = np.empty((nrow, ncol))
    for c in range(ncol):
        sl = np.asarray(P[it[c * kt]:it[c * kt] + kt, jf[0]:jf[0] + nrow * kf], float)
        img[:, c] = sl.mean(axis=0).reshape(nrow, kf).mean(axis=1)
    dt = feat.psd_dt * kt
    te = t[it[0]] - feat.psd_dt / 2 + dt * np.arange(ncol + 1)
    df = (f[1] - f[0]) * kf
    fe = f[jf[0]] - (f[1] - f[0]) / 2 + df * np.arange(nrow + 1)
    return te, fe, img


def plot_spectrogram(feat, channels=None, fmax=None, fmin=0.0, t_range=None, clim=None,
                     pct=(2, 99.5), log_f=False, markers=None, cmap="magma"):
    """One spectrogram panel per channel, dB re FS^2/Hz.  Colour limits from percentiles of
    the data (or clim=(lo, hi) to fix them, which you should do when comparing takes)."""
    chans = _chs(feat, channels)
    fig, axs = _axes(len(chans), 2.2)
    for ax, c in zip(axs, chans):
        te, fe, img = spectrogram_image(feat, c, t_range, fmax, fmin=max(fmin, 1.0 if log_f else fmin))
        D = dsp.db(img)
        lo, hi = clim or np.percentile(D[np.isfinite(D)], pct)
        pm = ax.pcolormesh(te, fe / 1e3, D, vmin=lo, vmax=hi, cmap=cmap, shading="flat",
                           rasterized=True)
        if log_f:
            ax.set_yscale("log")
        ax.set_ylabel("frequency [kHz]")
        ax.set_title(feat.label(c), loc="left")
        ax.grid(False)
        _marks(ax, markers)
        fig.colorbar(pm, ax=ax, pad=0.01, label="dB re FS²/Hz")
    axs[-1].set_xlabel("time from take start [s]")
    fig.suptitle(f"{feat.take}: spectrogram ({feat.psd_dt:g} s frames, "
                 f"{feat.freqs[1]-feat.freqs[0]:.1f} Hz bins)", fontsize=10)
    return fig


# ---------------------------------------------------------------- band levels
def plot_levels(feat, bands=("low", "audio", "rig"), channels=None, dt=1.0, markers=None,
                relative_to=None, t_range=None):
    """Band mean-square level vs time, one panel per channel, one line per band.
    relative_to=(t0, t1): plot dB change relative to the median level in that window
    (the physically meaningful quantity for an uncalibrated sensor)."""
    chans = _chs(feat, channels)
    bands = [b for b in bands if b in feat.bands]
    fig, axs = _axes(len(chans), 1.9)
    for ax, c in zip(axs, chans):
        _shade(ax, feat.mask_windows, t_range)
        for i, b in enumerate(bands):
            t, y, _ = feat.level_series(c, b, dt=dt)
            L = dsp.db(y)
            if relative_to:
                m = (t >= relative_to[0]) & (t < relative_to[1]) & np.isfinite(L)
                L = L - (np.nanmedian(L[m]) if m.any() else np.nan)
            lo, hi = feat.band_edges[b]
            lab = f"{b} {lo/1e3:g}-{hi/1e3:g} kHz" if b != "broadband" else "broadband"
            ax.plot(t, L, color=band_color(i), label=lab)
        ax.set_title(feat.label(c), loc="left")
        ax.set_ylabel("Δ level [dB]" if relative_to else "level [dB re FS²]")
        _marks(ax, markers)
        if t_range:
            ax.set_xlim(t_range)
    axs[0].legend(loc="upper right", ncol=min(4, len(bands)))
    axs[-1].set_xlabel("time from take start [s]")
    fig.suptitle(f"{feat.take}: band levels, {dt:g} s bins"
                 + (f", relative to {relative_to[0]:g}-{relative_to[1]:g} s" if relative_to else ""),
                 fontsize=10)
    return fig


# ---------------------------------------------------------------- spectra
def plot_psd(feat, windows, channels=None, fmax=None, fmin=None, shade_bands=("rig",),
             excess=False):
    """Time-averaged PSD in named windows, e.g. {"before acid": (5, 35), "after": (60, 170)},
    one panel per channel.  excess=True adds (window - first window) in LINEAR power:
    the spectrum of what was ADDED, which is what a new source contributes."""
    chans = _chs(feat, channels)
    names = list(windows)
    fig, axs = _axes(len(chans), 2.4, sharex=True)
    f = feat.freqs
    fm = (f > (fmin or f[1])) & (f <= (fmax or f[-1]))
    for ax, c in zip(axs, chans):
        ref = None
        for i, w in enumerate(names):
            P, n = feat.psd_mean(c, *windows[w])
            if P is None:
                continue
            ref = P if ref is None else ref
            ax.plot(f[fm] / 1e3, dsp.db(P[fm]), color=band_color(i),
                    label=f"{w} ({windows[w][0]:g}-{windows[w][1]:g} s, {n} frames)")
            if excess and i > 0:
                ex = np.maximum(P - ref, 0)
                ax.plot(f[fm] / 1e3, dsp.db(np.where(ex > 0, ex, np.nan)[fm]), color=band_color(i),
                        ls=":", lw=1.0, label=f"{w} minus {names[0]} (excess)")
        for b in shade_bands or []:
            if b in feat.band_edges:
                lo, hi = feat.band_edges[b]
                ax.axvspan(lo / 1e3, hi / 1e3, color="0.8", alpha=0.5, lw=0)
        ax.set_xscale("log")
        ax.set_ylabel("PSD [dB re FS²/Hz]")
        ax.set_title(feat.label(c), loc="left")
    axs[0].legend(loc="upper right")
    axs[-1].set_xlabel("frequency [kHz]")
    fig.suptitle(f"{feat.take}: mean spectra (masked time excluded)"
                 + ("; grey = " + ", ".join(shade_bands) + " band" if shade_bands else ""),
                 fontsize=10)
    return fig


# ---------------------------------------------------------------- events
def plot_event_rate(feat, band="audio", channels=None, bin_s=5.0, markers=None, log=True):
    """Live-time-normalised STA/LTA TRIGGER rate per channel (not a bubble rate)."""
    chans = [c for c in _chs(feat, channels)]
    fig, axs = _axes(len(chans), 1.6)
    for ax, c in zip(axs, chans):
        _shade(ax, feat.mask_windows)
        t, r, n, _ = feat.event_rate(c, band, bin_s)
        ax.step(t, r, where="mid", color=ch_color(c))
        if log and np.nanmax(np.r_[r, 0]) > 0:
            ax.set_yscale("symlog", linthresh=0.1)
        ax.set_ylabel("triggers / s")
        ax.set_title(f"{feat.label(c)}  ({int(n.sum())} triggers)", loc="left")
        _marks(ax, markers)
    axs[-1].set_xlabel("time from take start [s]")
    fig.suptitle(f"{feat.take}: trigger rate, {band} band, {bin_s:g} s bins "
                 f"(STA/LTA on {feat.meta['detector']['on']:g}/{feat.meta['detector']['off']:g})",
                 fontsize=10)
    return fig


def plot_detector(feat, ch, band="audio", t0=0.0, t1=None):
    """Draw exactly what the detector sees: envelope, STA/LTA ratio, thresholds, triggers.
    Look at this before trusting any event count."""
    t1 = t1 or min(feat.duration_s, t0 + 2.0)
    e = feat.env(ch, band)
    dtv = feat.env_dt
    t = (np.arange(len(e)) + 0.5) * dtv
    k0, k1 = int(t0 / dtv), int(t1 / dtv)
    pre = max(0, k0 - int(feat.meta["detector"]["lta_s"] / dtv) - 2)
    d = dsp.StaLta(dtv, **feat.meta["detector"])
    ee = e[pre:k1].copy()
    ee[M.mask_array(t[pre:k1], feat.mask_windows)] = np.nan
    r = d.ratio(ee)
    ev = feat.events
    ev = ev[(ev.channel == ch) & (ev.band == band) & (ev.t_s >= t0) & (ev.t_s < t1)]
    fig, axs = _axes(2, 1.8)
    s = slice(k0 - pre, k1 - pre)
    axs[0].plot(t[k0:k1], dsp.db(e[k0:k1]), color=ch_color(ch), lw=0.7)
    axs[0].set_ylabel("envelope [dB re FS²]")
    axs[0].set_title(f"{feat.label(ch)}: {band} band envelope, {dtv*1e3:g} ms samples", loc="left")
    axs[1].plot(t[k0:k1], r[s], color=INK, lw=0.7)
    axs[1].axhline(feat.meta["detector"]["on"], color=PALETTE[1], ls="--", lw=0.8, label="on")
    axs[1].axhline(feat.meta["detector"]["off"], color=PALETTE[2], ls="--", lw=0.8, label="off")
    axs[1].set_yscale("log")
    axs[1].set_ylabel("STA / LTA")
    axs[1].legend(loc="upper right")
    for ax in axs:
        for x in ev.t_s:
            ax.axvline(x, color=PALETTE[1], lw=0.5, alpha=0.7)
        _shade(ax, feat.mask_windows, (t0, t1))
    axs[-1].set_xlabel("time from take start [s]")
    fig.suptitle(f"{feat.take}: detector anatomy, {len(ev)} triggers (orange lines)", fontsize=10)
    return fig


# ---------------------------------------------------------------- timelines and pH
def _datetime_axis(ax):
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def plot_timeline(series_by_panel, ph=None, ph_col="pH", title="", markers=None,
                  extra_ph_cols=()):
    """Stacked panels on one lab-clock axis.

    series_by_panel: {panel title: [(label, pandas Series in dB or rate, color), ...]}
    ph: DataFrame from ph.load_ph (optional) -> its own bottom panel(s)."""
    n = len(series_by_panel) + (1 if ph is not None else 0) + len(extra_ph_cols)
    fig, axs = _axes(n, 1.8)
    for ax, (ttl, items) in zip(axs, series_by_panel.items()):
        for lab, s, col in items:
            ax.plot(s.index, s.to_numpy(), color=col, label=lab)
        ax.set_title(ttl, loc="left")
        if len(items) > 1:
            ax.legend(loc="upper right")
    k = len(series_by_panel)
    if ph is not None:
        ax = axs[k]
        ax.plot(ph.index, ph[ph_col], "o-", ms=3, color=INK, lw=0.8)
        ax.set_ylabel(ph_col)
        ax.set_title(ph_col, loc="left")
        for j, c in enumerate(extra_ph_cols, 1):
            axs[k + j].plot(ph.index, ph[c], "o-", ms=3, color="0.4", lw=0.8)
            axs[k + j].set_title(c, loc="left")
            axs[k + j].set_ylabel(c)
    for ax in axs:
        for lab, t in (markers or {}).items():
            ax.axvline(t, color=INK, lw=0.8, ls="--")
        _datetime_axis(ax)
    axs[-1].set_xlabel("lab time")
    fig.suptitle(title, fontsize=10)
    return fig


def plot_ph_scatter(aligned, x="pH", y="acoustic_dB", fit=True, label="", color=None):
    """Acoustic level vs pH, points coloured by time, with an ordinary least-squares line.
    The fitted slope is printed; its CI assumes independent points, which a time series
    is not -- treat it as descriptive."""
    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    tnum = (aligned.index - aligned.index[0]).total_seconds() / 60
    sc = ax.scatter(aligned[x], aligned[y], c=tnum, cmap="viridis", s=18, edgecolor="none")
    fig.colorbar(sc, ax=ax, label="minutes since first point")
    txt = ""
    if fit and len(aligned) >= 3:
        p, cov = np.polyfit(aligned[x], aligned[y], 1, cov=True)
        xx = np.linspace(aligned[x].min(), aligned[x].max(), 50)
        ax.plot(xx, np.polyval(p, xx), color=color or PALETTE[1], lw=1.5)
        txt = f"slope {p[0]:.2f} ± {np.sqrt(cov[0,0]):.2f} dB per unit, n = {len(aligned)}"
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=8)
    ax.set_xlabel(x)
    ax.set_ylabel(f"{label} level [dB re FS²]" if y == "acoustic_dB" else y)
    return fig


def plot_lag(corr, title=""):
    fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
    ax.plot(corr.lag_s / 60, corr.r, color=INK)
    j = int(np.nanargmax(np.abs(corr.r.to_numpy()))) if corr.r.notna().any() else None
    if j is not None:
        ax.axvline(corr.lag_s.iloc[j] / 60, color=PALETTE[1], ls="--", lw=0.8)
        ax.text(corr.lag_s.iloc[j] / 60, corr.r.iloc[j], f" r={corr.r.iloc[j]:.2f} at "
                f"{corr.lag_s.iloc[j]/60:+.1f} min", fontsize=8, va="bottom")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel("lag [min]  (positive: second series lags the first)")
    ax.set_ylabel("correlation")
    ax.set_title(title, loc="left")
    return fig


def plot_compare_psd(features, ch, t_window=None, fmax=None, windows_by_take=None, title=None):
    """One channel, several takes: time-averaged PSD of each take (whole take by default, or
    windows_by_take={take: (t0, t1)}).  Same sensor across takes = comparable shape; an
    absolute offset between takes is only meaningful if nothing was moved or re-gained."""
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for i, f in enumerate(features):
        if ch not in f.channels:
            continue
        w = (windows_by_take or {}).get(f.take, t_window or (0, f.duration_s))
        P, n = f.psd_mean(ch, *w)
        if P is None:
            continue
        m = (f.freqs > 0) & (f.freqs <= (fmax or f.freqs[-1]))
        ax.plot(f.freqs[m] / 1e3, dsp.db(P[m]), color=band_color(i),
                label=f"{f.take} ({w[0]:g}-{w[1]:g} s)")
    ax.set_xscale("log")
    ax.set_xlabel("frequency [kHz]")
    ax.set_ylabel("PSD [dB re FS²/Hz]")
    ax.legend()
    ax.set_title(title or f"Ch{ch}: mean spectrum per take", loc="left")
    return fig


def band_legend_text(bands_with_edges):
    """Readable band list with the Minnaert radius range, for captions."""
    return "; ".join(f"{n}: {lo/1e3:g}-{hi/1e3:g} kHz ({band_radius_label(lo, hi)})"
                     for n, lo, hi in bands_with_edges if lo > 0)


# ---------------------------------------------------------------- glide
def plot_glide(feat, tracks_by_ch, base_window, show_ch=None, fmax=20000, clim=None, markers=None,
               base_feat=None):
    """Top: excess spectrogram (dB over the baseline mean spectrum) of one channel with its
    ridge overlaid.  Bottom: ridge frequency vs time for every channel tracked, log scale.
    tracks_by_ch: {channel: DataFrame(t_s, f_hz)} from glide.track_ridge."""
    from .glide import excess_spectrogram
    chs = [c for c in tracks_by_ch if len(tracks_by_ch[c])]
    show_ch = show_ch or (chs[0] if chs else feat.channels[0])
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True,
                            gridspec_kw=dict(height_ratios=[1.3, 1]))
    t, f, D = excess_spectrogram(feat, show_ch, base_window, f_range=(100.0, fmax),
                                 base_feat=base_feat)
    Ddb = dsp.db(D)
    lo, hi = clim or np.percentile(Ddb, [2, 99.5])
    m = max(abs(lo), abs(hi))
    pm = axs[0].pcolormesh(t, f / 1e3, Ddb.T, cmap="RdBu_r", vmin=-m, vmax=m,
                           shading="nearest", rasterized=True)
    fig.colorbar(pm, ax=axs[0], pad=0.01, label=f"dB over {base_window[0]:g}-{base_window[1]:g} s mean")
    if show_ch in tracks_by_ch and len(tracks_by_ch[show_ch]):
        tr = tracks_by_ch[show_ch]
        axs[0].plot(tr.t_s, tr.f_hz / 1e3, color=INK, lw=1.6, label="tracked ridge")
        axs[0].legend(loc="upper right")
    axs[0].set_yscale("log")
    axs[0].set_ylim(max(0.1, f[0] / 1e3), fmax / 1e3)
    axs[0].set_ylabel("frequency [kHz]")
    axs[0].set_title(f"{feat.label(show_ch)}: excess spectrogram", loc="left")
    axs[0].grid(False)
    from .glide import summarize
    for c in chs:
        tr = tracks_by_ch[c]
        sm = summarize(tr)
        lab = (f"{feat.label(c)}  {sm['octaves']:+.2f} oct, contrast {sm['median_contrast_db']:.0f} dB"
               + ("" if sm["significant"] else "  (NOT significant)")) if sm else feat.label(c)
        axs[1].plot(tr.t_s, tr.f_hz / 1e3, color=ch_color(c), label=lab,
                    ls="-" if (sm and sm["significant"]) else ":")
    axs[1].set_yscale("log")
    axs[1].set_ylabel("ridge frequency [kHz]")
    axs[1].set_xlabel("time from take start [s]")
    axs[1].legend(loc="lower left")
    axs[1].set_title("ridge per channel (an air mic that glides too = not liquid-borne)", loc="left")
    for ax in axs:
        _marks(ax, markers)
    fig.suptitle(f"{feat.take}: glide analysis", fontsize=10)
    return fig


# ---------------------------------------------------------------- catalogue / clustering
def plot_interevent(t, windows=None, dead_s=1.5e-3, title=""):
    """Histogram of log10 inter-event gaps against the exponential (Poisson) expectation
    with the same mean; the CV is printed.  Excess at short gaps = clustering or multiple
    triggers per source; excess at long gaps = quiet spells."""
    from .catalogue import live_gaps
    g, _ = live_gaps(t, windows)
    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    if g.size < 10:
        ax.text(0.5, 0.5, "fewer than 10 gaps", transform=ax.transAxes, ha="center")
        return fig
    edges = np.logspace(np.log10(max(g.min(), 1e-4)), np.log10(g.max()), 50)
    ax.hist(g, edges, color=PALETTE[0], alpha=0.8, label=f"observed, n = {g.size}")
    mu = g.mean()
    cdf = 1 - np.exp(-edges / mu)
    ax.step(edges[:-1], g.size * np.diff(cdf), where="post", color=INK,
            label="Poisson (exponential, same mean)")
    ax.axvline(2 * dead_s, color=PALETTE[1], ls="--", lw=0.8, label="2 × dead time")
    ax.set_xscale("log")
    ax.set_xlabel("gap between successive events [s]")
    ax.set_ylabel("count")
    ax.legend()
    ax.set_title(f"{title}  CV = {g.std(ddof=1)/mu:.2f} (Poisson 1.00)", loc="left")
    return fig


def plot_coincidence(res, title=""):
    fig, ax = plt.subplots(figsize=(6, 3.2), constrained_layout=True)
    if not res.get("reportable"):
        ax.text(0.5, 0.5, "too few events", transform=ax.transAxes, ha="center")
        return fig
    ax.hist(res["lags_s"] * 1e3, 60, color=PALETTE[2])
    ax.set_xlabel("lag, B minus A [ms]")
    ax.set_ylabel("coincident pairs")
    ax.set_title(f"{title}  {100*res['frac_obs']:.1f} % coincident vs {100*res['frac_chance']:.1f} % "
                 f"by chance (excess {100*res['excess']:.1f} %)", loc="left", fontsize=9)
    return fig


def plot_waveform_gallery(W, sr, labels=None, n_per_row=8, max_rows=6, pre_s=0.002, title=""):
    """Grid of event waveforms, one row per cluster (largest first) if labels are given."""
    t = (np.arange(W.shape[1]) / sr - pre_s) * 1e3
    if labels is None:
        labels = np.ones(len(W), int)
    ks = [k for k, _ in sorted(((k, (labels == k).sum()) for k in np.unique(labels)),
                               key=lambda x: -x[1])][:max_rows]
    fig, axs = plt.subplots(len(ks), n_per_row, figsize=(1.5 * n_per_row, 1.2 * len(ks) + 0.6),
                            squeeze=False, sharex=True, constrained_layout=True)
    for r, k in enumerate(ks):
        idx = np.flatnonzero(labels == k)[:n_per_row]
        for c in range(n_per_row):
            ax = axs[r, c]
            ax.set_yticks([])
            ax.grid(False)
            if c < idx.size:
                w = W[idx[c]]
                ax.plot(t, w / (np.abs(w).max() + 1e-30), color=band_color(r), lw=0.6)
            else:
                ax.axis("off")
        axs[r, 0].set_ylabel(f"cl {k}\nn={int((labels == k).sum())}", fontsize=7)
    for ax in axs[-1]:
        ax.set_xlabel("ms", fontsize=7)
    fig.suptitle(title or "event waveforms (normalised), one row per cluster", fontsize=10)
    return fig


def plot_corr_matrix(C, labels):
    """Pairwise waveform correlation, rows/columns sorted by cluster."""
    o = np.argsort(labels, kind="stable")
    fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    im = ax.imshow(C[np.ix_(o, o)], vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="peak normalised cross-correlation")
    b = np.flatnonzero(np.diff(labels[o])) + 0.5
    for x in b:
        ax.axhline(x, color="w", lw=0.4)
        ax.axvline(x, color="w", lw=0.4)
    ax.set_xlabel("event (sorted by cluster)")
    ax.set_ylabel("event")
    ax.grid(False)
    return fig


def plot_feature_space(features, labels, scores, evr, x="f_peak_hz", y="decay_ms"):
    """Left: two raw features coloured by cluster; right: first two principal components."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for i, k in enumerate(np.unique(labels)):
        m = labels == k
        axs[0].scatter(features[x][m], features[y][m], s=8, color=band_color(i), label=f"cl {k} (n={m.sum()})")
        axs[1].scatter(scores[m, 0], scores[m, 1], s=8, color=band_color(i))
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel(x)
    axs[0].set_ylabel(y)
    axs[0].legend()
    axs[1].set_xlabel(f"PC1 ({100*evr[0]:.0f} %)")
    axs[1].set_ylabel(f"PC2 ({100*evr[1]:.0f} %)")
    axs[1].set_title("standardised log-features, PCA", loc="left")
    return fig
