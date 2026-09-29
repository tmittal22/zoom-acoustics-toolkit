"""
report.py -- save the STANDARD FIGURE SET for one take, plus an index page.

    za.report.save_figure_set(cfg, "260910_011", out_dir, baseline=(3, 25), after=(60, 380))

writes numbered PNGs and INDEX.md into out_dir:

    00_overview_zoom.png   full range + automatic zooms (baseline, onset, loudest, glide, event)
    00b_difference.png     the same as a difference image against the baseline window
    01_waveforms.png       raw samples around the onset (or a chosen time), all channels
    02_spectrogram.png     spectrogram per channel
    03_levels_delta.png    band levels as change relative to the baseline window
    04_spectra.png         mean PSD baseline vs after, and the excess (after - baseline)
    05_trigger_rate.png    live-time-normalised STA/LTA trigger rate per channel
    06_detector.png        detector anatomy (envelope, ratio, thresholds) for 0.5 s
    07_interevent.png      inter-event gaps vs Poisson, one channel
    08_glide.png           excess spectrogram + ridge tracks per channel
    08b_glide_interpretation.png   what the strongest ridge requires under each mechanism
    09_summary.csv         per-channel numbers (levels, onset, trigger rate, glide)

Every figure is also returned, so a notebook can show them inline.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import dsp, plots as zp, glide as G, catalogue as K
from .features import load_features
from .io import discover_takes, find_take


def save_figure_set(cfg, take_name, out_dir=None, baseline=(3.0, 25.0), after=None,
                    band="audio", markers=None, fmax=24000, waveform_t=None,
                    glide_seed=(4000.0, 12000.0), glide_t0=None, close=True, title=None,
                    view_channel=None):
    """Write the standard figure set for one processed take.  Returns (paths, summary)."""
    f = load_features(cfg, take_name)
    out = Path(out_dir or (cfg.figures_path / f"{take_name}_figure_set"))
    out.mkdir(parents=True, exist_ok=True)
    after = after or (baseline[1] + 20.0, f.duration_s)
    fmax = min(fmax, f.sr / 2)
    rows = {c: dict(channel=c, label=f.label(c), sensor=f.info(c)["sensor"]) for c in f.channels}
    saved = []

    def put(fig, name, caption):
        p = zp.save(fig, out, name)[0]
        saved.append((p.name, caption))
        if close:
            plt.close(fig)

    # onset per channel (also used for the waveform window)
    ons = {}
    for c in f.channels:
        on, base = f.onset(c, band, base_window=baseline)
        ons[c] = on
        rows[c].update(onset_s=on, baseline_dB=float(dsp.db(base)) if np.isfinite(base) else np.nan)
    first_on = min([o for o in ons.values() if o is not None], default=None)
    mk = dict(markers or {})
    if first_on is not None and not mk:
        mk = {"onset": first_on}

    from . import views as V
    try:
        take = find_take(discover_takes(cfg.data_dir, recursive=cfg.recursive,
                                        pattern=cfg.file_pattern, verbose=False), take_name)
    except Exception:
        take = None
    ch_view = view_channel or f.channels[0]
    zooms = V.suggest_zooms(f, ch_view, baseline=baseline, fmax=fmax, glide_seed=glide_seed)
    fig = V.plot_overview_zoom(f, ch_view, zooms, fmax=fmax, take=take, markers=mk)
    put(fig, "00_overview_zoom", f"{f.label(ch_view)}: the whole take (masked time blank), band levels "
        "for context, and zooms chosen by the analysis: baseline, onset, loudest 30 s, a significant "
        "glide, and the strongest single event from the raw audio.")
    fig = V.plot_overview_zoom(f, ch_view, zooms, fmax=fmax, take=take, markers=mk, background=baseline)
    put(fig, "00b_difference", f"Difference image: dB over the mean spectrum of the {baseline[0]:g}-"
        f"{baseline[1]:g} s baseline. White = unchanged, red = added, blue = removed.")

    # 01 waveforms
    try:
        if take is None:
            raise FileNotFoundError
        tw = waveform_t if waveform_t is not None else (first_on + 5.0 if first_on else f.duration_s / 2)
        fig = zp.plot_waveforms(take, tw, tw + 0.5, cfg=cfg, highpass_hz=500)
        put(fig, "01_waveforms", f"Raw samples, {tw:.1f}-{tw+0.5:.1f} s, high-passed at 500 Hz. "
            "Each spike is one transient; look for dead or clipped channels.")
    except Exception as e:                                    # raw files not available
        take = None
        saved.append(("01_waveforms.png", f"skipped: raw WAV not available ({e.__class__.__name__})"))

    # 02 spectrogram
    fig = zp.plot_spectrogram(f, fmax=fmax, markers=mk)
    put(fig, "02_spectrogram", "Spectrogram per channel, dB re FS²/Hz, colour limits from the data "
        "(2nd-99.5th percentile). Masked time is not removed here, so chirps show as vertical stripes.")

    # 03 levels
    bands = [b for b in ("low", "audio", "rig", "hb1") if b in f.bands]
    fig = zp.plot_levels(f, bands=bands, dt=1.0, markers=mk, relative_to=baseline)
    put(fig, "03_levels_delta", f"Band levels in 1 s bins as change from the {baseline[0]:g}-"
        f"{baseline[1]:g} s median. Grey = masked. The change, not the absolute dB, is comparable "
        "between channels.")
    for c in f.channels:
        for b in bands:
            v0 = f.band_level_in_window(c, b, *baseline, stat="median")
            v1 = f.band_level_in_window(c, b, *after, stat="median")
            rows[c][f"{b}_delta_dB"] = float(dsp.db(v1) - dsp.db(v0))   # median: robust to impulses

    # 04 spectra
    fig = zp.plot_psd(f, {"baseline": baseline, "after": after}, fmax=fmax, fmin=100, excess=True)
    put(fig, "04_spectra", "Mean spectra, baseline vs after, and the excess (after minus baseline "
        "in linear power) as a dotted line. Grey band = 'rig' apparatus band.")

    # 05 trigger rate, 06 detector, 07 inter-event
    det = band if band in f.meta["detect_bands"] else (f.meta["detect_bands"] or [None])[0]
    if det:
        fig = zp.plot_event_rate(f, det, bin_s=5.0, markers=mk)
        put(fig, "05_trigger_rate", "STA/LTA trigger rate per live second, 5 s bins. A trigger is "
            "not a bubble: at high rates the count saturates.")
        ev = f.events
        for c in f.channels:
            t, r, n, live = f.event_rate(c, det, bin_s=f.duration_s, events=ev)
            rows[c]["trigger_rate_per_s"] = float(n.sum() / max(live.sum(), 1e-9))
        c0 = f.channels[0]
        if f.meta.get("env_stored"):
            t_chk = (after[0], after[0] + 0.5)
            fig = zp.plot_detector(f, c0, det, *t_chk)
            put(fig, "06_detector", f"What the detector saw on {f.label(c0)}, {t_chk[0]:g}-"
                f"{t_chk[1]:g} s: envelope, STA/LTA ratio, on/off thresholds, triggers.")
        tt = ev[(ev.channel == c0) & (ev.band == det)].t_s.to_numpy()
        tt = tt[(tt >= after[0]) & (tt < after[1])]
        fig = zp.plot_interevent(tt, f.mask_windows, f.meta["detector"]["dead_s"],
                                 title=f"{f.label(c0)}, {after[0]:g}-{after[1]:g} s:")
        put(fig, "07_interevent", "Gaps between successive triggers against a Poisson process with "
            "the same mean (CV 1). Gaps across masked windows are excluded.")

    # 08 glide
    try:
        t0 = glide_t0 if glide_t0 is not None else ((first_on or baseline[1]) + 6.0)
        tracks = {}
        for c in f.channels:
            tr = G.track_ridge(f, c, baseline, t0, seed=glide_seed)
            s = G.summarize(tr)
            if s:
                tracks[c] = tr
                rows[c].update(glide_octaves=s["octaves"], glide_contrast_dB=s["median_contrast_db"],
                               glide_significant=s["significant"])
        if tracks:
            best = max(tracks, key=lambda c: rows[c].get("glide_contrast_dB", 0))
            fig = zp.plot_glide(f, tracks, baseline, show_ch=best, fmax=min(20000, fmax))
            put(fig, "08_glide", "Top: excess spectrogram (dB over the baseline spectrum) of the "
                "channel with the strongest ridge, with its track. Bottom: ridge per channel; dotted "
                "= not significant (median contrast under 6 dB).")
            if rows[best].get("glide_significant"):
                it = G.interpret(tracks[best])
                fig = zp.plot_glide_interpretation(it, f"{take_name} {f.label(best)}")
                put(fig, "08b_glide_interpretation", "What the strongest ridge would require: a free "
                    "or wall-attached growing bubble (radius vs the capillary length) or a bubbly "
                    "layer (void fraction for assumed thicknesses). See docs/GLIDE_INTERPRETATION.md.")
    except ValueError as e:
        saved.append(("08_glide.png", f"skipped: {e}"))

    summ = pd.DataFrame(list(rows.values()))
    summ.to_csv(out / "09_summary.csv", index=False)
    saved.append(("09_summary.csv", "per-channel numbers"))

    lines = [f"# Figure set: {title or take_name}", "",
             f"Take `{take_name}`, {f.duration_s:.1f} s, {f.sr} Hz, channels {f.channels}. "
             f"Baseline {baseline[0]:g}-{baseline[1]:g} s, 'after' window {after[0]:g}-{after[1]:.0f} s.", ""]
    for name, cap in saved:
        if name.endswith(".png") and not cap.startswith("skipped"):
            lines += [f"## {name[:-4].replace('_', ' ')}", "", cap, "", f"![{name}]({name})", ""]
        else:
            lines += [f"* `{name}`: {cap}", ""]
    tab = (summ.round(3).to_markdown(index=False) if _has_tabulate()
           else "```\n" + summ.round(3).to_string(index=False) + "\n```")
    lines += ["## Per-channel summary", "", tab, ""]
    (out / "INDEX.md").write_text("\n".join(lines))
    return [out / n for n, _ in saved], summ


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False
