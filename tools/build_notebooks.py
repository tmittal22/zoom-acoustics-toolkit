"""
build_notebooks.py -- generate notebooks/*.ipynb from the cell lists below.

Edit HERE, not in the .ipynb files, then run:   python tools/build_notebooks.py
(Students: just use the notebooks; you never need this script.)
"""
from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = '''\
# ---- standard setup (same in every notebook) ----
%matplotlib inline
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import zoom_acoustics as za
from zoom_acoustics import plots as zp, dsp

cfg = za.load_config(CONFIG)
print("config   :", cfg.config_file)
print("data_dir :", cfg.data_dir)
print("cache_dir:", cfg.cache_dir)
print("figures  :", cfg.figures_dir)'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


NOTEBOOKS = {}

# ============================================================================ 00
NOTEBOOKS["00_setup_and_inventory"] = [
    md('''
# 00 · Setup and inventory

**Run this first**, every time you start on a new data folder.

1. Checks that the Python environment is the right one.
2. Loads your config file (all paths live there, nowhere else).
3. Lists every take found in `data_dir`: start time, duration, number of channels, sample rate,
   split files.
4. Estimates how big the processed cache will be.

**What you edit:** only the cell marked `YOUR SETTINGS`. To start a real experiment, copy
`config/TEMPLATE.yaml` to `config/local_<experiment>.yaml`, fill it in, and point `CONFIG` at it.
The demo config works out of the box after `python -m zoom_acoustics demo demo_data`.
'''),
    code('''
# ==== YOUR SETTINGS ====
CONFIG = "../config/demo.yaml"
'''),
    code('''
import sys, importlib
print("python:", sys.executable)
print("       ", sys.version.split()[0])
for m in ["numpy", "scipy", "pandas", "matplotlib", "soundfile", "yaml", "openpyxl"]:
    mod = importlib.import_module(m)
    print(f"  {m:11s} {getattr(mod, '__version__', '?')}")
'''),
    code(SETUP),
    md('''
## Takes found

A **take** is one continuous recording. The recorder may have written it as

* one multi-channel file, `260910_011.WAV`;
* several split files (the recorder starts a new file at about 2 GB), `260909_008_0001.WAV`, `_0002.WAV`, which are
  joined seamlessly;
* one mono file per track, `..._Tr1.WAV`, `..._Tr2.WAV`, which become the channels of one take.

`start` is the **recorder clock** time written into each file (1 s resolution). If the recorder clock
was wrong, set `recorder_clock_offset_s` in the config (see the student guide, *Clocks*).
'''),
    code('''
takes = za.discover_takes(cfg.data_dir, recursive=cfg.recursive, pattern=cfg.file_pattern)
bad = cfg.unmatched_take_keys(takes)
if bad:
    print("WARNING: config 'takes:' entries that match no take (typo?):", bad)
table = za.takes_table(takes)
table
'''),
    md('''
## Channels

Channel numbers are the recorder **track numbers** (Ch1 = Tr1). Names and sensor types come from the
`channels:` section of the config. `sensor` must be one of `hydrophone`, `contact`, `air`, `other`.
Takes may have different channel sets. That is expected; every tool skips channels a take does not have.
'''),
    code('''
all_ch = sorted({c for t in takes for c in t.channel_numbers})
pd.DataFrame([dict(channel=c, **cfg.channel_info(c),
                   in_takes=sum(c in t.channel_numbers for t in takes)) for c in all_ch])
'''),
    md('''
## Processing plan and cache size

`cache_GB` is what processing will write to `cache_dir`. If it is large (long recordings), raise
`dsp.psd_dt` (e.g. 1 s or 5 s) or set `dsp.psd_fmax_hz` in the config; see the student guide, *Long recordings*.
'''),
    code('''
rows = []
for t in takes:
    p = za.plan(t, cfg)
    rows.append(dict(take=t.name, channels=p["channels"], sr=t.sr,
                     freq_resolution_Hz=round(t.sr / p["nperseg"], 2),
                     store_envelope=p["store_env"], cache_GB=round(sum(p["size"].values()), 3),
                     notes="; ".join(p["notes"])))
pd.DataFrame(rows)
'''),
]

# ============================================================================ 01
NOTEBOOKS["01_quicklook_one_take"] = [
    md('''
# 01 · Quick look at one take

Raw waveforms first, before any processing, because a mis-wired or silent channel, a clipped
channel or a knock on the bench is obvious in the raw samples and invisible in a summary number.
Then process that one take (a single streaming pass, cached) and look at its spectrogram.
'''),
    code('''
# ==== YOUR SETTINGS ====
CONFIG = "../config/demo.yaml"
TAKE = "DEMO_002"            # a take name from notebook 00
T0, T1 = 39.5, 41.5          # a short window [s] for the raw waveform view
'''),
    code(SETUP),
    code('''
takes = za.discover_takes(cfg.data_dir, recursive=cfg.recursive, pattern=cfg.file_pattern)
take = za.find_take(takes, TAKE)
take.describe()
'''),
    md('''
## Raw waveforms

Units are the recorded sample value (FS = digital full scale). Files are 32-bit float, so values
above 1 are legal and are **not** clipping. `highpass_hz` removes rumble so small transients are visible.
'''),
    code('''
fig = zp.plot_waveforms(take, T0, T1, cfg=cfg)
fig = zp.plot_waveforms(take, T0, T1, cfg=cfg, highpass_hz=1000)
'''),
    md('''
## Process this take

This reads the files once, block by block, and writes the feature bundle to `cache_dir/<take>/`.
Running it again does nothing unless the files or the settings changed.
'''),
    code('''
za.process_take(take, cfg)
f = za.load_features(cfg, TAKE)
print("channels:", f.channels, " bands:", f.bands)
print("peak |sample| per channel:", np.round(f.meta["peak_abs"], 4))
print("samples with |x| > 1 (legal in float files, check if unexpected):", f.meta["n_samples_over_unity"])
'''),
    md('''
## Full range, then zoom

The top panel is the whole take. The boxes are zooms chosen from the analysis (quiet baseline, onset,
loudest 30 s, a significant glide, the strongest single event). The event zoom is recomputed from the
**raw audio**, because a 0.25 s spectrogram frame is longer than the event. Grey gaps are masked time
(chirps). Change `CH` to look at another sensor.
'''),
    code('''
CH = f.channels[0]
zooms = za.views.suggest_zooms(f, CH, baseline=(T0 - 30, T0 - 5) if T0 > 35 else (1, 10),
                               fmax=min(24000, f.sr / 2))
fig = za.views.plot_overview_zoom(f, CH, zooms, take=take, fmax=min(24000, f.sr / 2))
'''),
    code('''
fig = zp.plot_spectrogram(f, fmax=min(24000, f.sr / 2))
'''),
    code('''
fig = zp.plot_levels(f, bands=[b for b in ("low", "audio", "rig") if b in f.bands], dt=1.0)
'''),
    md('''
**Look for:** channels that never change (disconnected?), steps where something was moved, periodic
spikes (active chirps: mask them, see notebook 03), and the reaction onset.
'''),
]

# ============================================================================ 02
NOTEBOOKS["02_process_all_takes"] = [
    md('''
# 02 · Process every take

One streaming pass per take, one take at a time (keeps memory at a few hundred MB even for multi-GB files).
Up-to-date bundles are skipped, so re-running is cheap. Use `force=True` after changing the DSP settings
if you want to be sure, although changed settings are detected automatically.

Command-line equivalent (useful for long runs, e.g. overnight):

```bash
python -m zoom_acoustics process config/local_myexperiment.yaml
```
'''),
    code('''
# ==== YOUR SETTINGS ====
CONFIG = "../config/demo.yaml"
ONLY = None          # e.g. ["DEMO_002", "DEMO_003"] to process a subset
FORCE = False
'''),
    code(SETUP),
    code('''
takes = za.discover_takes(cfg.data_dir, recursive=cfg.recursive, pattern=cfg.file_pattern)
if ONLY:
    takes = [za.find_take(takes, n) for n in ONLY]
za.process_all(takes, cfg, force=FORCE)
'''),
    md('## Summary of every processed take'),
    code('''
feats = za.load_all_features(cfg)
za.timeline.takes_overview(feats)
'''),
    code('''
summary = pd.concat([f.summary() for f in feats], ignore_index=True)
summary.to_csv(cfg.figures_path / "summary_all_takes.csv", index=False)
summary
'''),
]

# ============================================================================ 03
NOTEBOOKS["03_single_take_analysis"] = [
    md('''
# 03 · Analysis of one take

Masks → spectrogram → band levels relative to baseline → spectra before/after (and the *excess*) →
onset → events and trigger rate → detector check → figures saved.

Everything here reads the cached bundle, so it is fast. Re-run cells freely.
'''),
    code('''
# ==== YOUR SETTINGS ====
CONFIG = "../config/demo.yaml"
TAKE = "DEMO_002"
BASELINE = (5.0, 34.0)            # [s] quiet reference window BEFORE anything is added
AFTER = (60.0, 170.0)             # [s] window to compare with the baseline
MARKERS = {"drop": 35.0, "acid": 40.0}   # events you noted in the lab book (s from take start)
FMAX = 24000                      # highest frequency to show [Hz]
REF_CHANNEL_FOR_CHIRPS = 4        # the channel that hears the active source best (air mic)
FIND_CHIRPS = False               # True: locate a periodic source automatically and mask it
CHIRP_PERIOD_S = 10.0             # known period from the signal-generator log (None = fit it)
'''),
    code(SETUP),
    code('''
f = za.load_features(cfg, TAKE)
print(f.take, f"{f.duration_s:.1f} s", "channels", f.channels, "sr", f.sr)
print("masks from config:", len(f.mask_windows), "windows")
'''),
    md('''
## 1. Masks (active chirps, known disturbances)

If a periodic active source was running, every passive number must exclude it. Either put it in the
config (`takes: <take>: periodic: {...}`) **before** processing (best: the detector then ignores it too),
or find it here and add it after the fact. Grey bands in every plot are masked time.
'''),
    code('''
if FIND_CHIRPS and REF_CHANNEL_FOR_CHIRPS in f.channels:
    g = za.masks.find_periodic_bursts(f.t_level, f.level(REF_CHANNEL_FOR_CHIRPS, "broadband"),
                                      period_s=CHIRP_PERIOD_S)
    print(g)
    if g:
        # the chirp sweep starts at 100 Hz; pad generously before the detected rise
        f.add_masks(za.masks.periodic_windows(g["first_s"], g["period_s"], f.duration_s,
                                              pad_before_s=0.5, pad_after_s=1.5))
live = za.masks.live_seconds(0, f.duration_s, f.mask_windows)
print(f"masked: {100*(1-live/f.duration_s):.1f} % of the take")
'''),
    md('## 2. Spectrogram'),
    code('''
fig = zp.plot_spectrogram(f, fmax=FMAX, markers=MARKERS)
zp.save(fig, cfg, f"{TAKE}_spectrogram")
'''),
    md('''
### Full range + automatic zooms, and a difference image

`clim=(lo, hi)` fixes the colour range, so use the same values for every take you compare.
`background=` turns the picture into a **difference image** against a quiet part: white = unchanged,
red = added. For a separate background take, pass `background_feat=za.load_features(cfg, "<take>")`.
`units="pa"` shows dB re 1 µPa²/Hz if the channel has `pa_per_fs` in the config.
'''),
    code('''
take = za.find_take(za.discover_takes(cfg.data_dir, recursive=cfg.recursive, pattern=cfg.file_pattern), TAKE)
CH = f.channels[0]
zooms = za.views.suggest_zooms(f, CH, baseline=BASELINE, fmax=FMAX)
pd.DataFrame([{k: v for k, v in z.items() if k != "track"} for z in zooms])
'''),
    code('''
fig = za.views.plot_overview_zoom(f, CH, zooms, take=take, fmax=FMAX, markers=MARKERS)
zp.save(fig, cfg, f"{TAKE}_overview_zoom")
'''),
    code('''
fig = za.views.plot_overview_zoom(f, CH, zooms, take=take, fmax=FMAX, markers=MARKERS, background=BASELINE)
zp.save(fig, cfg, f"{TAKE}_difference")
'''),
    md('''
Zoom by hand anywhere: a list of boxes `dict(name, t0, t1, f0, f1, why)`, or a single raw-audio view of
one event.
'''),
    code('''
my_zooms = [dict(name="early reaction", t0=AFTER[0], t1=AFTER[0] + 20, f0=1000, f1=10000, why="hand-picked"),
            dict(name="late", t0=AFTER[1] - 20, t1=AFTER[1], f0=1000, f1=10000, why="hand-picked")]
fig = za.views.plot_overview_zoom(f, CH, my_zooms, fmax=FMAX, background=BASELINE, clim=(-30, 30))
ev = f.events[(f.events.channel == CH) & (f.events.t_s > AFTER[0])]
if len(ev):
    t_ev = float(ev.t_s.iloc[0])
    fig = za.views.plot_raw_zoom(take, CH, t_ev - 0.01, t_ev + 0.05, fmax=FMAX, cfg=cfg)
'''),
    md('''
## 3. Band levels, as change relative to the baseline

Absolute dB values are **uncalibrated** (dB re digital full scale) and differ between sensors for
reasons unrelated to the sample. The change relative to a quiet baseline, within one channel, is the robust quantity.
'''),
    code('''
bands = [b for b in ("low", "audio", "rig", "hb1", "hb2", "hb3") if b in f.bands]
fig = zp.plot_levels(f, bands=bands, dt=1.0, markers=MARKERS, relative_to=BASELINE)
zp.save(fig, cfg, f"{TAKE}_levels")
'''),
    md('''
## 4. Spectra: baseline vs after, and the excess

The dotted line is `after − baseline` in **linear power**: the spectrum of what was *added*. Grey = the
`rig` band (a 5.6 kHz apparatus resonance in the Sep-2026 setup; check whether yours has one in a
water-only take).
'''),
    code('''
fig = zp.plot_psd(f, {"baseline": BASELINE, "after": AFTER}, fmax=FMAX, fmin=100, excess=True)
zp.save(fig, cfg, f"{TAKE}_spectra")
'''),
    md('''
## 5. Onset

First time the audio-band level rises 6 dB above baseline **and stays up** for 10 s. The hold
requirement is what rejects a single knock or the sample being dropped in.
'''),
    code('''
rows = []
for c in f.channels:
    on, base = f.onset(c, "audio", base_window=BASELINE, rise_db=6, hold_s=10)
    rows.append(dict(channel=c, label=f.label(c), onset_s=on, baseline_dB=dsp.db(base)))
onsets = pd.DataFrame(rows); onsets
'''),
    md('''
## 6. Events (STA/LTA triggers) and trigger rate

A trigger is a short transient that stands out from the preceding 60 ms. **A trigger rate is not a
bubble rate:** at high bubble rates triggers merge and the LTA is inflated, so the rate saturates
(see `docs/VALIDATION.md`, R1/R2). Compare rates within a channel, over time.
'''),
    code('''
det = f.meta["detect_bands"][0] if f.meta["detect_bands"] else None
if det:
    fig = zp.plot_event_rate(f, det, bin_s=5.0, markers=MARKERS)
    zp.save(fig, cfg, f"{TAKE}_trigger_rate")
'''),
    md('''
### Check the detector by eye

Before trusting any count, look at what the detector saw in a short window: envelope, STA/LTA ratio,
thresholds and the triggers. Pick a window inside the reaction.
'''),
    code('''
if det and f.meta["env_stored"]:
    ch0 = f.channels[0]
    t_chk = (AFTER[0], AFTER[0] + 0.5)
    fig = zp.plot_detector(f, ch0, det, *t_chk)
'''),
    md('''
### Re-run the detector with other settings (no need to re-process)

Only possible when the fine envelope was stored (`env_stored` is True).
'''),
    code('''
if det and f.meta["env_stored"]:
    for on in (5, 8, 12):
        ev, live = f.redetect(f.channels[0], det, on=on)
        print(f"on={on:>2}: {len(ev)} triggers, {len(ev)/live:.1f} per live second")
'''),
    md('## 7. Summary table for this take'),
    code('''
tab = za.timeline.window_table([f], {"baseline": BASELINE, "after": AFTER}, bands=bands)
tab.pivot_table(index=["channel", "label"], columns=["band", "window"], values="level_dB").round(1)
'''),
    code('''
tab[tab["window"] == "after"].pivot_table(index=["channel", "label"], columns="band", values="delta_dB").round(1)
'''),
    md('''
## 8. Save the standard figure set

One call writes every standard figure for this take (overview + zooms, difference image, waveforms,
spectrogram, levels, spectra, trigger rate, detector, inter-event gaps, glide, glide interpretation) plus a
summary CSV and an `INDEX.md` page describing each figure, into `figures_dir/<take>_figure_set/`.
Examples for real data are in `examples/`.
'''),
    code('''
paths, summ = za.report.save_figure_set(cfg, TAKE, baseline=BASELINE, after=AFTER, markers=MARKERS,
                                        fmax=FMAX, glide_seed=(4000, 8000))
print("wrote", len(paths), "files to", paths[0].parent)
summ.round(2)
'''),
]

# ============================================================================ 04
NOTEBOOKS["04_time_variation_and_pH"] = [
    md('''
# 04 · Variation over time, and pH

Puts every processed take on one **lab-clock** axis (hours, days), loads a pH time series, aligns the
two and compares them.

Two clocks are involved (recorder, pH logger). Put their offsets in the config
(`recorder_clock_offset_s`, `ph: clock_offset_s`). A wrong offset shifts one curve against the other and
changes every correlation below. Check it with a known event (the acid addition is usually the easiest one).
'''),
    code('''
# ==== YOUR SETTINGS ====
CONFIG = "../config/demo.yaml"
CHANNELS = None             # None = every channel any take has; or e.g. [1, 2]
BAND = "audio"
DT_S = 10.0                 # time bin for the timeline [s]
BASELINE_TAKE = "DEMO_002"  # take + window used as the quiet reference for EXCESS power
BASELINE = (5.0, 34.0)
PH_WINDOW_S = 10.0          # acoustic average around each pH sample [s]
EVENT_TIMES = {"acid": "2026-10-01 10:02:40"}   # lab-clock times to mark (from the lab book)
'''),
    code(SETUP),
    code('''
feats = za.load_all_features(cfg)
za.timeline.takes_overview(feats)
'''),
    code('''
chans = CHANNELS or sorted({c for f in feats for c in f.channels})
series = {}
for c in chans:
    s = za.timeline.level_timeline(feats, c, BAND, dt=DT_S)
    if s.notna().any():
        series[c] = s
lab = {c: next(f.label(c) for f in feats if c in f.channels) for c in series}
print({c: lab[c] for c in series})
'''),
    md('''
## Load pH

The file is described in the `ph:` section of the config. Auto-detection handles a timestamp column,
separate `Date` + `Time` columns, or elapsed time plus a start. If it guesses wrong, name the columns
explicitly there.
'''),
    code('''
ph = za.ph.load_ph_from_config(cfg)
print(len(ph), "pH samples from", ph.index[0], "to", ph.index[-1])
ph.head()
'''),
    md('''
## Timeline: acoustic level per channel, trigger rate, pH (stacked, shared time axis)

Each quantity gets its own panel. There is no twin y-axis on purpose: two scales on one plot invite
reading a relation into the choice of axis limits.
'''),
    code('''
marks = {k: pd.Timestamp(v) for k, v in EVENT_TIMES.items()}
panels = {f"{lab[c]}: {BAND} level [dB re FS²]": [(lab[c], dsp.db(s), zp.ch_color(c))] for c, s in series.items()}
c0 = chans[0]
rate = za.timeline.rate_timeline(feats, c0, BAND, bin_s=DT_S)
if rate.notna().any():
    panels[f"{lab[c0]}: trigger rate [1/s]"] = [("rate", rate, zp.ch_color(c0))]
extra = [c for c in ph.columns if c != "pH"][:1]
fig = zp.plot_timeline(panels, ph=ph, markers=marks, extra_ph_cols=extra,
                       title=f"{cfg.project}: {BAND} band, {DT_S:g} s bins, all takes")
zp.save(fig, cfg, "timeline_all")
'''),
    md('''
## Excess power

Subtract the quiet baseline (in **linear** power) before comparing with pH. The raw level contains the
noise floor and any apparatus tone. Those add a constant that flattens every trend near the floor (in the
demo it moves the slope from −10.5 to −8.9 dB per pH unit).
'''),
    code('''
fb = za.load_features(cfg, BASELINE_TAKE)
excess = {}
for c, s in series.items():
    if c in fb.channels:
        base = fb.band_level_in_window(c, BAND, *BASELINE)
        excess[c] = (s - base).clip(lower=1e-30)
        print(f"{lab[c]}: baseline {dsp.db(base):.1f} dB re FS²")
'''),
    md('## Align with pH and compare'),
    code('''
t_first = min(marks.values()) if marks else ph.index[0]
ph_use = ph[ph.index > t_first + pd.Timedelta(seconds=20)]   # after the reaction starts
figs = {}
for c, s in excess.items():
    al = za.ph.align(ph_use, s, window_s=PH_WINDOW_S)
    if len(al) < 3:
        print(lab[c], ": fewer than 3 pH samples overlap with recordings"); continue
    fig = zp.plot_ph_scatter(al, label=f"{lab[c]} excess")
    fig.axes[0].set_title(lab[c], loc="left")
    zp.save(fig, cfg, f"level_vs_pH_ch{c}")
    rho = al["pH"].corr(al["acoustic_dB"], method="spearman")
    print(f"{lab[c]}: n={len(al)}, Spearman rho={rho:.3f}")
'''),
    md('''
### Lag between pH and sound

If the pH probe sits away from the sample, or responds slowly, pH lags the acoustics. The lag with the
largest |r| is a *hint*, not a measurement: with smooth, autocorrelated series many lags correlate well.
Only lags covered by recordings count (see `n`).
'''),
    code('''
c = next(iter(excess))
corr = za.ph.lagged_correlation(dsp.db(excess[c]).rename("ac"), ph_use["pH"], dt_s=DT_S, max_lag_s=300)
fig = zp.plot_lag(corr, title=f"{lab[c]} excess level vs pH")
corr.loc[corr.r.abs().idxmax()] if corr.r.notna().any() else corr
'''),
    md('## Export the aligned table (for your own plots, Excel, etc.)'),
    code('''
out = pd.DataFrame({"pH": ph["pH"]})
for c, s in excess.items():
    out[f"ch{c}_{BAND}_excess_dB"] = za.ph.acoustic_at(ph.index, s, PH_WINDOW_S).to_numpy()
out.to_csv(cfg.figures_path / "ph_acoustic_aligned.csv")
out.dropna().head()
'''),
]

# ============================================================================ 05
NOTEBOOKS["05_compare_takes"] = [
    md('''
# 05 · Compare takes

Same channel, different takes: spectra, band-level changes, and trigger rates side by side. Use it for
before/after-treatment, different concentrations, different days.

**Only compare like with like.** Absolute levels between takes mean something only if no sensor was
moved, re-mounted or re-gained in between. The change relative to each take's own baseline
(`delta_dB`) is much more robust.
'''),
    code('''
# ==== YOUR SETTINGS ====
CONFIG = "../config/demo.yaml"
TAKES = None                  # None = all processed takes, or a list of names
CHANNEL = 1
BAND = "audio"
WINDOWS = {"early": (5.0, 30.0), "late": (60.0, 110.0)}   # [s] from each take's start
FMAX = 24000
'''),
    code(SETUP),
    code('''
feats = za.load_all_features(cfg)
if TAKES:
    feats = [f for f in feats if f.take in TAKES]
[f.take for f in feats]
'''),
    code('''
fig = zp.plot_compare_psd(feats, CHANNEL, fmax=FMAX)
zp.save(fig, cfg, f"compare_psd_ch{CHANNEL}")
'''),
    code('''
tab = za.timeline.window_table(feats, WINDOWS, bands=[BAND])
tab.pivot_table(index=["take"], columns=["channel", "window"], values="level_dB").round(1)
'''),
    code('''
fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
sub = tab[tab["window"] == list(WINDOWS)[-1]]
for c in sorted(sub["channel"].unique()):
    s = sub[sub["channel"] == c]
    ax.plot(s["take"], s["delta_dB"], "o-", color=zp.ch_color(c), label=f"Ch{c}")
ax.set_ylabel(f"{list(WINDOWS)[-1]} minus {list(WINDOWS)[0]} [dB]")
ax.set_title(f"{BAND} band change within each take", loc="left")
ax.legend(); ax.axhline(0, color="0.6", lw=0.6)
zp.save(fig, cfg, "compare_delta")
'''),
    code('''
rows = []
for f in feats:
    if CHANNEL in f.channels and BAND in f.meta["detect_bands"]:
        for w, (a, b) in WINDOWS.items():
            t, r, n, live = f.event_rate(CHANNEL, BAND, bin_s=1.0)
            m = (t >= a) & (t < b)
            rows.append(dict(take=f.take, window=w, triggers=int(n[m].sum()),
                             live_s=float(live[m].sum()),
                             rate_per_s=float(n[m].sum() / max(live[m].sum(), 1e-9))))
pd.DataFrame(rows).pivot_table(index="take", columns="window", values="rate_per_s").round(2)
'''),
]

# ============================================================================ 06
NOTEBOOKS["06_event_catalogue_and_clustering"] = [
    md('''
# 06 · Event catalogue and clustering

From triggers to a catalogue, and three separate meanings of "clustering":

1. **In time:** are events bunched more than a random (Poisson) process? Inter-event CV, Fano factor, burst sweep.
2. **Across channels:** do two sensors see the same events? Coincidence compared with a time-shifted chance level.
3. **By waveform:** are some events copies of each other (a rattling fitting, a dripping tap)? Cross-correlation
   clustering, plus per-event features (peak frequency, decay time, Q, energy) and feature-space clustering.

Read `docs/STUDENT_GUIDE.md` §7 and §12 first. Triggers are not bubbles, and correlation clusters of *bubbles*
are expected (similar-frequency ringdowns look alike).
'''),
    code('''
# ==== YOUR SETTINGS ====
CONFIG = "../config/demo.yaml"
TAKE = "DEMO_002"
CH = 3                       # channel for the waveform catalogue
BAND = "audio"               # detection band
WINDOW = (50.0, 175.0)       # [s] part of the take to analyse (e.g. after the acid)
MAX_EVENTS = 600             # waveforms to read (random subset if more)
PRE_S, POST_S = 0.002, 0.010 # snippet: 2 ms before to 10 ms after the trigger
RHO_MIN = 0.8                # correlation threshold for waveform clusters
N_FEATURE_CLUSTERS = 3
'''),
    code(SETUP),
    code('''
f = za.load_features(cfg, TAKE)
take = za.find_take(za.discover_takes(cfg.data_dir, recursive=cfg.recursive, pattern=cfg.file_pattern), TAKE)
from zoom_acoustics import catalogue as K
ev = f.events
ev = ev[(ev.band == BAND) & (ev.t_s >= WINDOW[0]) & (ev.t_s < WINDOW[1])]
ev = ev[~za.masks.mask_array(ev.t_s.to_numpy(), f.mask_windows)]
ev.groupby("channel").size().rename("triggers in window")
'''),
    md('''
## 1. Clustering in time

CV = 1 and Fano = 1 for a Poisson process. Gaps that straddle a masked window are removed (they are holes
cut by the mask, not intervals). A burst count is meaningful only if it has a **plateau** over a range of gap values.
'''),
    code('''
rows = []
for c in f.channels:
    t = ev[ev.channel == c].t_s.to_numpy()
    st = K.interevent_stats(t, windows=f.mask_windows, dead_s=f.meta["detector"]["dead_s"], t_range=WINDOW)
    rows.append(dict(channel=c, label=f.label(c), **{k: v for k, v in st.items()}))
pd.DataFrame(rows).round(3)
'''),
    code('''
t = ev[ev.channel == CH].t_s.to_numpy()
fig = zp.plot_interevent(t, f.mask_windows, f.meta["detector"]["dead_s"], title=f"{f.label(CH)}")
zp.save(fig, cfg, f"{TAKE}_interevent_ch{CH}")
K.burst_sweep(t, windows=f.mask_windows)
'''),
    md('''
## 2. Coincidence between channels

For each pair: the fraction of A's events with a B event within ±3 ms, compared with the same after shifting B by
several seconds (chance). Only the **excess** means anything. The lag is B minus A. A constant lag of a fraction of
a millisecond between two sensors a few cm apart is mostly instrument group delay, not travel time.
'''),
    code('''
rows = []
for a in f.channels:
    for b in f.channels:
        if a < b:
            r = K.coincidence(ev[ev.channel == a].t_s, ev[ev.channel == b].t_s, win_s=3e-3)
            rows.append(dict(A=a, B=b, **{k: v for k, v in r.items() if k != "lags_s"}))
pd.DataFrame(rows).round(3)
'''),
    md('''
## 3. Waveforms, features and waveform clusters

Snippets are read from the WAV files (not the cache). Features per event: peak amplitude, energy, rise and decay
time, peak and centroid frequency, bandwidth, Q = π f τ, and the level of the second spectral peak.
'''),
    code('''
W, tt, sr = K.extract_waveforms(take, ev[ev.channel == CH].t_s, CH, PRE_S, POST_S, max_events=MAX_EVENTS)
feat_df = K.waveform_features(W, sr, PRE_S)
feat_df.insert(0, "t_s", tt)
feat_df.describe().round(3)
'''),
    code('''
C = K.xcorr_matrix(W)
lab, clusters, med = K.cluster_waveforms(C, rho_min=RHO_MIN, features=feat_df, times=tt)
print(f"median pairwise correlation of all events: {med:.2f}")
clusters.head(10).round(2)
'''),
    md('''
**How to read this table.** Clusters are ranked by internal similarity (`median_rho`). A repeating mechanical
source stands out only if it is clearly *more* similar than the rest. Check `f_peak_hz`: a cluster on an apparatus
line (5.6 kHz in the Sep-2026 rig) is the line. `gap_cv` well below 1 means regular timing. Then **look at the
gallery.**
'''),
    code('''
top = clusters.cluster.head(5).tolist()
lab_show = np.where(np.isin(lab, top), lab, 0)
fig = zp.plot_waveform_gallery(W[lab_show > 0], sr, lab_show[lab_show > 0], pre_s=PRE_S,
                               title=f"{TAKE} {f.label(CH)}: five most self-similar clusters")
zp.save(fig, cfg, f"{TAKE}_gallery_ch{CH}")
fig = zp.plot_corr_matrix(C, lab)
'''),
    md('''
### Feature-space clustering (descriptive)

Ward clustering of standardised log-features. **k is your choice**, so re-run with other k and other feature
sets, and only trust groups that persist.
'''),
    code('''
flab, scores, evr = K.feature_clusters(feat_df, n_clusters=N_FEATURE_CLUSTERS)
fig = zp.plot_feature_space(feat_df, flab, scores, evr)
zp.save(fig, cfg, f"{TAKE}_features_ch{CH}")
feat_df.assign(cluster=flab).groupby("cluster")[["f_peak_hz", "decay_ms", "q_est", "energy"]].median().round(4)
'''),
    code('''
out = feat_df.assign(xcorr_cluster=lab, feature_cluster=flab, channel=CH, take=TAKE)
out.to_csv(cfg.figures_path / f"{TAKE}_catalogue_ch{CH}.csv", index=False)
print("wrote", cfg.figures_path / f"{TAKE}_catalogue_ch{CH}.csv", len(out), "events")
'''),
]

# ============================================================================ 07
NOTEBOOKS["07_glide_analysis"] = [
    md('''
# 07 · Glides: descending spectral ridges

After the acid, a band in the **excess spectrogram** (each frame divided by a quiet baseline spectrum) may
descend smoothly in frequency (Sep 2026: 6.5 → 0.9 kHz in 328 s on one hydrophone, never on the air mic). This
notebook tracks it per channel, tests whether it is real (contrast), checks harmonics, and follows it across takes.

Read `docs/STUDENT_GUIDE.md` §13. The mechanism is not established, and a Minnaert radius read off the ridge
is shown only with its Bond number, as a feasibility check.
'''),
    code('''
# ==== YOUR SETTINGS ====
CONFIG = "../config/demo.yaml"
TAKE = "DEMO_002"
BASE_TAKE = "DEMO_002"        # take that holds the quiet baseline (same sensors, same geometry)
BASELINE = (5.0, 34.0)        # [s] baseline window inside BASE_TAKE
T0 = 46.0                     # [s] start tracking here (after the onset)
SEED = (4000.0, 12000.0)      # [Hz] where to pick up the ridge at T0
FMAX = 20000
LATER_TAKES = ["DEMO_003", "DEMO_004"]   # follow the ridge into later takes (baseline from BASE_TAKE)
'''),
    code(SETUP),
    code('''
from zoom_acoustics import glide as G
f = za.load_features(cfg, TAKE)
fb = za.load_features(cfg, BASE_TAKE)
tracks, rows = {}, []
for c in f.channels:
    tr = G.track_ridge(f, c, BASELINE, T0, seed=SEED, base_feat=fb)
    s = G.summarize(tr)
    if s:
        tracks[c] = tr
        rows.append(dict(channel=c, label=f.label(c), **s))
summary = pd.DataFrame(rows)
summary[["channel", "label", "f_start_hz", "f_end_hz", "octaves", "octaves_per_min",
         "median_contrast_db", "significant", "t_lock_s", "frac_locked",
         "minnaert_R_end_mm", "bond_end", "free_bubble_feasible_at_end"]].round(3)
'''),
    md('''
**Read `significant` first.** A tracker always returns a path, even through pure noise, and a noise path drifts.
Only tracks with median contrast well above ~3 dB are ridges. The air mic is the control: if it "glides" too with
high contrast, the effect is not liquid-borne. `t_lock_s` later than `T0` means the seed started on something else.
'''),
    code('''
show = int(summary.sort_values("median_contrast_db").channel.iloc[-1]) if len(summary) else f.channels[0]
fig = zp.plot_glide(f, tracks, BASELINE, show_ch=show, fmax=FMAX, base_feat=fb)
zp.save(fig, cfg, f"{TAKE}_glide")
'''),
    md('''
## Harmonics along the ridge

Is there a line at m × f(t)? Non-integer multiples (1.5×, 2.5×) are the noise controls. A strong 0.5× means the
tracked ridge is itself a **harmonic**: re-seed lower.
'''),
    code('''
G.harmonic_ladder(f, show, BASELINE, tracks[show], base_feat=fb).round(2)
'''),
    md('''
## Following the ridge into later takes

Each later take is tracked with the baseline of `BASE_TAKE`, seeded near where the ridge was last seen. Only
valid if nothing was moved or re-gained between takes.
'''),
    code('''
all_rows = [dict(take=TAKE, **r) for r in rows]
last = {r["channel"]: r["f_end_hz"] for r in rows}
for name in LATER_TAKES:
    fl = za.load_features(cfg, name)
    for c in fl.channels:
        if c not in last:
            continue
        tr = G.track_ridge(fl, c, BASELINE, 1.0, seed=(0.7 * last[c], 1.3 * last[c]), base_feat=fb)
        s = G.summarize(tr)
        if s:
            all_rows.append(dict(take=name, channel=c, label=fl.label(c), **s))
            if s["significant"]:
                last[c] = s["f_end_hz"]
tab = pd.DataFrame(all_rows)
tab[["take", "channel", "f_start_hz", "f_end_hz", "octaves", "median_contrast_db", "significant"]].round(2)
'''),
    code('''
tab.to_csv(cfg.figures_path / "glide_summary.csv", index=False)
print("wrote", cfg.figures_path / "glide_summary.csv")
'''),
    md('''
## Interpretation: what would each mechanism need?

`glide.interpret` turns the ridge f(t) into the parameter each candidate mechanism requires
(`docs/GLIDE_INTERPRETATION.md` has the physics):

* **A, one growing bubble:** Minnaert radius R(t) and growth rate. Bond number ≥ 1 (R above the 2.73 mm
  capillary length) means impossible as a free sphere. A bubble on a wall rings 0.816× lower, so it can be
  smaller.
* **B, a bubbly layer:** Wood's-law void fraction for assumed layer thicknesses. Gaps mean no void fraction
  ≤ 2 % works.

This does not identify the mechanism. It shows which readings are physically possible. Discriminating
tests (depth dependence, sensor dependence, odd vs integer harmonics, a camera) are in the doc.
'''),
    code('''
best = int(summary.sort_values("median_contrast_db").channel.iloc[-1])
it = G.interpret(tracks[best], depths_m=(0.005, 0.01, 0.02, 0.04))
fig = zp.plot_glide_interpretation(it, f"{TAKE} {f.label(best)}")
zp.save(fig, cfg, f"{TAKE}_glide_interpretation")
G.interpretation_summary(it)
'''),
]

# ============================================================================ 08
NOTEBOOKS["08_acoustics_vs_pH_correlation"] = [
    md('''
# 08 · Correlating sound with pH

Build acoustic time series (the level in **any narrow band you choose**, excess over a background, or the
trigger rate of all or only loud events), and pH or **dpH/dt**. Then correlate them honestly:

* **n_eff**: two smooth series correlate whatever the mechanism. The effective number of independent pairs
  (Bretherton et al. 1999) sets the p-value, not n.
* **differences=True**: correlate changes against changes, which removes a shared trend.
* **band scan**: which frequency band tracks pH? Look for a coherent range of bands, not a single one.

On the demo (a single decaying reaction) every band correlates at |r| ≈ 0.9 with n_eff ≈ 2, which is
**not significant**. That is the correct answer: one monotonic run cannot establish a relation. Several takes
under different conditions can.
'''),
    code('''
# ==== YOUR SETTINGS ====
CONFIG = "../config/demo.yaml"
CH = 1
BAND_HZ = (3000.0, 6000.0)        # any band, chosen after processing (from the stored spectrogram)
BASELINE_TAKE, BASELINE = "DEMO_002", (5.0, 34.0)
START_AFTER = "2026-10-01 10:03:00"   # use pH after the reaction started
DT_S = 10.0
MAX_LAG_S = 120.0
LOUD_EVENT_DB = -55.0             # "loud" triggers only (envelope dB re FS^2); None = all
'''),
    code(SETUP),
    code('''
from zoom_acoustics import correlate as R
feats = [f for f in za.load_all_features(cfg) if CH in f.channels]
fb = za.load_features(cfg, BASELINE_TAKE)
ph = za.ph.load_ph_from_config(cfg)
ph = ph[ph.index > pd.Timestamp(START_AFTER)]
dph = R.ph_rate(ph, smooth_s=90)
band = R.band_timeline(feats, CH, *BAND_HZ, dt=DT_S)
b0 = R.band_level_from_psd(fb, CH, *BAND_HZ)
base = float(b0[(b0.index >= BASELINE[0]) & (b0.index < BASELINE[1])].mean())
band_ex = R.excess(band, base)
rate_all = R.event_count_timeline(feats, CH, bin_s=DT_S)
rate_loud = R.event_count_timeline(feats, CH, bin_s=DT_S, min_env_db=LOUD_EVENT_DB)
print(f"baseline in {BAND_HZ} Hz: {10*np.log10(base):.1f} dB re FS^2")
'''),
    md('## Level in the chosen band vs pH, and vs dpH/dt'),
    code('''
res, scan = R.correlate(band_ex, ph["pH"], dt_s=DT_S, max_lag_s=MAX_LAG_S)
fig = zp.plot_correlation_panel(band_ex, ph["pH"], res, x_label=f"excess {BAND_HZ[0]/1e3:g}-{BAND_HZ[1]/1e3:g} kHz [dB]",
                                title=f"Ch{CH} band excess vs pH")
zp.save(fig, cfg, "corr_band_vs_pH")
pd.Series(res).drop("note", errors="ignore").to_frame("value")
'''),
    code('''
res_d, _ = R.correlate(band_ex, dph, dt_s=DT_S, max_lag_s=MAX_LAG_S)
res_diff, _ = R.correlate(band_ex, ph["pH"], dt_s=DT_S, differences=True)
pd.DataFrame({"vs pH": res, "vs dpH/dt": res_d, "changes vs changes": res_diff}).T[
    ["lag_s", "pearson_r", "spearman_rho", "slope_y_per_dB", "n", "n_eff", "p_value_neff", "note"]]
'''),
    md('## Trigger rate (all, and loud events only) vs pH'),
    code('''
rows = {}
for name, s in (("all triggers", rate_all), (f"triggers > {LOUD_EVENT_DB} dB", rate_loud)):
    r, _ = R.correlate(s, ph["pH"], dt_s=DT_S, x_in_db=False)
    rows[name] = r
fig = zp.plot_correlation_panel(rate_all, ph["pH"], rows["all triggers"], x_label="triggers / s",
                                x_in_db=False, title=f"Ch{CH} trigger rate vs pH")
pd.DataFrame(rows).T[["pearson_r", "spearman_rho", "n", "n_eff", "p_value_neff", "note"]]
'''),
    md('''
## Which band tracks pH? (third-octave scan, excess over the baseline)

Filled markers: p(n_eff) < 0.05. With 16 bands, about 1 passes by chance. The lower panel is the best lag
per band: if it jumps around, the lag is noise.
'''),
    code('''
scan = R.band_scan(feats, CH, ph["pH"], bands=R.log_bands(500, 20000, 3), dt_s=DT_S,
                   max_lag_s=MAX_LAG_S, baseline=(fb, BASELINE))
fig = zp.plot_band_scan(scan, f"Ch{CH}: third-octave excess vs pH")
zp.save(fig, cfg, "corr_band_scan")
scan[["f_lo", "f_hi", "lag_s", "pearson_r", "slope_y_per_dB", "n", "n_eff", "p_value_neff"]].round(3)
'''),
    code('''
out = pd.DataFrame({"pH": ph["pH"], "dpH_dt_per_min": dph})
out[f"ch{CH}_{BAND_HZ[0]:g}_{BAND_HZ[1]:g}Hz_excess_dB"] = 10 * np.log10(
    za.ph.acoustic_at(ph.index, band_ex, DT_S, how="mean").to_numpy())
out.to_csv(cfg.figures_path / "ph_band_aligned.csv")
out.dropna().head()
'''),
]

# ============================================================================ 09
NOTEBOOKS["09_real_example_sep2026"] = [
    md('''
# 09 · A real example: Sep-2026 calcite in HCl (takes 260910_011 and 260910_016)

This is what **real** data look like, and what the standard analysis gives. The notebook was executed on the
lab machine where the recordings live (1.2 GB each, not in the repository), and its outputs are saved here,
so you can read it without the files. The same figure sets are in `examples/real_260910_011/` and
`examples/real_260910_016/`.

Setup: 4 channels at 192 kHz (two hydrophones on opposite sides of the crystal, a contact sensor bonded to
it, an air mic as veto). An active chirp every 10 s is masked. Take 011 is 1 M HCl, with the sample dropped
at ~33 s, acid at ~45 s and a second addition at ~250 s. Take 016 is 0.5 M.
'''),
    code('''
# ==== YOUR SETTINGS ====
CONFIG = "../config/real_example_sep2026.yaml"
'''),
    code(SETUP),
    code('''
takes = za.discover_takes(cfg.data_dir, recursive=cfg.recursive, pattern=cfg.file_pattern)
za.process_all(takes, cfg, verbose=False)
za.takes_table(takes)
'''),
    md('## Take 011: full range, zooms, difference image'),
    code('''
f = za.load_features(cfg, "260910_011")
take = za.find_take(takes, "260910_011")
MARK = {"sample drop": 33.0, "acid": 45.0, "2nd addition": 250.0}
z = za.views.suggest_zooms(f, 1, baseline=(3, 25))
fig = za.views.plot_overview_zoom(f, 1, z, take=take, markers=MARK)
'''),
    code('''
fig = za.views.plot_overview_zoom(f, 1, z, take=take, markers=MARK, background=(3, 25))
'''),
    md('''
**Read it like this.** Before the acid only the 5.6 kHz apparatus line and low-frequency room noise are
present. At acid contact (~45 s) all wetted channels rise by 13–35 dB in the audio band. Right after the
onset there is a fan of parallel descending ridges. The air mic rises by only ~2 dB, so the sound is
liquid/solid-borne. The contact sensor responds strongly to the second addition at ~250 s.
'''),
    code('''
fig = zp.plot_levels(f, bands=("low", "audio", "rig", "hb1"), dt=1.0, relative_to=(3, 25), markers=MARK)
'''),
    code('''
fig = zp.plot_psd(f, {"baseline": (3, 25), "reaction": (60, 240)}, fmax=40000, fmin=100, excess=True)
'''),
    code('''
fig = zp.plot_event_rate(f, "audio", bin_s=5.0, markers=MARK)
fig = zp.plot_detector(f, 1, "audio", 60.0, 60.5)
'''),
    md('## Take 016: the strongest glide, and what it would require'),
    code('''
from zoom_acoustics import glide as G
f16 = za.load_features(cfg, "260910_016")
tracks, rows = {}, []
for c in f16.channels:
    tr = G.track_ridge(f16, c, (3, 25), 49.5, seed=(4000, 12000))
    s = G.summarize(tr)
    tracks[c] = tr
    rows.append(dict(channel=c, label=f16.label(c), **s))
pd.DataFrame(rows)[["channel", "label", "f_start_hz", "f_end_hz", "octaves", "median_contrast_db", "significant"]].round(2)
'''),
    code('''
fig = zp.plot_glide(f16, tracks, (3, 25), show_ch=1)
'''),
    code('''
it = G.interpret(tracks[1])
fig = zp.plot_glide_interpretation(it, "260910_016 Ch1")
G.interpretation_summary(it)
'''),
    code('''
G.harmonic_ladder(f16, 1, (3, 25), tracks[1]).round(2)
'''),
    md('''
**Interpretation (see docs/GLIDE_INTERPRETATION.md).** Only Ch1 has a significant ridge (6.3 → 1.0 kHz). A
single free bubble would have to exceed the capillary length (Bond > 1) in the late part, so it is not
self-consistent. A wall-attached bubble or a bubbly layer ≥ ~4 cm thick fits the frequencies. Harmonics are
not resolved above the controls. The mechanism is **not** established. The discriminating experiment is to
record the liquid depth and repeat at two depths.
'''),
]


def main(argv=()):
    """--execute: run every notebook (demo data; notebook 09 needs the Sep-2026 files) and
    save it WITH outputs, so figures are visible on GitHub.  --only NAME: just that one."""
    import sys
    argv = list(argv or sys.argv[1:])
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    NB.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        if only and only not in name:
            continue
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata["kernelspec"] = dict(name="python3", display_name="Python 3", language="python")
        if "--execute" in argv:
            from nbclient import NotebookClient
            NotebookClient(nb, timeout=1800, kernel_name="python3",
                           resources={"metadata": {"path": str(NB)}}).execute()
        nbf.write(nb, NB / f"{name}.ipynb")
        print("wrote", NB / f"{name}.ipynb", "(executed)" if "--execute" in argv else "")


if __name__ == "__main__":
    main()
