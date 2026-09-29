# Walkthrough: how the code works, start to finish

This follows one take from the WAV files to a figure, naming the function that does each step. The
equations are in [THEORY.md](../THEORY.md). Read this if you want to change or extend something, or
just to know what you are trusting.

## 1. From files to a `Take` (`zoom_acoustics/io.py`)

`discover_takes(folder)` lists every `.wav` in the folder and groups them by name with one regex:

```
<base>[_TrN][_NNNN].wav      base = take name, TrN = mono track file, NNNN = split part
```

* Files with the same `<base>` belong to one take.
* `_0001`, `_0002`, … are consecutive **time segments** (the recorder splits at ~2 GB). They are joined
  only if they run 0001, 0002, … without gaps **and** each starts (bext time) within 5 s of where the
  previous one ended. Otherwise each file is its own take: `EXP_0011.WAV` and `EXP_0012.WAV` recorded
  hours apart are two takes.
* `_Tr1`, `_Tr2`, … are **channels** recorded as separate mono files. The suffix may come before or
  after the part number.
* Stereo mixdowns (`_LR`, `_TrLR`, `_TrL`, `_TrR`, `_TrMix`) are skipped. If a poly file and mono files
  exist for the same take, the poly file is used.
* With `recursive=True`, same-named takes in different folders become `<folder>__<name>`.

For every file, `read_header()` gets the sample rate, channel count and length from libsndfile
(`soundfile`). It then walks the RIFF chunks itself to read the BWF `bext` chunk (recorder-clock start
date and time) and the `iXML` chunk (track names). `_check_take()` insists that all parts of a take
have the same layout and sample rate.

A `Take` is only a description: nothing has been read yet. It offers:

* `take.read(t0, t1, channels)`: a short window as an array, for waveform plots;
* `take.blocks(n, channels)`: a generator of equal-sized blocks that crosses split-file boundaries
  seamlessly (it keeps a small carry-over buffer).

Channel numbers are 1-based recorder track numbers everywhere in the public API. The only place that
converts to 0-based column indices is `Take._col()`, which keeps off-by-one errors in one spot.

## 2. Settings (`zoom_acoustics/config.py`)

`load_config(yaml)` merges the YAML with `DEFAULTS`, resolves relative paths against the YAML file's
own folder, and validates sensor types. `channel_info(ch, take)` returns name and sensor type, with
per-take overrides applied. `take_opts(take)` returns the take's section (masks, `use_channels`,
periodic source, clock offset).

## 3. One streaming pass (`zoom_acoustics/features.py: process_take`)

`plan()` first turns the settings into integer sample counts for this take's sample rate: level hop
$L$, envelope hop $E$, PSD frame $P$, FFT length $N$. The hops are **nested**: $E$ divides $L$ and $P$,
and the smaller of $L, P$ divides the larger, so the block length (a multiple of $\max(L, P)$ close to
`block_s`) contains whole hops of every kind at any sample rate, and no output sample ever straddles
two blocks. Any hop that had to move by more than 1 % from its requested value is reported in the
plan's notes. Bands above 0.95 Nyquist are dropped, and a band reaching it becomes a high-pass whose stored edge is Nyquist. `plan()` also estimates the output size and
decides whether the fine envelope is stored.

Then a single pass over `take.blocks()`. For each block and each channel:

1. peak |x| and the count of |x| > 1 (diagnostics);
2. for each band: `sosfilt` with the filter state `zi` carried over from the previous block → square →
   mean over each hop of $L$ samples → `level.npy`;
3. for detection bands also: mean over each hop of $E$ → envelope → (masked samples set to NaN) →
   `StaLta.process()` → triggers appended to the event list, and the envelope written to `env.npy`
   if stored;
4. the block is cut into frames of $P$ samples and **one vectorised Welch call** computes all frames'
   PSDs → `psd.npy`.

Outputs are written through `numpy.lib.format.open_memmap`, straight to disk, so the arrays never
need to fit in memory: a day-long recording works the same as a 1-minute one. `meta.json` is written
**last**. A bundle without it is incomplete and is rebuilt. It includes a *fingerprint* (files, frame
count, channels, bands, hops, detector settings, masks). If the fingerprint of the current config
differs from the stored one, the next `process_take` rebuilds automatically, so a changed setting can
never silently reuse an old cache.

## 4. The detector (`zoom_acoustics/dsp.py: StaLta`)

`StaLta` is a small state machine fed one envelope block at a time:

* `ratio(e)` prepends the last $n_l-1$ samples of the previous block (`_tail`, initially NaN) and
  computes both running means from cumulative sums **of finite values only**, with a running count
  of finite values. The ratio for sample $k$ therefore uses windows ending at $k$ that may reach back
  into the previous block. It is NaN where either window is at most half valid.
* `process(e)` walks the threshold crossings: while armed, the next index with $r > $ `on` that is
  at least `dead` samples after the last trigger fires; while disarmed, the next index with
  $r <$ `off` re-arms. `armed` and `_last` persist across calls.

`detect(e, dt)` runs the same class over a whole array. It is used for re-detection from a stored
envelope (`Features.redetect`) and by the tests, so the streaming and offline detectors are the same
code.

## 5. Reading a bundle (`Features`)

`load_features(cfg, take)` memory-maps the arrays. Nothing large is read until you slice it. Methods:

| method | returns |
|---|---|
| `level(ch, band)` | raw mean-square series on the `level_dt` grid (`t_level`) |
| `level_series(ch, band, dt)` | masked, re-binned to `dt` in linear power; `(t, ms, n_used)` |
| `psd(ch)` | memory-mapped `[frames, freqs]` array |
| `psd_mean(ch, t0, t1)` | mean PSD over unmasked frames (chunked, so long takes are fine) |
| `band_level_in_window(ch, band, t0, t1)` | one number, FS² |
| `events` | DataFrame of all triggers |
| `event_rate(ch, band, bin_s)` | live-time-normalised trigger rate, counts, live seconds |
| `redetect(ch, band, on=..., ...)` | re-run the detector on the stored envelope |
| `onset(ch, band, base_window)` | sustained-rise onset |
| `add_masks(windows)` | extra masks applied from now on |
| `abs_time(t)` | seconds → lab-clock datetimes (start + clock offset) |
| `summary()` | one row per channel: medians per band, trigger counts and rates |

## 6. Many takes and pH (`timeline.py`, `ph.py`)

`level_timeline(features, ch, band, dt)` concatenates each take's `level_series` on the lab clock and
inserts a NaN after each take, so plots show gaps instead of bridging them. Takes that lack the channel
are skipped. `rate_timeline` does the same for trigger rates. `window_table` builds a long-format table
of band levels in named windows for every take and channel, with `delta_dB` relative to the first
window.

`ph.load_ph` reads csv/tsv/txt/xlsx, sniffs the delimiter, finds the pH column, and builds a datetime
index from one of: a timestamp column, `Date` + `Time` columns, or elapsed time plus `elapsed_start`.
It then applies `clock_offset_s`. `ph.align` averages the acoustic series around each pH sample (linear
power, ± window/2) and drops samples without acoustic data. `ph.lagged_correlation` resamples both
onto a common grid and computes Spearman correlation against lag.

## 6b. Events, clustering and glides (`catalogue.py`, `glide.py`)

`catalogue.extract_waveforms(take, times, ch)` reads short raw snippets around trigger times directly from
the WAV (a random subset if there are many). `waveform_features` turns them into a table.
`xcorr_matrix` → `cluster_waveforms` gives ranked candidate clusters; `feature_clusters` does Ward
clustering on log-features. `interevent_stats`, `burst_sweep` and `coincidence` work on trigger times
alone. All of them take the mask windows so that gaps across masks are never counted.

`glide.excess_spectrogram` divides each PSD frame by a baseline mean spectrum (optionally from another
take). `track_ridge` follows one ridge under the continuity constraint and records per-frame contrast.
`summarize` finds the locked part, the glide in octaves and its rate, significance, and the
Minnaert/Bond feasibility. `harmonic_ladder` tests m × f(t) against non-integer controls.
`track_ridges` seeds several ridges and flags pairs that merge onto one feature.

## 7. Figures (`plots.py`)

Every function returns a Figure and handles any number of channels (`squeeze=False` subplots, one row
per channel). Colours are fixed per channel number (`ch_color`), so Ch4 is the same colour in a 4-channel
and a 2-channel figure. Masks are drawn as grey bands. There are no twin y-axes: pH always gets its own
panel. `spectrogram_image` block-averages time and frequency in linear power to at most ~1500 × 700
cells, so a day-long spectrogram renders quickly. `save(fig, cfg, name)` writes to `figures_dir`.

## 8. Extending it

* **A new band:** add it to `dsp.bands` in your config. The cache rebuilds automatically.
* **A new per-take quantity:** write a function that takes a `Features` object, and add a test with a
  synthetic signal whose answer you know (see `tests/test_features.py` for the pattern: write a WAV
  with `conftest.write_array`, process it, compare with the analytic value).
* **A new logger (conductivity, temperature):** `ph.load_ph(path, ph_col="Conductivity",
  value_name="Conductivity")` works for any single-column time series.
* **Notebooks** are generated by `tools/build_notebooks.py`. Edit the cell text there and re-run it, so
  all notebooks stay consistent.

## 9. Where it can break

* **Wrong clock offset.** Everything on the lab-clock axis shifts. Check with a known event (§3 of the
  guide).
* **Files renamed by hand** so that the `_NNNN` / `_TrN` pattern no longer matches: parts are then
  treated as separate takes. Keep the recorder's names.
* **A gap in the recording between split parts** (e.g. a card swap): parts are treated as contiguous,
  and a warning is printed if the bext times disagree by more than 1.5 s.
* **The detector at high event rates** undercounts (THEORY §4). The level does not saturate the same
  way, so prefer levels for trends.
* **Bands too narrow for the FFT resolution** (fewer than ~5 bins): band levels from the filter bank
  are still fine, but spectral Q and PSD band integrals are resolution-limited.
* **Processing several huge takes in parallel** multiplies memory. `process_all` is deliberately
  sequential.
