# Student guide: how to get trustworthy results from these recordings

This toolkit came out of four rounds of analysis (Sep 2026) of Zoom F6 recordings of calcite
dissolving in HCl, with two hydrophones, a contact sensor bonded to the crystal and an air
microphone. Several conclusions from those rounds were later **retracted**. Every section below is a
mistake that was actually made, the number that exposed it, and what the code or you must do instead.
Read this before interpreting any figure.

---

## 0. The workflow

1. **Background first.** Record a water-only (or sample-in-water, no acid) take in the exact
   geometry, before every experiment. Ideally record two in a row: the take-to-take difference between
   two untouched backgrounds (1–4 dB in Sep 2026) is the error bar on everything else.
2. `00` inventory → `01` look at raw waveforms → `02` process → `03` one take in depth → `04` time and
   pH → `05` compare.
3. Before believing any number, go back to the raw waveform or the detector plot at that time
   (`plot_waveforms`, `plot_detector`). Every serious bug found in Sep 2026 was found by *looking at
   the data next to the number*, and none was visible in a summary statistic.

## 1. Units: nothing is calibrated

* Levels are **dB re 1 FS²** (FS = digital full scale), and spectra are dB re FS²/Hz. There is no
  pascal or volt calibration for any sensor. No calibrated source was ever measured, so converting
  energy to a physical flux is not possible with this setup.
* **Never compare absolute levels between two sensors**, including the two hydrophones in the same
  dish. In Sep 2026 they differed by up to 24.5 dB in level and 1.21× in peak frequency. The robust
  quantity is the **change within one channel** relative to its own baseline (`relative_to=` in
  `plot_levels`, `delta_dB` in `window_table`), because every unknown gain and coupling factor cancels.
* Files are 32-bit float. **Values above 1.0 are legal and are not clipping.** The bundle reports
  `n_samples_over_unity` so unusual levels are visible, but a test `|x| >= 1` for "clipping" is wrong
  on this data.
* Recorder gain changes between takes break every between-take comparison. Write the gain down.

## 2. Channels

* Channel numbers here are recorder **track numbers** (Ch1 = Tr1). Put names and sensor types in the
  config.
* **Track assignment changes between sessions.** In Sep 2026 the near-crystal hydrophone was Ch2 on
  day 1 and Ch1 on day 2, and an early analysis compared them as if they were the same. Write the
  assignment into the config for every session (per take under `takes:` if it changed mid-session).
* The **air microphone is a veto, not a measurement**. Use it to reject events that are airborne
  (knocks, voices, the sample drop), always as a *ratio* against the liquid channels. It fails by
  construction for bubbles bursting at the free surface, which radiate into the air directly.
* The **contact sensor** has a different transduction and a comb of mount resonances. Do not apply
  bubble physics (Minnaert) to it, and do not compare its absolute level or rate with a hydrophone.
* Check that the sensor is actually in the liquid. In one Sep-2026 protocol **no sensor was
  submerged** and the "hydrophone" results were air measurements. In another take a hydrophone was
  above the water line for the first 252 s (a +11 dB step when it was moved). Photograph the setup and
  write down the liquid level at the start and end.

## 3. Clocks

* The take start time comes from the WAV header (recorder clock, 1 s resolution). In the Sep-2026
  files the header of take 260910_011 reads 15:33:20, while the file's modification time on disk reads
  11:39 local for a file that ends 385 s after it starts. That is a 4 h difference, so the recorder was
  probably on UTC (not confirmed). **Verify, do not assume.** Clap or tap the beaker at a
  wall-clock time you write down, then find the transient in the recording and set
  `recorder_clock_offset_s`.
* pH loggers have their own clock. Photograph both displays together once per session, then set
  `ph: clock_offset_s`.
* A wrong offset silently shifts pH against sound. The acid addition is a good check: the sound
  onset and the pH drop must line up.

## 4. Masks: remove what is not the sample

* **Active chirps.** If a loudspeaker or transducer played sweeps, they are 40–70 dB above the
  reaction, and every passive level, spectrum and rate must exclude them. Put the period and phase in
  the config (`periodic:`) before processing, or find them with `find_periodic_bursts` on the air mic's
  **broadband** level (notebook 03). On real take 260910_011 the finder recovered 39 of 40 chirps on
  the 10.000 s grid, and masking removed 16.7 % of the time. Use the period from the signal-generator
  log if you have it: fitting the period from the audio once gave 4.81 s for a true 10.00 s.
* **Gaps straddling a mask are not intervals between events.** An earlier inter-event statistic
  treated them as intervals, and about 1 % of gaps carried 60–92 % of the variance.
* **The sample drop / crystal drop** is a broadband impulse on all channels (a useful timing marker).
  It must not be read as the reaction onset. The onset rule requires the level to *stay* up for 10 s,
  which is what rejects it.
* Talking, door slams, bench knocks: add them under `takes: <take>: masks:`.

## 5. Spectra

* **Look for apparatus lines in the water-only take.** The Sep-2026 rig had a 5601–5625 Hz line on all
  four channels **with no bubbles present** ($Q$ = 83–213, where a bubble at that frequency has
  $Q\approx 28$). Peak-picking returned it as a spurious 574 µm bubble, and it cost two analysis rounds.
  The `rig` band tracks it so its contribution stays visible. Your setup may have a different line; find
  it in the background take and set the band accordingly.
* **The excess spectrum** (`plot_psd(..., excess=True)`) is `after − baseline` in *linear power*: the
  spectrum of what was added. Do not subtract dB spectra.
* **Band integrals use the rectangle rule** (`dsp.band_power`). The trapezoid rule under-integrates
  narrow bands (0.2 dB for 21 bins). That was found and fixed twice.
* **Above ~20 kHz**, check the channels against their own noise floor before claiming signal. In Sep
  2026 all four channels sat on a common electronic floor there. At 192 kHz there is an anti-alias cliff
  at **89.5 kHz**, and a "peak" there is the filter edge, not a source. The Sep-2026 review
  recommended 96 kHz sampling (half the file size) unless the ultrasonic band is the question.
* Spectral $Q$ is only meaningful if the line is wider than ~3 frequency bins (`dsp.line_q` flags it).

## 6. Bubble size: do not size from a spectral peak

Minnaert's formula ($f_0R \approx 3.17$ m·Hz for CO₂ in water) holds for a free, spherical, isolated
bubble. Per-event peak-frequency sizing was **falsified** on the Sep-2026 data:

* measured ringdown $Q \approx 6$ against $\approx 31$ predicted by the damping budget (5× over-damped,
  in all 28 channel-takes), so what rings is not a free bubble;
* the "largest bubble" sat at the lower edge of the sizing band in 26 of 28 channel-takes (the band
  edge, not a population);
* the two hydrophones in the same dish disagreed on the radius by ×5.9 (median);
* the apparatus line (§5) was sized as a bubble.

The band labels (e.g. "1.2–24 kHz ≈ R 0.13–2.6 mm") say which sizes *could* radiate in a band. They
are not a measurement. For bubble size, use a camera on the crystal.

## 7. Event rates: a trigger is not a bubble

* The detector counts transients that stand out from the preceding 60 ms. At high rates pulses
  overlap and the long-term average is inflated by earlier pulses, so **the count saturates**. On the
  demo (known truth) recall is 97 % at 6–9 bubbles/s and 82 % at 25–40/s, and the decay time fitted to
  the trigger rate is 344 s against 300 s true (docs/VALIDATION.md).
* In Sep 2026, regressing trigger rate on acid concentration gave a confidently **wrong-signed**
  reaction order ($n = -0.13 \pm 0.03$, $r^2 = 0.90$), because energy per detected event rose ×8 between
  0.1 and 0.75 M while the count saturated. **Do not fit kinetics to a trigger rate.**
* Always normalise by **live time** (masked time removed). `event_rate` does this. Forgetting it once
  made every rate 18 % low.
* Do not quote a rate ratio built on a handful of events: require at least ~100 triggers. With zero
  counts before the acid, the ratio is a lower bound, using $3/T_\mathrm{live}$ as the 95 % upper limit
  on the pre-acid rate.
* **Look at the detector** (`plot_detector`) in a busy and in a quiet stretch before trusting a count.
  Two detector bugs of the old pipeline (every trigger 59.5 ms late; `off` threshold ignored, so long
  events fired repeatedly) were invisible in every summary number and obvious in one plot. Both now
  have tests.

## 8. Comparing with pH

* Use **excess** power (baseline subtracted in linear power), not the raw level. Near the noise floor
  the raw level flattens trends (demo: −8.9 raw vs −10.5 ± 0.6 dB/pH excess, −10 true).
* Two quantities that both change smoothly with time will correlate whatever the mechanism. A
  correlation is only interesting if it survives (a) a background take (no reaction, same handling),
  (b) a different molarity or sample, and (c) a physically reasonable lag.
* The regression standard error assumes independent points, and a time series does not have
  independent points. The effective sample size is closer to (duration / correlation time) than to n.
* pH electrodes respond in seconds to tens of seconds and see the bulk, not the crystal surface. A lag
  of order the mixing time is expected.
* The Sep-2026 gravimetry showed that a *fixed pipetted volume fully consumed* gives mass loss ∝
  concentration for **any** rate law, so a clean slope there measured titration, not kinetics. The same
  trap applies to pH: record the liquid volume, or the chemistry cannot be inverted.

### Reading the pH file

* The loader refuses to guess day/month order. `12/09/2026` could be 12 Sep or 9 Dec, and per-row
  guessing once reordered a log silently. If it stops with "date order is ambiguous", set
  `ph: dayfirst: true` (or `false`) in the config.
* Time zones in timestamps are dropped and the **wall-clock** time is kept, to match the recorder.
* A column with times of day but no dates needs `ph: base_date:`. Otherwise every sample would land on
  today's date and silently fail to overlap the recordings.
* Always plot the loaded pH against time once, and check the first and last timestamps.

## 9. Long recordings (hours to days)

* Run `00` first. It prints the cache size per take. With defaults (0.25 s frames, full frequency
  range) the spectrogram of a 192 kHz recording takes about 0.24 GB per channel-hour (≈0.95 GB/h
  for 4 channels). For day-long runs set `dsp.psd_dt: 5` and `dsp.psd_fmax_hz: 25000`: ≈3 MB per
  channel-hour.
* The fine detector envelope (0.5 ms) costs 29 MB per channel-hour per detection band. It is kept
  only while it fits in `max_envelope_gb` (2 GB ≈ 17 h of 4 channels). Beyond that, events are
  still detected during processing, but re-detection with new thresholds needs re-processing.
* Processing speed on the lab machine: 385 s of 4-channel 192 kHz audio in 40 s (~10× real time),
  0.65 GB peak memory. Process one take at a time: several multi-GB takes in parallel exhausted
  the machine once.
* Across many takes, `timeline.level_timeline` leaves gaps between takes as gaps. Never interpolate
  across them.

## 10. What to write in the lab book (each item cost a result in Sep 2026)

| record | why |
|---|---|
| liquid volume and depth, per take | without it, "fast reaction" and "acid used up" cannot be told apart |
| pH at start and end (or the logger) | measures the consumed fraction directly |
| temperature | enters gas volume, sound speed, reaction rate |
| sample mass before/after, same balance zero | the only absolute rate measurement; the handling floor was 19.6 mg |
| exposed surface area | factor ~10 on any surface-normalised rate |
| sensor positions, which track is which, photo | track assignments changed between sessions |
| recorder gain, sample rate | any gain change breaks between-take comparisons |
| clap/tap at a written wall-clock time | clock alignment (§3) |
| every action with a time: sample in, acid in, stir, move a sensor | markers and masks |
| two background takes, and at least one replicate per condition | the error bar; the only replicate pair in Sep 2026 was contaminated |

## 11. Checklist before you show a figure

- [ ] Channel names and sensor types correct for this take?
- [ ] Masks cover every active source and known disturbance (grey bands where expected)?
- [ ] Levels shown as change relative to a stated baseline window, or clearly labelled uncalibrated?
- [ ] No absolute comparison between different sensors?
- [ ] Rates called "trigger rate", live-time normalised, with counts?
- [ ] Looked at the raw waveform / detector at the interesting times?
- [ ] Same effect absent in the background take?
- [ ] Clock offsets checked against a known event?
- [ ] Axis labels with units, the take name and the windows used, all in the figure itself?

## 12. Event catalogue and clustering: what it can and cannot tell you

* **Clustering in time** (CV, Fano, bursts): a CV well above 1 means bunching. But one bubble firing
  the detector twice also raises it, so look at the fraction of gaps below 2× the dead time. A burst
  count is only meaningful if it has a plateau over a range of gap values. In Sep 2026 it fell from 38 to
  5 as the gap went from 1 s to 4 s, i.e. there was no natural burst scale.
* **Coincidence** between channels: only the excess over the time-shifted chance level counts. A
  consistent sub-millisecond lag between sensors a few cm apart is mostly sensor/electronics group delay,
  not travel time (Sep 2026: the contact sensor led by 0.26–0.31 ms, 15× too large for propagation).
* **Waveform clusters:** similar-frequency ringdowns correlate strongly whatever their source. On the demo,
  bubbles form 20–60 tight clusters, and a genuinely repeating rattle stands out only on the channel
  where it is loud (ρ 0.94 against ≤ 0.87). Treat clusters as **candidates**, check `f_peak_hz` against
  known apparatus lines, and look at the gallery.
* **Feature clusters** (Ward on log-features) are descriptive. Change k and the feature set, and trust only
  the groups that survive.

## 13. Glides

* A tracker **always** returns a path, even through noise, and a noise path drifts. On the demo air mic
  (no ridge) it "glides" +0.30 octaves. Read `significant` (median contrast > 6 dB) before anything else.
* Re-examined with this criterion, the Sep-2026 take 016 shows a significant ridge only on Ch1
  (6.3 → 1.0 kHz). The Ch2/Ch3 "glides" of the earlier analysis are at the noise-path contrast. Redo any
  glide statistics with the contrast criterion, and state the threshold.
* **Seed window:** if it contains another feature (a rattle, an apparatus line), the track starts there.
  Check `t_lock_s` and choose a seed window that excludes the other feature.
* **Harmonics:** run `harmonic_ladder`. A strong 0.5× means you tracked a harmonic, so re-seed lower.
* **The air mic is the control.** A glide seen on the air mic too is not liquid-borne.
* **Mechanism unknown.** A Minnaert reading of the Sep-2026 ridge ends at R = 3.2 mm, Bond number 1.3,
  which cannot be a free bubble. The code prints the Bond number next to every radius for this reason.
  Do not quote the radius.

## 14. Seeing the data: full range, zooms, difference images

* Start every take with `views.plot_overview_zoom(f, ch, take=take)`: the whole take plus automatic zooms
  on the baseline, the onset, the loudest 30 s, a significant glide and the strongest single event (the
  last drawn from the **raw audio**, because a 0.25 s frame is longer than the event).
* Add `background=(t0, t1)` for a **difference image**: what changed relative to a quiet part of the same
  take, or of a separate background take (`background_feat=`). It is the fastest way to see what the
  reaction added.
* Fix the colour range with `clim=(lo, hi)` when comparing takes. Otherwise each figure scales itself and
  equal colours do not mean equal levels.
* Absolute pressure needs a calibration: put `pa_per_fs` for a channel in the config (from the
  hydrophone sensitivity and the recorder gain) and use `units="pa"`. Without it, levels are dB re FS.

## 15. Correlating sound with pH

* Correlate **excess** levels (background subtracted), in bands you choose (`correlate.band_timeline`),
  or trigger rates (`correlate.event_count_timeline`, optionally only loud events), against pH or
  **dpH/dt** (`correlate.ph_rate`).
* Read **n_eff**, not n. One take with one decaying reaction gives n_eff ≈ 2 in the demo: every band
  correlates at |r| ≈ 0.9, and nothing is significant. A relation between sound and pH is established by
  **several takes under different conditions** (molarity, sample, temperature) that fall on one line, or by
  `differences=True` (changes vs changes). A high r from a single run does not establish it.
* `correlate.band_scan` shows which bands track pH. With 16 bands, one "significant" band is expected by
  chance, so look for a coherent range of bands.
* Best lags that jump from band to band are noise. Trust a lag only if it is stable across bands and
  takes, and physically reasonable (≥ 0, seconds to tens of seconds: probe response and mixing).
