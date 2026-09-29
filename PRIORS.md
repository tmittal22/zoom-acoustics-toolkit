# PRIORS: constraints a result must satisfy before it is believed

Each prior states what must hold, why, and how to check it with this toolkit. A result that
violates one is wrong or needs an explanation written next to it.

| # | prior | why | how to check |
|---|---|---|---|
| P1 | Level changes are **positive in linear excess power** after acid is added, on liquid-coupled channels | a new source adds power; a drop means something moved or changed gain | `plot_levels(relative_to=baseline)`, `plot_psd(excess=True)` |
| P2 | The **background take shows no effect** of the same size | handling, pours and knocks make sound too | run the identical analysis on the water-only take |
| P3 | Onset time is **after** the acid is added (lab book) and **after** the sample-drop impulse | causality; the sample-drop impulse is not a reaction | `Features.onset` vs the take's `events:` note; the onset must not coincide with the drop |
| P4 | Onsets agree between liquid channels to within the level step (0.25 s), and differ in amplitude | same source, different coupling | onset table in notebook 03 |
| P5 | The air mic shows a much smaller relative change than the liquid channels for liquid-borne events | the air-water interface reflects ~99.9 % of incident sound (≈30 dB) | compare Δlevel of air vs hydrophone; a similar Δ means airborne or surface-bursting sound |
| P6 | Trigger rate ≤ (event duration)⁻¹ and the rate curve is not flat-topped | a flat top means saturation, not a constant reaction | `plot_event_rate`; compare with the Δlevel curve, which does not saturate the same way |
| P7 | Rates are live-time normalised and quoted with counts N; ratios need N ≳ 100 | Poisson error √N; masks remove time | `event_rate` returns counts and live seconds |
| P8 | Any correlation with pH has the **physically expected sign**: sound decreasing as pH rises (acid consumed) | the reaction slows as [H⁺] falls | `plot_ph_scatter` slope < 0 |
| P9 | The acoustic-pH lag is ≥ 0 and of the order of the mixing / probe response time (s to tens of s) | the probe sees the bulk later than the surface | `lagged_correlation`; a large negative lag is a clock error |
| P10 | pH at the end is consistent with the acid added and the sample mass lost | stoichiometry: CaCO₃ + 2H⁺ → Ca²⁺ + H₂O + CO₂ (needs the liquid volume) | lab book; mass loss [mg] / 100.09 = mmol CO₂ |
| P11 | No spectral feature used as signal is present in the background take | apparatus resonances (5.6 kHz in the Sep-2026 rig) | `plot_compare_psd` background vs reaction |
| P12 | Nothing is claimed above the anti-alias cliff (~0.47 f_s) or where the channel sits on its electronic floor | filter edge and preamp noise | `plot_psd` of the background take |
| P13 | Per-bubble sizes from Minnaert are **not** reported without an independent check (camera) | falsified on the Sep-2026 data (STUDENT_GUIDE §6) | none in this toolkit, by design |
| P14 | Two takes are compared in absolute dB only if no sensor, gain or geometry changed | uncalibrated sensors | lab book; otherwise use Δ relative to each take's baseline |
| P15 | Results do not depend on nuisance settings: dt, band edges ±10 %, detector `on` 5 to 12 | a real effect is robust | re-run with changed settings (`redetect(on=...)`, `level_series(dt=...)`) |
| P16 | A glide is reported only if its median ridge contrast is clearly above the noise-path value (2–4 dB), and the air mic shows none | a tracker returns a path through noise too | `glide.summarize` → `significant`, air-mic row |
| P17 | A "repeater" cluster is clearly more self-similar than the bubble clusters and not on an apparatus line | similar-frequency ringdowns correlate regardless of source | `cluster_waveforms` table + gallery |
| P18 | Temporal statistics exclude gaps across masks | mask holes are not intervals (they inflated CV 2.58 vs 1.24 in Sep 2026) | `catalogue.interevent_stats(windows=...)` |
| P19 | A sound–pH relation is quoted with n_eff and holds across takes/conditions, not only within one monotonic run | shared trends correlate whatever the mechanism | `correlate.correlate` n_eff, several takes |
| P20 | A glide mechanism is stated only with its required parameter shown physical (Bond < 1, beta <= 2 %) and the discriminating test named | frequency alone cannot identify the mechanism | `glide.interpret`, GLIDE_INTERPRETATION §3 |
