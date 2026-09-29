# THEORY: what the code computes, exactly

Every quantity the toolkit produces is defined here, with units, assumptions, and the function that
implements it. If the code and this file disagree, one of them is a bug.

Notation: $x_c[n]$ is the recorded sample value of channel $c$ at sample $n$, sampling rate $f_s$
[Hz], $\Delta t = 1/f_s$. Samples are in units of **digital full scale (FS)**. They are not pascals
or volts: no sensor in this setup is calibrated.

---

## 1. Levels (`features.process_take`, `dsp.bandpass_sos`, `dsp.db`)

**Band-pass.** For each band $b = [f_\ell, f_h)$, a 4th-order Butterworth band-pass $h_b$ in
second-order sections (SOS). It becomes a low-pass if $f_\ell \le 0$ and a high-pass if
$f_h \ge 0.95 f_N$ ($f_N = f_s/2$). Bands with $f_\ell \ge 0.95 f_N$ are dropped (`dsp.usable_bands`).

$$y_{c,b}[n] = (h_b * x_c)[n]$$

The filter state is **carried across blocks**, so the blocked result equals filtering the whole record
at once (test: `test_streaming_filter_equals_whole_signal`, max difference < 1e-12). SOS is required
because at $f_\ell = 20$ Hz, $f_N = 96$ kHz the normalised edge is $2\times10^{-4}$, where the $(b, a)$
polynomial form is numerically singular.

**Band level (mean square)** over hops of $L = \mathrm{round}(\texttt{level\_dt}\cdot f_s)$ samples:

$$\overline{y^2}_{c,b}[k] = \frac{1}{L}\sum_{n=kL}^{(k+1)L-1} y_{c,b}[n]^2 \qquad [\mathrm{FS}^2]$$

stored at time $t_k = (k + \tfrac12)\,\texttt{level\_dt}$ (bin centre). The extra band `broadband` uses
$y = x$ (no filter). **Decibels:** $L_\mathrm{dB} = 10\log_{10}\overline{y^2}$ [dB re 1 FS²]. A
full-scale sine has $\overline{y^2} = 1/2$, i.e. −3.01 dB.

**Averaging in time is always done in linear power**, never in dB (`Features.level_series`,
`ph.acoustic_at`). The mean of dB values is the log of the geometric mean. That differs from the
energy mean and is biased low by an amount that depends on the variance (test:
`test_acoustic_at_linear_power_mean`: 0 dB and 10 dB average to 7.40 dB, not 5 dB).

**Excess power.** The reaction adds a source on top of an existing noise floor. Power adds, dB does
not, so the contribution of the new source is

$$P_\mathrm{exc}(t) = \overline{y^2}(t) - \overline{y^2}_\mathrm{base}, \qquad \overline{y^2}_\mathrm{base} = \text{mean over a quiet reference window}$$

and any comparison with chemistry (pH, rate) should use $10\log_{10} P_\mathrm{exc}$. Near the floor
the raw level flattens every trend: in the demo, level-vs-pH slope is −8.9 dB/pH raw against −10.5
± 0.6 excess, with −10 put in (docs/VALIDATION.md R3).

**Change relative to baseline** (plots, `relative_to=`): $\Delta L = L_\mathrm{dB}(t) -
\mathrm{median}_{t\in\mathrm{base}} L_\mathrm{dB}$. Every uncalibrated factor of the channel
(sensitivity, gain, coupling) cancels in $\Delta L$, which is why it is the quantity to compare between
channels and between takes.

## 2. Spectra (`features.process_take`, `Features.psd_mean`, `dsp.band_power`)

Per frame of $P = \mathrm{round}(\texttt{psd\_dt}\cdot f_s)$ samples, a Welch estimate: Hann window,
segment length $N$ (`psd_nperseg`, default the power of two nearest $f_s/23.4$ Hz, i.e. 8192 at
192 kHz and 2048 at 48 kHz), 50 % overlap, no detrending, one-sided density scaling:

$$\hat S_c(t_j, f_m) \quad [\mathrm{FS}^2/\mathrm{Hz}], \qquad f_m = m\,f_s/N,\ \Delta f = f_s/N .$$

The scaling is such that $\sum_m \hat S(f_m)\,\Delta f \approx \overline{x^2}$ (discrete Parseval).
Hence the **band power is a rectangle-rule sum**

$$\overline{y^2}_{[f_\ell, f_h)} \approx \Delta f \sum_{f_\ell \le f_m < f_h} \hat S(f_m).$$

The trapezoid rule halves the two end bins and under-integrates by about $1/N_\mathrm{bins}$ of the
band: 0.2 dB for a 21-bin band. Test `test_band_power_parseval_narrow_band` shows the rectangle rule
within 0.05 dB of the analytic white-noise value and the trapezoid rule failing by more than 0.15 dB.
Frequency bins are **not** decimated on storage, because a narrow apparatus line (the Sep-2026 5.6 kHz
line had $Q$ up to 213, width ≈ 26 Hz) is only distinguishable from a bubble ($Q\approx 28$ at that
frequency) at full resolution.

`Features.psd_mean(ch, t0, t1)` is the arithmetic mean of $\hat S$ over unmasked frames in
$[t_0, t_1)$. `dsp.line_q` reports the peak frequency and half-power $Q = f_\mathrm{pk}/\Delta
f_{-3\mathrm{dB}}$ and flags `q_resolved = False` when the width is under $3\Delta f$ (then $Q$ is set by
$N$, not by the line).

## 3. Masks (`masks.py`)

A mask is a set of windows $W = \{[a_i, b_i)\}$. Masked level samples are NaN, masked PSD frames are
excluded from means, and masked envelope samples are NaN for the detector. The **live time** of an
interval is $T_\mathrm{live} = (t_1 - t_0) - |W \cap [t_0,t_1)|$ (`masks.live_seconds`).

`masks.find_periodic_bursts` finds a periodic loud source (e.g. 1 s chirps every 10 s). It smooths the
level with a 50 ms boxcar and, for a descending ladder of thresholds $\theta \cdot \mathrm{median}$,
$\theta \in \{32, 20, 12, 8, 5, 3\}$, takes the start times of excursions lasting 0.3 to 3 s. It then
fits $t_k = t_0 + kT$ (period $T$ fixed if given, otherwise median gap and then least squares) and
keeps the threshold with the most inliers within ±50 ms. (The 50 ms smoothing is a no-op when `level_dt` ≥ 50 ms.) The start time is quantised to `level_dt` and
refers to when the burst becomes loud **in the band used**. For an exponential sweep starting at
$f_1$, it enters a band starting at $f_\ell$ after $\tau = T_\mathrm{sw}\ln(f_\ell/f_1)/\ln(f_2/f_1)$
(0.47 s for 0.1→20 kHz into 1.2 kHz). Use the broadband level, and pad generously before
(`pad_before_s` ≥ 0.5 s).

## 4. Event detector (`dsp.StaLta`, `dsp.detect`)

**Envelope.** $e_c[k]$ is the band mean square over hops of $E = \mathrm{round}(\texttt{env\_dt}\cdot
f_s)$ samples (default 0.5 ms), exactly as in §1 but on the finer grid.

**Ratio.** With $n_a = \texttt{sta\_s}/\texttt{env\_dt}$ and $n_l = \texttt{lta\_s}/\texttt{env\_dt}$
(defaults 1 and 120):

$$\mathrm{STA}_k = \frac{1}{|V_a|}\sum_{i\in V_a} e[i],\quad V_a = \{i \in [k-n_a+1, k] : e[i]\ \mathrm{finite}\}$$
$$\mathrm{LTA}_k = \frac{1}{|V_l|}\sum_{i\in V_l} e[i],\quad V_l = \{i \in [k-n_l+1, k] : e[i]\ \mathrm{finite}\}$$
$$r_k = \mathrm{STA}_k/\mathrm{LTA}_k,\qquad r_k = \mathrm{NaN}\ \text{if}\ |V_a| \le n_a/2\ \text{or}\ |V_l| \le n_l/2\ \text{or}\ e[k]\ \text{masked}.$$

**Both windows end at sample $k$.** The Sep-2026 code once aligned the windows by their start instead,
which compared $e[k]$ with the 60 ms **after** it and delayed every trigger by $(n_l-1)\,\texttt{env\_dt}$
= 59.5 ms (`test_impulse_timing_exact` checks the exact time and rejects the misaligned version).

**Trigger with hysteresis.** State `armed` (initially true). At sample $k$ with finite $r_k$:
if armed and $r_k > \texttt{on}$ and $k - k_\mathrm{last} \ge n_\mathrm{dead}$, then fire (record
$t = (k+\tfrac12)\,\texttt{env\_dt}$ and $r_k$), set $k_\mathrm{last} = k$, and disarm; if disarmed and
$r_k < \texttt{off}$, then re-arm. One excursion above `on` gives one trigger however long it lasts
(`test_one_trigger_per_sustained_excursion`). The Sep-2026 `analysis4` code re-armed immediately
after firing, so `off` did nothing and a long excursion fired once per dead time (1.5 ms). On real take
260910_011 that inflated audio-band trigger counts by ×1.09 to ×1.27 per channel (VALIDATION V2).

**Streaming.** The detector carries the last $n_l - 1$ envelope samples, its armed state and
$k_\mathrm{last}$ across blocks, so any block partition gives exactly the one-shot result
(`test_block_invariance`, 200 random block sizes). The loop runs over threshold crossings, not
samples (`searchsorted` on the index sets $\{r>\texttt{on}\}$ and $\{r<\texttt{off}\}$), so its cost
scales with the number of triggers.

**Trigger rate** (`Features.event_rate`): in bins of width $B$,

$$\hat\lambda_j = \frac{N_j}{T_{\mathrm{live},j}},$$

with $T_\mathrm{live}$ excluding the time in which the ratio is NaN
(`masks.detector_dead_windows`): the first $\texttt{lta\_s}/2$ of the take (the LTA needs more
than half its window valid), every mask, and $\texttt{lta\_s}/2$ after the end of each mask
at least that long (the LTA re-fills). It is NaN if $T_\mathrm{live} < 0.2B$. The summed live time
equals the detector's own count of valid ratio samples to within a few envelope samples (test
`test_event_rate_live_time_matches_detector`). With $N$ counts, the Poisson 95 % interval is roughly $N \pm 2\sqrt{N}$ for
$N\gtrsim 20$; for $N = 0$ the 95 % upper limit on the rate is $3/T_\mathrm{live}$.

**What a trigger is.** It is a transient that stands out from the preceding 60 ms of the same band, and
it is **not** a bubble. At high event rates, (a) pulses overlap, and (b) the LTA is inflated by
preceding pulses, so a weaker pulse within ~60 ms of a strong one cannot reach `on`. Both reduce the
count, and the detected rate saturates. On the demo (known truth) recall is 97 % at 6–9 s⁻¹ and 82 % at
25–40 s⁻¹. Because of that, the decay time fitted to the trigger rate is 344 s against 300 s input
(VALIDATION R1/R2).

## 5. Onset (`dsp.find_onset`)

Given a level series $y(t)$ on step $\delta$, baseline $\bar y_0 = \mathrm{median}\{y(t) : t \in
[t_a, t_b)\}$, and $A(t) = [10\log_{10}(y/\bar y_0) > \texttt{rise\_db}]$. The onset is the first
$t \ge t_b$ with $A(t)$ true such that, over the next $H = \texttt{hold\_s}/\delta$ samples,
more than half are unmasked and more than a fraction `frac` of the unmasked ones satisfy $A$. Counts are
integers computed by cumulative sums (exact). The hold rejects single impulses: a sample drop or a
knock fires a bare threshold but cannot hold for 10 s (`test_onset_rejects_single_impulse`, including
the control with no hold, which picks the impulse).

## 6. Time axis and clocks (`io.read_header`, `Features.abs_time`, `ph.load_ph`)

The start time of a take is the BWF `bext` OriginationDate/OriginationTime (recorder clock, 1 s
resolution). Byte offsets: 320–329 date, 330–337 time, 338–345 TimeReference (samples since midnight,
the timecode, which is **not** used because the F6 timecode need not follow the clock). Lab time of
sample $n$:

$$t_\mathrm{lab}(n) = t_\mathrm{bext} + \texttt{recorder\_clock\_offset\_s} + n/f_s .$$

Split parts are treated as contiguous. The loader warns if a part's bext start differs from the running
end time by more than 1.5 s. On Dataset1, part 2 of take 008 starts 559 s after part 1 and part 1 lasts
558.958 s, consistent to the 1 s resolution. pH lab time is logger time + `ph.clock_offset_s`. Clock
drift within a take is not modelled (quartz: ~1e-5, i.e. 36 ms per hour).

## 7. Alignment with pH (`ph.acoustic_at`, `ph.align`, `ph.lagged_correlation`)

For each pH sample at $t_i$, the acoustic value is the linear-power mean of the binned series over
$[t_i - w/2, t_i + w/2)$, converted to dB. It is NaN if the window contains no acoustic data (between
takes). The regression slope in `plots.plot_ph_scatter` is ordinary least squares, and its quoted
standard error assumes independent residuals. Consecutive samples of a smooth time series are not
independent, so the true uncertainty is larger. Treat the slope as descriptive.

Lag correlation: both series are averaged onto a common grid of step $\delta$, and for lag
$\ell = m\delta$, $r(\ell) = \mathrm{corr}(x(t), y(t+\ell))$ (Spearman by default) over times where
both exist. $\ell > 0$ means $y$ lags $x$ (test: a 60 s shift is recovered exactly). Two smooth
monotone series correlate strongly at every lag, so a peak in $r(\ell)$ is a hint, not a measurement.

## 8. Bubble resonance (`physics.py`)

Minnaert frequency with surface tension (Leighton, *The Acoustic Bubble*, 1994):

$$f_0(R) = \frac{1}{2\pi R\sqrt{\rho}}\sqrt{3\kappa\left(P_0 + \frac{2\sigma}{R}\right) - \frac{2\sigma}{R}}$$

with $\kappa = 1.304$ (CO₂, adiabatic), $P_0 = 101325$ Pa, $\rho = 998$ kg m⁻³, $\sigma = 0.0728$ N m⁻¹
(water, 20 °C). Large-$R$ limit: $f_0 R \to \sqrt{3\kappa P_0/\rho}/2\pi = 3.17$ m Hz. `minnaert_R`
inverts by bisection in $\log R$ (round trip < 1e-12). Tests: large-$R$ limit to 1e-4, and the control
without the factor 3 is off by more than 40 %.

**Assumptions that break in practice:** a free, spherical, isolated bubble ringing at its own
resonance, far from walls. Bubbles detaching from a crystal surface, in a small dish, or in a cloud
violate all four. In the Sep-2026 data the measured ringdown $Q$ (~6) was 5× lower than this model
predicts (~31), so per-event peak frequency was not a radius. See STUDENT_GUIDE §6.

Quarter-wave layer mode (`quarter_wave_mode`): $f = c/4h$ for a liquid layer of depth $h$ over a rigid
bottom. It is only an estimate: measure the real mode with a water-only take.

## 9. The synthetic demo (`demo.py`)

Bubble onsets $T_i$ are an inhomogeneous Poisson process with $\lambda(T) = \lambda_0
e^{-(T-T_a)/\tau}$ for $T > T_a$, generated by thinning ($\lambda_0 = 40$ s⁻¹, $\tau = 300$ s).
Radii are log-normal (median 0.7 mm, σ_ln 0.35). Each bubble is $a\,e^{-t/\tau_b}\sin 2\pi f_0 t$ with
$\tau_b = Q/(\pi f_0)$, $Q = 8$. Its energy $\propto a^2\tau_b/4$ does not depend on $T$, so the
expected excess power is $\propto \lambda(T)$. The acid concentration decays with the same $\tau$,
$[\mathrm{H}^+] = C_0 e^{-(T-T_a)/\tau}$, so $\mathrm{pH} = -\log_{10}C_0 + (T-T_a)/(\tau\ln 10)$ and

$$10\log_{10}P_\mathrm{exc} = \mathrm{const} + 10\log_{10}[\mathrm{H}^+] = \mathrm{const} - 10\,\mathrm{pH},$$

i.e. a slope of exactly −10 dB per pH unit by construction for the bubble power alone. Noise: white at
−80 dB re FS² per channel, a 5610 Hz tone of amplitude 3e-4 (the "rig" line), a broadband click at the
sample drop, and 1 s exponential chirps (0.1 to 20 kHz) every 10 s in two takes.

**Glide:** $f(T) = f_h (f_\ell/f_h)^{u}$, $u = \mathrm{clip}((T - T_a)/T_g, 0, 1)$, with
$f_h = 6$ kHz, $f_\ell = 1.5$ kHz, $T_g = 600$ s (−2 octaves). The phase is integrated analytically,
$\phi = 2\pi f_h T_g (e^{u \ln(f_\ell/f_h)} - 1)/\ln(f_\ell/f_h)$, so it does not depend on block
boundaries. Amplitude 3e-4 plus a 2nd harmonic at 1.5e-4, on Ch1–3 with gains 1, 0.7, 0.3, and **0 on
the air mic**. This tone adds audio-band power not proportional to λ, so the level-vs-pH slope of the
demo is not exactly −10 (VALIDATION R3). **Rattle:** a fixed two-tone transient
$a e^{-t/0.6\,\mathrm{ms}}[\sin 2\pi 9000 t + 0.6 \sin(2\pi 13100 t + 1)]$, $a = 0.03$, every
2.4–3.0 s in DEMO_002 from 50 s, gains 1, 0.6, 1.5, 0.

## 10. Glide tracking (`glide.py`)

**Excess spectrogram.** $D(t_j, f_m) = \hat S(t_j, f_m) / \bar S_0(f_m)$, where $\bar S_0$ is the mean
unmasked PSD over a baseline window (optionally from another take with the same sensors).

**Tracker.** Seed $f^{(0)} = \arg\max_{f \in [f_a, f_b]} \langle D(t, f)\rangle_{t_0 \le t < t_0 + 8\,\mathrm{s}}$.
For each later unmasked frame $j$, search only $|f - \tilde f_{j-1}| < \epsilon \tilde f_{j-1} + \delta$
($\epsilon = 0.12$, $\delta = 120$ Hz), take the peak $f^\ast_j$, and smooth,
$\tilde f_j = s \tilde f_{j-1} + (1 - s) f^\ast_j$ with $s = 0.82$. **Lag:** for a ridge
$\ln f = a - r t$, the smoother's steady-state lag is $\tau = \Delta t\, s/(1-s)$ (1.14 s at
$\Delta t = 0.25$ s), so $\tilde f$ is high by a factor $e^{r\tau} \approx 1 + r\tau$; the octave
difference between two times is unaffected. Tested: < 1 % against the lagged truth, > 1 % against the
unlagged one.

**Contrast and significance.** For each frame,
$C_j = 10\log_{10}\max_{\mathrm{win}} D - 10\log_{10}\mathrm{median}_{\mathrm{win}} D$. For
Welch estimates with ~10 averages, pure noise gives $C \approx 2$–4 dB. A track is **significant** if
$\mathrm{median}_j C_j > 6$ dB. **Locked** frames have a rolling (10 s) median contrast above 6 dB;
$f_\mathrm{start}$ and $f_\mathrm{end}$ are medians over the first and last 3 s of the locked frames, the
glide is $\log_2(f_\mathrm{end}/f_\mathrm{start})$ octaves, and the rate divides by the difference of
the median times of those two windows.

**Harmonic ladder.** For multiple $m$ and frame $j$:
$H_m = \max_{|f/(m\tilde f_j) - 1| < 0.04} D_{\mathrm{dB}} - \mathrm{median}_{0.10 < |f/(m\tilde f_j)-1| < 0.20} D_{\mathrm{dB}}$,
reported as its median over frames for $m \in \{0.5, 1, 1.5, 2, 2.5, 3, 4\}$. The non-integer $m > 1$
values are the noise controls. $H_2 \gg$ controls means a harmonic series. $H_{0.5} \gg$ controls means the
tracked ridge is itself a harmonic.

**Radius feasibility.** $R = $ `minnaert_R`$(f)$ is reported only together with the Bond number
$\mathrm{Bo} = \rho g R^2/\sigma = (R/\ell_c)^2$, $\ell_c = \sqrt{\sigma/\rho g} = 2.73$ mm. For
$\mathrm{Bo} \ge 1$ buoyancy dominates surface tension and a free spherical bubble of that size cannot
exist, so the Minnaert reading is not self-consistent.

## 11. Event catalogue and clustering (`catalogue.py`)

**Inter-event statistics.** Gaps $g_i = t_{i+1} - t_i$, keeping only gaps that cross no masked time.
$\mathrm{CV} = \mathrm{sd}(g)/\mathrm{mean}(g)$, which is 1 for a Poisson process. KS test of $g$ against
an exponential with the same mean. Fano factor $F_w = \mathrm{var}(N_w)/\mathrm{mean}(N_w)$ of counts in
fully unmasked windows of length $w$, which is 1 for Poisson. Burst count at gap $G$ is
$1 + \#\{g_i > G\}$, reported for several $G$: a plateau indicates a natural burst scale.

**Coincidence.** For each event $a$ on channel A, $d_a = \min_b (t_b - t_a)$ (signed, nearest). Observed
fraction $p = \Pr(|d_a| \le w)$; chance $p_0$ is the same quantity after shifting B by several seconds
(−7.3 … +7.7 s); excess $= (p - p_0)/(1 - p_0)$.

**Waveforms.** Snippets from the raw WAV, $[t - 2\,\mathrm{ms}, t + 10\,\mathrm{ms}]$, de-meaned. Features:
Hilbert envelope $e$; rise = time from the last sample below $\max(0.1 e_\mathrm{pk}, 2 e_\mathrm{base})$
to the peak; decay $\tau$ from a log-linear fit of $e$ after the peak down to 5 % of the peak;
$Q = \pi f_\mathrm{pk} \tau$ (exact for $e^{-t/\tau}\sin 2\pi f t$, where $Q = \pi f \tau$);
Hann-windowed spectrum peak and centroid; bandwidth = spectral standard deviation; `peak2_rel_db` = the
strongest peak more than 20 % away from $f_\mathrm{pk}$, relative to it.

**Correlation clustering.** $\rho_{ij} = \max_\ell |\sum_n \hat w_i[n]\hat w_j[n + \ell]|$ with
unit-norm $\hat w$ (FFT). Average linkage on $1 - \rho$, cut at $1 - \rho_\mathrm{min}$. For two ringdowns
of equal $Q$ and frequencies $f, f(1+\epsilon)$, $\rho$ stays high while $\epsilon \lesssim 1/Q$, so a
population of single-mode ringdowns forms tight clusters by frequency. That is why cluster size alone does
not identify a repeater (VALIDATION C1). **Feature clustering:** Ward linkage on z-scored
$\log_{10}$ features, $k$ clusters, and PCA scores by SVD for display.

## 12. Glide interpretation (`glide.interpret`, `physics.py`)

Derivations and the mechanism table are in [docs/GLIDE_INTERPRETATION.md](docs/GLIDE_INTERPRETATION.md).
Implemented relations:

* Single bubble: $R(t) = $ `minnaert_R`$(\tilde f(t))$ on the 10 s running-median track;
  $\dot R = \mathrm{d}R/\mathrm{d}t$ (central differences); Bond $= (R/\ell_c)^2$.
* Bubble touching a rigid wall: $f_\mathrm{wall} = f_\mathrm{free}(R)\,(1 + R/2h)^{-1/2}$, with
  $h = R$ giving $\sqrt{2/3} = 0.816$ (`wall_factor`). The observed $f$ then needs $R$ with
  $f_\mathrm{free}(R) = f/0.816$.
* Bubbly layer: Wood's law $1/(\rho_m c_m^2) = (1-\beta)/(\rho_w c_w^2) + \beta/(\kappa P_0)$,
  $\rho_m = (1-\beta)\rho_w$; quarter-wave $f = c_m/4h$; $\beta(f, h)$ by bisection in
  $\log\beta$, NaN if $\beta > 2\,\%$ would be needed or $f > c_w/4h$ (`beta_from_layer_mode`).
  Gas-dominated check: $c_m^2 \to \kappa P_0 / (\rho_w \beta(1-\beta))$ (tested to 2 % at
  $\beta = 10^{-2}$).
* Fritz departure: $\tfrac43\pi R^3 \rho g = \pi d \sigma$ (tested exactly).

## 13. Acoustics vs pH (`correlate.py`)

* **Narrow band from the stored PSD:** $\overline{y^2}_{[f_\ell, f_h)}(t_j) = \Delta f
  \sum_{f_\ell \le f_m < f_h} \hat S(t_j, f_m)$, masked frames NaN, then binned in linear power.
  Any band can be chosen after processing. Its resolution is the PSD bin width, so a band narrower
  than ~5 bins is resolution-limited.
* **pH rate:** a least-squares slope of pH against time within $\pm$`smooth_s`/2 of each sample,
  $\times 60$ → pH per minute (tested exactly on a linear ramp).
* **Correlation:** both series averaged onto a common grid; for lag $\ell$, Pearson $r$ and Spearman
  $\rho$ of $y(t+\ell)$ vs $10\log_{10}x(t)$; best $|r|$ reported. **Effective sample size**
  $n_\mathrm{eff} = n(1 - r_1 r_2)/(1 + r_1 r_2)$ from the lag-1 autocorrelations $r_1, r_2$
  (Bretherton et al. 1999); $t = r\sqrt{(n_\mathrm{eff}-2)/(1-r^2)}$, two-sided p with
  $n_\mathrm{eff}-2$ degrees of freedom. `differences=True` correlates first differences, which
  removes shared trends. The test `test_spurious_correlation_of_two_trends_is_not_significant` shows two
  independent random walks: the naive p is ~10⁻⁵⁰, while the $n_\mathrm{eff}$ p is over 100× larger.
* **Band scan:** the above for every third-octave band (or any list). With ~16 bands, about 1 passes
  p < 0.05 by chance.

## 14. Views (`views.py`, `plots.spectrogram_db`)

* **Difference image:** $D_\mathrm{ratio} = 10\log_{10}(\hat S / \bar S_\mathrm{bg})$ or
  $D_\mathrm{excess} = 10\log_{10}\max(\hat S - \bar S_\mathrm{bg}, 0)$, with $\bar S_\mathrm{bg}$ the
  mean unmasked PSD of a background window (same take or a background take) averaged onto the displayed
  frequency rows exactly.
* **Calibration:** with `pa_per_fs` = $k$ (Pa per unit full scale) for a channel,
  dB re 1 µPa²/Hz $=$ dB re FS²/Hz $+ 10\log_{10}(k^2/10^{-12})$ (tested: $k = 10$ gives +140 dB).
* **Colour/level range:** `clim` fixes the dB limits for the overview and every zoom; otherwise the
  2nd–99.5th percentiles are used (symmetric for difference images).
* **Raw zoom:** STFT of the raw samples, 256-sample Hann segments, 90 % overlap (1.3 ms at 192 kHz).
