"""
demo_recovery.py -- does the pipeline recover what was put into the synthetic demo data?

Checks (numbers go to validation/out/demo_recovery.json, figures to validation/out/):
  R1  event timing: fraction of true bubble onsets matched by a trigger within 1 ms, and
      fraction of triggers with no true bubble (false), on Ch1 of DEMO_002 and DEMO_004
  R2  trigger rate vs the true rate lambda(t) = lambda0 exp(-(t - t_acid)/tau) across
      DEMO_002..004 (includes the 2-channel split take and the 1-channel mono take)
  R3  audio-band level vs pH: slope should be -10 dB per pH unit by construction
  R4  onset: detected acid onset vs the true 40.0 s in DEMO_002
  R5  chirp finder: recovers first_s = 5.1 s, period 10 s on the air mic of DEMO_001
  G1  glide: tracked ridge vs the known 6 -> 1.5 kHz law on every channel of DEMO_002..004,
      the air mic (no ridge) flagged not significant, harmonic ladder sees the 2nd harmonic
  C1  waveform clustering: does the most self-similar cluster recover the repeating rattle?

Run after:  python -m zoom_acoustics demo demo_data && python -m zoom_acoustics process config/demo.yaml
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import zoom_acoustics as za

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / "out"


def match(true_t, trig_t, tol):
    true_t, trig_t = np.sort(true_t), np.sort(trig_t)
    j = np.searchsorted(trig_t, true_t - tol)
    hit = (j < len(trig_t)) & (np.abs(trig_t[np.minimum(j, len(trig_t) - 1)] - true_t) <= tol)
    k = np.searchsorted(true_t, trig_t - tol)
    ok = (k < len(true_t)) & (np.abs(true_t[np.minimum(k, len(true_t) - 1)] - trig_t) <= tol)
    return hit, ~ok


def main():
    cfg = za.load_config(ROOT / "config" / "demo.yaml")
    truth = json.loads((ROOT / "demo_data" / "truth.json").read_text())
    feats = {f.take: f for f in za.load_all_features(cfg)}
    OUT.mkdir(exist_ok=True)
    res = {}

    # ---- R1 timing
    r1 = {}
    for name in ("DEMO_002", "DEMO_004"):
        f = feats[name]
        tt = np.array(truth["takes"][name]["bubble_t_s"])
        ev = f.events
        trig = ev[(ev.channel == 1) & (ev.band == "audio")].t_s.to_numpy()
        trig = trig[~za.masks.mask_array(trig, f.mask_windows)]
        tt_live = tt[~za.masks.mask_array(tt, f.mask_windows)]
        hit, false = match(tt_live, trig, 1e-3)
        # which bubbles are missed? those closely followed/preceded by another one
        gap = np.minimum(np.r_[np.inf, np.diff(tt_live)], np.r_[np.diff(tt_live), np.inf])
        iso = gap > 0.02
        r1[name] = dict(n_true=int(len(tt_live)), n_trig=int(len(trig)),
                        recall=float(hit.mean()), false_frac=float(false.mean()),
                        recall_isolated_20ms=float(hit[iso].mean()),
                        recall_crowded=float(hit[~iso].mean()) if (~iso).any() else None,
                        triggers_per_bubble=float(len(trig) / max(len(tt_live), 1)))
    res["R1_timing"] = r1

    # ---- R2 rate curve across takes (lab clock)
    lam0, tau = truth["lambda0_per_s"], truth["tau_s"]
    t_acid = pd.Timestamp(truth["t_acid_lab"])
    fl = [feats[n] for n in ("DEMO_002", "DEMO_003", "DEMO_004")]
    rate = za.timeline.rate_timeline(fl, 1, "audio", bin_s=10.0).dropna()
    dt_s = (rate.index - t_acid).total_seconds().to_numpy()
    true_rate = np.where(dt_s > 0, lam0 * np.exp(-dt_s / tau), 0.0)
    post = dt_s > 10
    ratio = rate.to_numpy()[post] / true_rate[post]
    res["R2_rate"] = dict(median_detected_over_true=float(np.median(ratio)),
                          p10=float(np.percentile(ratio, 10)), p90=float(np.percentile(ratio, 90)),
                          n_bins=int(post.sum()))
    # fitted decay time from detected rate (log-linear fit), vs true tau
    p = np.polyfit(dt_s[post], np.log(rate.to_numpy()[post]), 1)
    res["R2_rate"]["tau_fit_s"] = float(-1 / p[0])
    res["R2_rate"]["tau_true_s"] = tau

    # ---- R3 level vs pH
    phdf = za.ph.load_ph_from_config(cfg)
    lev = za.timeline.level_timeline(fl, 1, "audio", dt=1.0)
    ph_post = phdf[phdf.index > t_acid + pd.Timedelta(seconds=20)]
    al_raw = za.ph.align(ph_post, lev, window_s=10)
    # EXCESS power: subtract the pre-acid baseline in LINEAR power first.  The raw level
    # includes the noise floor and the rig tone, which add a constant and flatten the slope.
    base = feats["DEMO_002"].band_level_in_window(1, "audio", 5, 34)
    al = za.ph.align(ph_post, (lev - base).clip(lower=1e-30), window_s=10)
    (s_raw, _), (slope, icpt) = np.polyfit(al_raw["pH"], al_raw["acoustic_dB"], 1), \
        np.polyfit(al["pH"], al["acoustic_dB"], 1)
    _, cov = np.polyfit(al["pH"], al["acoustic_dB"], 1, cov=True)
    res["R3_level_vs_pH"] = dict(slope_raw_dB_per_pH=float(s_raw),
                                 slope_excess_dB_per_pH=float(slope),
                                 slope_excess_se=float(np.sqrt(cov[0, 0])),
                                 expected=-10.0, n=int(len(al)))

    # ---- R4 onset
    on, base = feats["DEMO_002"].onset(1, "audio", base_window=(1, 30), hold_s=10)
    res["R4_onset"] = dict(detected_s=on, true_s=truth["takes"]["DEMO_002"]["acid_s"],
                           tick_s=truth["takes"]["DEMO_002"]["tick_s"])

    # ---- R5 chirp grid (DEMO_001 processed with the config's periodic mask; find it
    # blind on the air mic level and compare to the truth)
    f1 = feats["DEMO_001"]
    res["R5_chirp_grid"] = dict(
        broadband=za.masks.find_periodic_bursts(f1.t_level, f1.level(4, "broadband")),
        audio_band=za.masks.find_periodic_bursts(f1.t_level, f1.level(4, "audio")),
        truth=truth["chirp"])

    # ---- G1 glide
    from zoom_acoustics import glide as G, catalogue as K, demo
    ta = truth["glide"]["t_acid_session_s"]
    g1, tracks2 = [], {}
    f2 = feats["DEMO_002"]
    for name in ("DEMO_002", "DEMO_003", "DEMO_004"):
        f = feats[name]
        T0 = truth["takes"][name]["T0_session_s"]
        for c in f.channels:
            tr = G.track_ridge(f, c, (5, 34), 46.0 if name == "DEMO_002" else 1.0,
                               seed=(1000, 12000), base_feat=f2)
            sm = G.summarize(tr)
            tau = f.psd_dt * 0.82 / 0.18
            tt = np.array([sm["t_start_s"] + 1.5, sm["t_end_s"] - 1.5]) - tau + T0
            ft = demo.glide_freq(tt, ta)
            g1.append(dict(take=name, channel=c, octaves=round(sm["octaves"], 3),
                           octaves_true=round(float(np.log2(ft[1] / ft[0])), 3),
                           contrast_db=round(sm["median_contrast_db"], 1),
                           significant=sm["significant"], t_lock_s=sm["t_lock_s"]))
            if name == "DEMO_002":
                tracks2[c] = tr
    res["G1_glide"] = g1
    # seed-window lesson: on Ch3 the wide seed window picks the 9 kHz rattle first
    sm = G.summarize(G.track_ridge(f2, 3, (5, 34), 46.0, seed=(4000, 8000)))
    tau = f2.psd_dt * 0.82 / 0.18
    ft = demo.glide_freq(np.array([sm["t_start_s"] + 1.5, sm["t_end_s"] - 1.5]) - tau + 120.0, ta)
    res["G1_ch3_seed_4_8kHz"] = dict(octaves=round(sm["octaves"], 3),
                                     octaves_true=round(float(np.log2(ft[1] / ft[0])), 3),
                                     t_lock_s=sm["t_lock_s"])
    lad = G.harmonic_ladder(f2, 2, (5, 34), tracks2[2])
    res["G1_harmonic_ladder_ch2"] = lad.round(2).to_dict("records")

    # ---- C1 waveform clustering (DEMO_002, rattle known)
    take2 = za.find_take(za.discover_takes(cfg.data_dir, verbose=False), "DEMO_002")
    rt = np.array(truth["rattle"]["t_s"])
    c1 = {}
    for c in (1, 3):
        ev = f2.events
        ev = ev[(ev.channel == c) & (ev.band == "audio")]
        ev = ev[~za.masks.mask_array(ev.t_s.to_numpy(), f2.mask_windows)]
        W, tw, sr = K.extract_waveforms(take2, ev.t_s, c, max_events=600)
        ftab = K.waveform_features(W, sr)
        C = K.xcorr_matrix(W)
        lab, summ, med = K.cluster_waveforms(C, 0.8, features=ftab, times=tw)
        is_r = np.array([np.min(np.abs(rt - x)) < 2e-3 for x in tw])
        top = summ.iloc[0]
        m = lab == top.cluster
        c1[f"Ch{c}"] = dict(n_events=int(len(tw)), n_rattle_in_sample=int(is_r.sum()),
                            top_cluster_size=int(top["size"]), top_median_rho=round(top.median_rho, 3),
                            top_rattle_members=int((m & is_r).sum()),
                            next_best_rho=round(float(summ.median_rho.iloc[1]), 3),
                            n_candidate_clusters=int(len(summ)),
                            q_est_median_bubbles=round(float(ftab.q_est[~is_r].median()), 2),
                            f_peak_median_bubbles=round(float(ftab.f_peak_hz[~is_r].median()), 1))
        if c == 3:
            top5 = summ.cluster.head(5).tolist()
            ls = np.where(np.isin(lab, top5), lab, 0)
            fig = za.plots.plot_waveform_gallery(W[ls > 0], sr, ls[ls > 0],
                                                 title="C1: DEMO_002 Ch3, five most self-similar clusters "
                                                       "(top row = the rattle)")
            za.plots.save(fig, OUT, "demo_C1_gallery")
    res["C1_clustering"] = c1
    fig = za.plots.plot_glide(f2, tracks2, (5, 34), show_ch=2, fmax=20000, base_feat=f2)
    za.plots.save(fig, OUT, "demo_G1_glide")

    (OUT / "demo_recovery.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))

    # ---- figures
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True, sharex=True)
    tl = pd.date_range(t_acid - pd.Timedelta(minutes=3), rate.index[-1], freq="5s")
    dtl = (tl - t_acid).total_seconds()
    axs[0].plot(rate.index, rate, "o", ms=3, color=za.plots.PALETTE[0], label="detected trigger rate, Ch1")
    axs[0].plot(tl, np.where(dtl > 0, lam0 * np.exp(-dtl / tau), 0), color=za.plots.INK,
                label="true bubble rate λ(t)")
    axs[0].set_ylabel("events / s")
    axs[0].set_yscale("log")
    axs[0].legend()
    axs[0].set_title("R2: detected trigger rate vs input rate, three takes (4 ch, 2 ch split, 1 ch)", loc="left")
    axs[1].plot(lev.index, za.dsp.db(lev), color=za.plots.PALETTE[0], lw=0.8)
    axs[1].set_ylabel("Ch1 audio level [dB re FS²]")
    axs[1].set_title("same takes: audio-band level (gaps between takes are real gaps)", loc="left")
    za.plots._datetime_axis(axs[1])
    za.plots.save(fig, OUT, "demo_R2_rate")
    fig = za.plots.plot_ph_scatter(al, label="Ch1 audio excess")
    fig.axes[0].set_title(f"R3: excess level vs pH, slope {slope:.2f} dB/pH (input -10)", loc="left", fontsize=9)
    za.plots.save(fig, OUT, "demo_R3_level_vs_pH")
    fig = za.plots.plot_detector(feats["DEMO_002"], 1, "audio", 60.0, 60.5)
    ax = fig.axes[0]
    tt = np.array(truth["takes"]["DEMO_002"]["bubble_t_s"])
    for x in tt[(tt >= 60) & (tt < 60.5)]:
        ax.plot(x, 0.97, "v", color=za.plots.INK, ms=5, transform=ax.get_xaxis_transform())
    ax.set_title(ax.get_title() + "   (black triangles = true bubble onsets)", loc="left")
    za.plots.save(fig, OUT, "demo_R1_detector")
    return 0


if __name__ == "__main__":
    sys.exit(main())
