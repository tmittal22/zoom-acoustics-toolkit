"""
real_data_crosscheck.py -- regression anchor on REAL Zoom F6 data.

Processes take 260910_011 (Dataset2, 4 ch, 192 kHz, 385 s) with this toolkit and compares
it to the analysis4 L1 cache built from the same file by independent code in Sep 2026
(same band edges, same Welch settings).  Agreement to float32 rounding means the file
reader, block handling, filters and PSD are equivalent to the verified earlier pipeline.

Also runs the analysis4 trigger logic and this toolkit's hysteresis logic on the SAME
envelope, to measure how much the old immediate re-arm inflated trigger counts.

Only runs on the lab machine where the Sep-2026 data are mounted:
    python validation/real_data_crosscheck.py
Writes validation/out/real_crosscheck.json and figures.
"""
import json
import os
import tempfile
import sys
import time
from pathlib import Path

import numpy as np

import zoom_acoustics as za
from zoom_acoustics import dsp

# location of the Sep-2026 experiment folder (lab machine); override with an env variable
EXPT = Path(os.environ.get("ZA_SEP2026_EXPERIMENT",
                          "/media/tmittal/Disk1/BKG_Data_Drive/experiment"))
A4 = EXPT / "analysis4" / "cache" / "011_EXP3_1M_L1"
OUT = Path(__file__).parent / "out"


def a4_style_triggers(tt, r, on=8.0, dead=3):
    """analysis4 trigger_from_ratio, reproduced verbatim in effect: re-arms immediately."""
    trig, last = [], -10 ** 9
    for i in np.flatnonzero(np.isfinite(r) & (r > on)):
        if i - last < dead:
            continue
        trig.append(tt[i]); last = i
    return np.array(trig)


def main():
    if not A4.exists():
        print("Sep-2026 data not mounted; nothing to check")
        return 0
    OUT.mkdir(exist_ok=True)
    cfg = za.config_from_dict(dict(
        data_dir=str(EXPT / "Dataset2_ThusSep10" / "passive"),
        cache_dir=str(Path(tempfile.gettempdir()) / "za_real_cache"), figures_dir=str(OUT),
        file_pattern=r"260910_011\.WAV"))
    take, = za.discover_takes(cfg.data_dir, pattern=cfg.file_pattern)
    t0 = time.time()
    za.process_take(take, cfg)
    elapsed = time.time() - t0
    f = za.load_features(cfg, take.name)

    psd_a4 = np.load(A4 / "psd.npy", mmap_mode="r")
    env_a4 = np.load(A4 / "env.npy", mmap_mode="r")          # [ch, band, n] bands as a4
    res = dict(take=take.name, duration_s=take.duration_s, process_s=round(elapsed, 1))
    # PSD: compare on frames where both exist
    n = min(psd_a4.shape[1], f._psd.shape[1])
    rel = []
    for ci in range(4):
        a = np.asarray(psd_a4[ci, :n], float)
        b = np.asarray(f._psd[ci, :n], float)
        m = a > 0
        rel.append(float(np.max(np.abs(b[m] / a[m] - 1))))
    res["psd_max_rel_diff_per_channel"] = rel
    # audio-band envelope (a4 band index 1) vs this toolkit's stored env
    ne = min(env_a4.shape[-1], f._env.shape[-1])
    erel = []
    for ci in range(4):
        a = np.asarray(env_a4[ci, 1, :ne], float)
        b = np.asarray(f._env[ci, 0, :ne], float)
        m = a > 1e-20
        erel.append(float(np.median(np.abs(b[m] / a[m] - 1))))
        erel.append(float(np.max(np.abs(b[m] / a[m] - 1))))
    res["env_audio_rel_diff_median_max_per_channel"] = erel
    # band levels: this toolkit's 0.25 s level vs a4 env block-averaged to 0.25 s
    k = 500
    lrel = []
    for ci in range(4):
        a = np.asarray(env_a4[ci, 1, :ne // k * k], float).reshape(-1, k).mean(1)
        b = f.level(ci + 1, "audio")[:len(a)]
        lrel.append(float(np.max(np.abs(b / a - 1))))
    res["level_audio_vs_a4_env_max_rel"] = lrel

    # trigger logic comparison on identical envelopes
    trig = {}
    for ci in range(4):
        e = f.env(ci + 1, "audio")
        d = dsp.StaLta(f.env_dt, **f.meta["detector"])
        r = d.ratio(e)
        tt = (np.arange(len(e)) + 0.5) * f.env_dt
        n_new = int(((f.events.channel == ci + 1) & (f.events.band == "audio")).sum())
        n_old = len(a4_style_triggers(tt, r))
        trig[f"Ch{ci+1}"] = dict(hysteresis=n_new, a4_immediate_rearm=n_old,
                                 inflation=round(n_old / max(n_new, 1), 3))
    res["triggers_audio"] = trig
    (OUT / "real_crosscheck.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

    fig = za.plots.plot_spectrogram(f, fmax=24000)
    za.plots.save(fig, OUT, "real_011_spectrogram")
    fig = za.plots.plot_levels(f, bands=("low", "audio", "rig", "hb1"), dt=1.0,
                               relative_to=(3, 25))
    za.plots.save(fig, OUT, "real_011_levels_unmasked")

    # find the chirp train on the air mic (Ch4) and mask it.  The generator log for this
    # take (pulses_*.csv) declares a 10.000 s period, so pass it rather than fit it.
    g = za.masks.find_periodic_bursts(f.t_level, f.level(4, "broadband"), period_s=10.0)
    res["chirp_grid"] = g
    res["chirp_grid_audio_band"] = za.masks.find_periodic_bursts(f.t_level, f.level(4, "audio"),
                                                                 period_s=10.0)
    wins = za.masks.periodic_windows(g["first_s"], g["period_s"], f.duration_s)
    f.add_masks(wins)
    res["masked_fraction"] = 1 - za.masks.live_seconds(0, f.duration_s, f.mask_windows) / f.duration_s
    fig = za.plots.plot_levels(f, bands=("low", "audio", "rig", "hb1"), dt=1.0,
                               relative_to=(3, 25))
    za.plots.save(fig, OUT, "real_011_levels_masked")
    (OUT / "real_crosscheck.json").write_text(json.dumps(res, indent=1))
    print("chirp grid:", g, res["chirp_grid_audio_band"], "masked fraction", res["masked_fraction"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
