"""
real_glide_crosscheck.py -- glide tracker on REAL take 260910_016 against analysis4.

analysis4 (a4_glide.py, Sep 2026) reported on this take: Ch1 6525 -> 910 Hz (-2.84 oct),
Ch2 -0.37, Ch3 -0.58, Ch4 (air) +0.04, tracking from 49.5 s with seed 4-12 kHz, baseline
= pre-acid mean, chirps masked.  Same settings here.  analysis4 took single-frame end
points; this toolkit takes medians over the first/last 3 s of the LOCKED track and reports
the ridge contrast, so small differences in octaves are expected and explained by that.

Only runs where the Sep-2026 data are mounted:  python validation/real_glide_crosscheck.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import zoom_acoustics as za
from zoom_acoustics import glide as G

# location of the Sep-2026 experiment folder (lab machine); override with an env variable
EXPT = Path(os.environ.get("ZA_SEP2026_EXPERIMENT",
                          "/media/tmittal/Disk1/BKG_Data_Drive/experiment"))
A4 = EXPT / "analysis4" / "exp" / "016_EXP3_050M"
OUT = Path(__file__).parent / "out"


def main():
    if not A4.exists():
        print("Sep-2026 data not mounted; nothing to check")
        return 0
    OUT.mkdir(exist_ok=True)
    a4 = json.loads((A4 / "glide.json").read_text())
    pv = json.loads((A4 / "passive.json").read_text())
    ch = pv["chirp"]
    cfg = za.config_from_dict(dict(
        data_dir=str(EXPT / "Dataset2_ThusSep10" / "passive"),
        cache_dir=str(Path(tempfile.gettempdir()) / "za_real_cache"), figures_dir=str(OUT),
        file_pattern=r"260910_016\.WAV",
        takes={"260910_016": {"periodic": dict(period_s=ch["period_s"], first_s=ch["t0_s"],
                                                pad_before_s=0.25, pad_after_s=1.4)}}))
    take, = za.discover_takes(cfg.data_dir, pattern=cfg.file_pattern, verbose=False)
    za.process_take(take, cfg)
    f = za.load_features(cfg, take.name)
    t0 = a4["track_window_s"][0]
    tracks, res = {}, {}
    for c in f.channels:
        tr = G.track_ridge(f, c, (3.0, 25.0), t0, seed=(4000, 12000))
        s = G.summarize(tr)
        tracks[c] = tr
        a = a4["channels"][["Ch1 side-A", "Ch2 side-B", "Ch3 crystal", "Ch4 air"][c - 1]]
        res[f"Ch{c}"] = dict(this_octaves=round(s["octaves"], 3), a4_octaves=round(a["octaves"], 3),
                             this_f_start=round(s["f_start_hz"]), a4_f_start=round(a["f_start_hz"]),
                             this_f_end=round(s["f_end_hz"]), a4_f_end=round(a["f_end_hz"]),
                             contrast_db=round(s["median_contrast_db"], 1),
                             significant=s["significant"], t_lock_s=s["t_lock_s"],
                             frac_locked=round(s["frac_locked"], 2),
                             bond_end=round(s["bond_end"], 2))
    (OUT / "real_glide_crosscheck.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    fig = za.plots.plot_glide(f, tracks, (3.0, 25.0), show_ch=1, fmax=20000)
    za.plots.save(fig, OUT, "real_016_glide")
    return 0


if __name__ == "__main__":
    sys.exit(main())
