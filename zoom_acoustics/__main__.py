"""Command line:  python -m zoom_acoustics <command> <config.yaml> [take ...]

    list     show every take found in data_dir
    plan     show processing parameters and estimated cache size per take
    process  build feature bundles (skips up-to-date ones; --force rebuilds)
    demo     write the synthetic demo dataset:  python -m zoom_acoustics demo <out_dir>
"""
import sys

import pandas as pd

from . import config, features, io


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2 or argv[0] not in ("list", "plan", "process", "demo"):
        print(__doc__)
        return 2
    cmd, target, rest = argv[0], argv[1], argv[2:]
    if cmd == "demo":
        from .demo import make_demo_dataset
        make_demo_dataset(target)
        return 0
    cfg = config.load_config(target)
    force = "--force" in rest
    names = [a for a in rest if not a.startswith("--")]
    takes = io.discover_takes(cfg.data_dir, recursive=cfg.recursive, pattern=cfg.file_pattern)
    if names:
        takes = [io.find_take(takes, n) for n in names]
    pd.set_option("display.width", 200)
    if cmd == "list":
        print(io.takes_table(takes).to_string(index=False))
    elif cmd == "plan":
        rows = []
        for t in takes:
            p = features.plan(t, cfg)
            rows.append(dict(take=t.name, channels=p["channels"], nperseg=p["nperseg"],
                             df_Hz=round(t.sr / p["nperseg"], 2), store_env=p["store_env"],
                             cache_GB=round(sum(p["size"].values()), 3),
                             notes="; ".join(p["notes"])))
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        features.process_all(takes, cfg, force=force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
