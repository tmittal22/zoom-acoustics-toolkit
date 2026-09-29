"""zoom_acoustics -- passive-acoustic analysis of multi-channel Zoom F-series recordings.

Typical use (see notebooks/):

    import zoom_acoustics as za
    cfg   = za.load_config("config/my_experiment.yaml")
    takes = za.discover_takes(cfg.data_dir)
    za.process_all(takes, cfg)                 # one streaming pass per take, cached
    f     = za.load_features(cfg, takes[0].name)
    za.plots.plot_levels(f)
"""
from . import dsp, masks, physics, plots, ph, timeline, glide, catalogue, report, views, correlate
from .config import load_config, config_from_dict
from .io import discover_takes, find_take, takes_table, read_header
from .features import process_take, process_all, load_features, load_all_features, Features, plan

__version__ = "1.0.0"
