import datetime as dt

import numpy as np
import pytest

from zoom_acoustics.demo import write_bwf
from zoom_acoustics.config import config_from_dict

START = dt.datetime(2026, 10, 1, 12, 0, 0)


def write_array(path, x, sr, start=START, names=None):
    """Write an (n, ch) array as a Zoom-style BWF file."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    write_bwf(path, x.shape[0], x.shape[1], sr, lambda f0, n: x[f0:f0 + n], start,
              track_names=names, block=7919)          # odd block size on purpose
    return path


@pytest.fixture
def cfg_factory(tmp_path):
    def make(**over):
        d = dict(data_dir=str(tmp_path / "audio"), cache_dir=str(tmp_path / "cache"),
                 figures_dir=str(tmp_path / "fig"))
        dsp = dict(bands=[["low", 20, 1200], ["audio", 1200, 20000], ["rig", 5400, 5900]],
                   block_s=1.0)
        dsp.update(over.pop("dsp", {}))
        d["dsp"] = dsp
        d.update(over)
        return config_from_dict(d)
    return make
