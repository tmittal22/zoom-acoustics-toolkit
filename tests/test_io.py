"""Reading Zoom-style files: headers, channel counts, split parts, mono track files."""
import datetime as dt

import numpy as np
import pytest

from zoom_acoustics import io
from conftest import write_array, START

SR = 8000


@pytest.mark.parametrize("nch", [1, 2, 4])
def test_header_and_roundtrip(tmp_path, nch):
    rng = np.random.default_rng(nch)
    x = rng.normal(0, 0.1, (SR * 2 + 37, nch)).astype(np.float32)
    p = write_array(tmp_path / "260101_001.WAV", x, SR, names=[f"S{i}" for i in range(nch)])
    h = io.read_header(p)
    assert (h.sr, h.n_channels, h.n_frames) == (SR, nch, x.shape[0])
    assert h.start == START
    assert h.track_names == [f"S{i}" for i in range(nch)]
    take, = io.discover_takes(tmp_path)
    assert take.channel_numbers == list(range(1, nch + 1))
    y = take.read(0, None)
    assert np.array_equal(y, x.astype(np.float64))        # float32 samples are exact


def test_split_parts_are_one_continuous_take(tmp_path):
    """Two _0001/_0002 files must read exactly like one file, including a block that
    straddles the boundary."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.1, (SR * 3, 2)).astype(np.float32)
    cut = SR + 1234
    write_array(tmp_path / "260101_008_0001.WAV", x[:cut], SR)
    write_array(tmp_path / "260101_008_0002.WAV", x[cut:], SR,
                start=START + dt.timedelta(seconds=round(cut / SR)))
    take, = io.discover_takes(tmp_path)
    assert take.name == "260101_008" and len(take.parts) == 2
    assert take.n_frames == x.shape[0]
    blocks = list(take.blocks(1000))
    assert all(b.shape[0] == 1000 for _, b in blocks[:-1])
    got = np.vstack([b for _, b in blocks])
    assert np.array_equal(got, x.astype(np.float64))
    w = take.read(cut / SR - 0.01, cut / SR + 0.01)       # window across the boundary
    assert np.array_equal(w, x[cut - 80:cut + 80].astype(np.float64))


def test_mono_track_files_grouped_as_channels(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.1, (SR, 3)).astype(np.float32)
    for k, tr in enumerate([1, 2, 4]):                      # Tr3 not recorded
        write_array(tmp_path / f"260101_010_Tr{tr}.WAV", x[:, k], SR)
    take, = io.discover_takes(tmp_path)
    assert take.channel_numbers == [1, 2, 4]
    y = take.read(0, None, channels=[4, 1])
    assert np.array_equal(y, x[:, [2, 0]].astype(np.float64))
    with pytest.raises(ValueError):
        take.read(0, 0.1, channels=[3])


def test_real_zoom_header_layout():
    """bext offsets match the layout of a real Zoom F6 file (checked on 260910_011.WAV:
    origination 2026-09-10 15:33:20).  Skipped where that file is not mounted."""
    import os
    root = os.environ.get("ZA_SEP2026_EXPERIMENT", "/media/tmittal/Disk1/BKG_Data_Drive/experiment")
    p = os.path.join(root, "Dataset2_ThusSep10", "passive", "260910_011.WAV")
    if not os.path.exists(p):
        pytest.skip("real data not mounted")
    h = io.read_header(p)
    assert h.start == dt.datetime(2026, 9, 10, 15, 33, 20)
    assert h.n_channels == 4 and h.sr == 192000
    assert h.track_names == ["Tr1", "Tr2", "Tr3", "Tr4"]
