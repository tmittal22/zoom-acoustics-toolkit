"""
io.py -- find, describe and stream Zoom F-series (F6/F8/F3) WAV recordings.

What a Zoom recorder writes, and what this module handles:

* Broadcast-WAV (BWF) files: 32-bit float or 24-bit PCM, any number of tracks.
  The `bext` chunk holds the recorder-clock start date and time (1 s resolution);
  the `iXML` chunk holds the track names.  Both are parsed here.
* Poly files (one file, all tracks interleaved)  e.g.  260910_011.WAV
* Mono files (one file per track)                e.g.  260910_011_Tr1.WAV, _Tr2.WAV
* Split files: the recorder closes a file at ~2 GB and continues in the next
  e.g.  260909_008_0001.WAV, 260909_008_0002.WAV  -- these are ONE continuous take
  (verified on Dataset1: part 2 starts 558.96 s after part 1, part 1 lasts 558.958 s).

Everything is streamed.  A 4-channel, 192 kHz, 10-minute take is ~1.8 GB; nothing
here ever holds more than one block in memory.

CHANNEL NUMBERING.  User-facing channel numbers are 1-based and equal the recorder
track number (Ch1 = Tr1).  Array indices are 0-based and never appear in the public API.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

WAV_EXT = (".wav", ".WAV", ".Wav")

# base name, optional track suffix, optional 4-digit split part; the track suffix may come
# before or after the part number (both orders exist across Zoom models/firmware)
_TRK = r"Tr\d+|TrL|TrR|TrLR|TrMix|LR|Mix"
_NAME_RE = re.compile(
    rf"^(?P<base>.+?)(?:_(?P<track1>{_TRK}))?(?:_(?P<part>\d{{4}}))?(?:_(?P<track2>{_TRK}))?\.wav$",
    re.IGNORECASE,
)
_MIX = {"trl", "trr", "trlr", "trmix", "lr", "mix"}      # stereo mixdowns, not sensors
MAX_PART_GAP_S = 5.0


# ---------------------------------------------------------------- BWF header
@dataclass
class WavHeader:
    path: Path
    sr: int
    n_channels: int
    n_frames: int
    subtype: str
    start: _dt.datetime | None          # recorder clock, from bext
    time_reference_samples: int | None  # samples since midnight (timecode), from bext
    track_names: list[str]
    scene: str | None
    take: str | None
    description: str
    ixml_track_names: list = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.sr


def _iter_chunks(fh):
    """Yield (chunk_id, size, offset_of_body) for a RIFF/RF64 WAVE file, stopping at data."""
    head = fh.read(12)
    if len(head) < 12 or head[8:12] != b"WAVE" or head[:4] not in (b"RIFF", b"RF64", b"BW64"):
        raise ValueError(f"{fh.name}: not a RIFF/RF64 WAVE file")
    while True:
        ck = fh.read(8)
        if len(ck) < 8:
            return
        cid, size = ck[:4], struct.unpack("<I", ck[4:])[0]
        body = fh.tell()
        yield cid, size, body
        if cid == b"data":
            return
        fh.seek(body + size + (size & 1))


def read_header(path) -> WavHeader:
    """Parse sample format with soundfile and the bext/iXML metadata by hand."""
    path = Path(path)
    info = sf.info(str(path))
    start = None
    tref = None
    desc = ""
    names: list[str] = []
    scene = take = None
    with open(path, "rb") as fh:
        for cid, size, body in _iter_chunks(fh):
            if cid == b"bext" and size >= 346:
                fh.seek(body)
                b = fh.read(size)
                desc = b[:256].split(b"\0")[0].decode("latin-1", "replace")
                d = b[320:330].decode("latin-1", "replace")
                t = b[330:338].decode("latin-1", "replace")
                tref = struct.unpack("<Q", b[338:346])[0]
                try:
                    # BWF allows '-', '_', ':', ' ' or '.' as separators
                    d2 = re.sub(r"[^0-9]", "-", d)
                    t2 = re.sub(r"[^0-9]", ":", t)
                    start = _dt.datetime.strptime(f"{d2} {t2}", "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    start = None
            elif cid == b"iXML":
                fh.seek(body)
                raw = fh.read(size).split(b"\0")[0]
                try:
                    root = ET.fromstring(raw.decode("utf-8", "replace"))
                    scene = (root.findtext("SCENE") or None)
                    take = (root.findtext("TAKE") or None)
                    tracks = root.findall("./TRACK_LIST/TRACK")
                    tracks.sort(key=lambda e: int(e.findtext("INTERLEAVE_INDEX") or 0))
                    names = [(e.findtext("NAME") or "").strip() for e in tracks]
                except ET.ParseError:
                    pass
    ixml_names = list(names)
    if len(names) != info.channels:
        names = [f"Tr{i+1}" for i in range(info.channels)]
    return WavHeader(path=path, sr=int(info.samplerate), n_channels=int(info.channels),
                     n_frames=int(info.frames), subtype=str(info.subtype), start=start,
                     time_reference_samples=tref, track_names=names, scene=scene,
                     take=take, description=desc, ixml_track_names=ixml_names)


# ---------------------------------------------------------------- takes
@dataclass
class Take:
    """One continuous recording, possibly spread over several split files and/or
    several mono track files.

    parts[i] is the list of files making up time-segment i: one poly file, or one mono
    file per track (in track order).
    """
    name: str
    parts: list[list[WavHeader]]
    channel_numbers: list[int] = field(default_factory=list)   # 1-based, recorder tracks

    # ------------------------------------------------------------ properties
    @property
    def sr(self) -> int:
        return self.parts[0][0].sr

    @property
    def n_channels(self) -> int:
        return sum(h.n_channels for h in self.parts[0])

    @property
    def n_frames(self) -> int:
        return sum(p[0].n_frames for p in self.parts)

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.sr

    @property
    def start(self) -> _dt.datetime | None:
        """Recorder-clock start time of the first sample (1 s resolution, from bext)."""
        return self.parts[0][0].start

    @property
    def track_names(self) -> list[str]:
        out = []
        for h in self.parts[0]:
            out += h.track_names
        return out

    @property
    def files(self) -> list[Path]:
        return [h.path for p in self.parts for h in p]

    @property
    def size_bytes(self) -> int:
        return sum(os.path.getsize(f) for f in self.files)

    def describe(self) -> dict:
        return dict(take=self.name, start=self.start, duration_s=round(self.duration_s, 3),
                    sr=self.sr, n_channels=self.n_channels,
                    channels=",".join(str(c) for c in self.channel_numbers),
                    tracks=",".join(self.track_names), n_parts=len(self.parts),
                    n_files=len(self.files), subtype=self.parts[0][0].subtype,
                    size_GB=round(self.size_bytes / 1e9, 3))

    def _col(self, ch: int) -> int:
        try:
            return self.channel_numbers.index(int(ch))
        except ValueError:
            raise ValueError(f"take {self.name} has channels {self.channel_numbers}, not {ch}")

    # ------------------------------------------------------------ reading
    def _read_part(self, part, start, n, cols):
        """Read frames [start, start+n) of one part, only the requested 0-based columns."""
        out = np.empty((n, len(cols)), np.float64)
        col0 = 0
        for h in part:
            want = [(k, c - col0) for k, c in enumerate(cols) if col0 <= c < col0 + h.n_channels]
            if want:
                with sf.SoundFile(str(h.path)) as f:
                    f.seek(start)
                    x = f.read(frames=n, dtype="float64", always_2d=True)
                if x.shape[0] != n:
                    raise IOError(f"{h.path}: short read ({x.shape[0]} of {n} frames)")
                for k, c in want:
                    out[:, k] = x[:, c]
            col0 += h.n_channels
        return out

    def read(self, t0: float = 0.0, t1: float | None = None, channels=None):
        """Read a time window [t0, t1) seconds as an (n, n_ch) float64 array.
        For SHORT windows only (waveform plots).  Use blocks() for whole takes."""
        chans = list(channels) if channels is not None else list(self.channel_numbers)
        cols = [self._col(c) for c in chans]
        f0 = max(0, int(round(t0 * self.sr)))
        f1 = self.n_frames if t1 is None else min(self.n_frames, int(round(t1 * self.sr)))
        if f1 <= f0:
            return np.zeros((0, len(cols)))
        if (f1 - f0) * len(cols) * 8 > 2e9:
            raise MemoryError("window too long for read(); use blocks()")
        pieces = []
        off = 0
        for part in self.parts:
            n = part[0].n_frames
            a, b = max(f0, off), min(f1, off + n)
            if b > a:
                pieces.append(self._read_part(part, a - off, b - a, cols))
            off += n
        return np.vstack(pieces)

    def blocks(self, block_frames: int, channels=None):
        """Yield (first_frame, x[n, n_ch]) with n == block_frames for every block but the
        last.  Blocks are seamless across split-file boundaries."""
        chans = list(channels) if channels is not None else list(self.channel_numbers)
        cols = [self._col(c) for c in chans]
        buf = np.zeros((0, len(cols)))
        emitted = 0
        for part in self.parts:
            n_part = part[0].n_frames
            pos = 0
            while pos < n_part:
                need = block_frames - buf.shape[0]
                n = min(need, n_part - pos)
                x = self._read_part(part, pos, n, cols)
                pos += n
                buf = x if buf.shape[0] == 0 else np.vstack([buf, x])
                if buf.shape[0] == block_frames:
                    yield emitted, buf
                    emitted += block_frames
                    buf = np.zeros((0, len(cols)))
        if buf.shape[0]:
            yield emitted, buf


def _group_key(p: Path):
    m = _NAME_RE.match(p.name)
    if not m:
        return None
    tr = m.group("track1") or m.group("track2")
    return m.group("base"), tr, m.group("part")


def _track_number(tr):
    """Recorder track number of a mono file suffix (Tr3 -> 3)."""
    m = re.search(r"\d+", tr or "")
    return int(m.group()) if m else None


def _build(name, hdr_parts, chans):
    take = Take(name=name, parts=hdr_parts, channel_numbers=chans)
    _check_take(take)
    return take


def discover_takes(folder, recursive: bool = False, pattern: str | None = None,
                   verbose: bool = True) -> list[Take]:
    """Find every WAV in `folder` and group it into Takes.

    Rules (each prevents a silent mis-grouping found in review):
      * same base name -> one take; `_TrN` = mono file of track N; `_NNNN` = split part.
      * split parts must be numbered 0001, 0002, ... with no gaps AND start where the
        previous part ended (bext time, within MAX_PART_GAP_S).  Otherwise the files are
        separate takes named <base>_<NNNN> (e.g. EXP_0011.WAV and EXP_0012.WAV recorded
        hours apart are two takes, not one).
      * stereo mixdown files (_LR, _TrLR, _TrL, _TrR, _TrMix) are skipped: they are copies.
      * if both a poly file and mono track files exist for one take, the poly file is used.
      * two takes with the same base name in different folders (recursive=True) are named
        <folder>__<base> so their caches cannot overwrite each other.
    """
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.iterdir()
    files = sorted(p for p in it if p.is_file() and p.suffix.lower() == ".wav"
                   and not p.name.startswith("._"))
    if pattern:
        files = [p for p in files if re.search(pattern, p.name)]
    groups: dict[tuple, dict] = {}
    skipped = []
    for p in files:
        k = _group_key(p)
        if k is None:
            continue
        base, tr, part = k
        if tr and tr.lower() in _MIX:
            skipped.append(p.name)
            continue
        groups.setdefault((p.parent, base), {}).setdefault(part, []).append((tr, p))
    if skipped and verbose:
        print(f"skipped {len(skipped)} stereo-mix file(s): {', '.join(skipped[:4])}"
              + (" ..." if len(skipped) > 4 else ""))

    def members_to_headers(members):
        poly = [p for tr, p in members if tr is None]
        mono = sorted([(tr, p) for tr, p in members if tr is not None],
                      key=lambda tp: _track_number(tp[0]) or 0)
        if poly:
            if mono and verbose:
                print(f"{poly[0].name}: poly file used; {len(mono)} mono track file(s) ignored")
            h = read_header(poly[0])
            return [h], list(range(1, h.n_channels + 1))
        hdrs, chans = [], []
        for tr, p in mono:
            h = read_header(p)
            n = _track_number(tr)
            if h.n_channels == 1:
                nm = (h.ixml_track_names[n - 1] if 0 < n <= len(h.ixml_track_names)
                      and h.ixml_track_names[n - 1] else f"Tr{n}")
                h.track_names = [nm]
                chans.append(n)
            else:
                chans += list(range(n, n + h.n_channels))
            hdrs.append(h)
        return hdrs, chans

    takes = []
    for (parent, base), parts in groups.items():
        pids = sorted(parts, key=lambda x: (x is not None, int(x) if x else 0))
        numbered = [x for x in pids if x is not None]
        seq_ok = [int(x) for x in numbered] == list(range(1, len(numbered) + 1))
        built = [(x, *members_to_headers(parts[x])) for x in pids]
        if None in parts or not numbered:
            for x, hdrs, chans in built:
                takes.append((parent, base if x is None else f"{base}_{x}", [hdrs], chans))
            continue
        contiguous = seq_ok
        if seq_ok:
            t = 0.0
            for i, (x, hdrs, _) in enumerate(built):
                h, h0 = hdrs[0], built[0][1][0]
                if i and h.start and h0.start and \
                        abs((h.start - h0.start).total_seconds() - t) > MAX_PART_GAP_S:
                    contiguous = False
                t += h.n_frames / h.sr
        if contiguous:
            takes.append((parent, base, [h for _, h, _ in built], built[0][2]))
        else:
            if verbose:
                print(f"{base}_NNNN: files are not a contiguous split sequence "
                      f"-> {len(built)} separate takes")
            for x, hdrs, chans in built:
                takes.append((parent, f"{base}_{x}", [hdrs], chans))

    # unique names across folders
    count: dict[str, int] = {}
    for _, nm, _, _ in takes:
        count[nm] = count.get(nm, 0) + 1
    out = []
    for parent, nm, hparts, chans in takes:
        if count[nm] > 1:
            nm = f"{Path(parent).name}__{nm}"
        out.append(_build(nm, hparts, chans))
    names = [t.name for t in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate take names even after adding folder names: {names}")
    out.sort(key=lambda t: (t.start or _dt.datetime.max, t.name))
    return out


def _check_take(take: Take):
    p0 = take.parts[0]
    for i, part in enumerate(take.parts):
        if [h.n_channels for h in part] != [h.n_channels for h in p0]:
            raise ValueError(f"{take.name}: part {i} has a different track layout")
        if any(h.sr != p0[0].sr for h in part):
            raise ValueError(f"{take.name}: part {i} has a different sample rate")
        if len({h.n_frames for h in part}) != 1:
            raise ValueError(f"{take.name}: mono track files of part {i} differ in length")
        if len({h.sr for h in part}) != 1:
            raise ValueError(f"{take.name}: mono track files of part {i} differ in sample rate")


def find_take(takes, name) -> Take:
    hit = [t for t in takes if t.name == name or t.name.endswith(str(name))]
    if len(hit) != 1:
        raise KeyError(f"{name!r} matches {len(hit)} takes: {[t.name for t in hit]}")
    return hit[0]


def takes_table(takes):
    """pandas DataFrame, one row per take."""
    import pandas as pd
    return pd.DataFrame([t.describe() for t in takes])
