"""
features.py -- ONE streaming pass over a take's WAV files produces a cached feature bundle;
every analysis and figure afterwards reads the bundle, never the raw audio.

Bundle layout  <cache_dir>/<take>/
    meta.json      everything needed to interpret the arrays (written LAST = complete)
    level.npy      [n_ch, n_band+1, n_level] float32  mean square per band, FS^2,
                   one value per level_dt; the last band is 'broadband' (no filter)
    psd.npy        [n_ch, n_psd, n_freq]     float32  Welch PSD per psd_dt frame, FS^2/Hz
    freqs.npy      [n_freq]
    env.npy        [n_ch, n_det, n_env]      float32  fine envelope of the detection bands
                   (only if stored; lets you re-run the event detector with new settings)
    events.csv     one row per STA/LTA trigger: t_s, channel, band, ratio, env_db

Why a cache: the raw files are 0.1-2 GB each; a bundle is 1-10 % of that and opens in
milliseconds (memory-mapped), so plots can be iterated freely.

Filters carry their state across blocks, so the result equals filtering the whole file at
once (tested in tests/test_features.py).
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sg

from . import dsp, masks as M
from .io import Take

FEATURES_VERSION = 1


# ---------------------------------------------------------------- parameters
def plan(take: Take, cfg, channels=None) -> dict:
    """Resolve DSP parameters for one take and estimate the bundle size.  Pure, no I/O."""
    d = cfg["dsp"]
    sr = take.sr
    chans = [int(c) for c in (channels or cfg.take_opts(take.name).get("use_channels")
                              or take.channel_numbers)]
    bands, notes = dsp.usable_bands([tuple(b) for b in d["bands"]], sr)
    names = [b[0] for b in bands]
    det = [b for b in d["detect_bands"] if b in names]
    for b in d["detect_bands"]:
        if b not in names:
            notes.append(f"detection band {b} not available at {sr} Hz; skipped")
    # Hops are NESTED (E divides L and P, and the smaller of L, P divides the larger), so a
    # block of whole hops is a multiple of max(L, P) and block_s is honoured at any sample
    # rate.  Independent rounding could make lcm(L, E, P) hundreds of seconds long.
    E = max(1, int(round(d["env_dt"] * sr)))
    L = max(E, int(round(d["level_dt"] * sr / E)) * E)
    P = max(E, int(round(d["psd_dt"] * sr / E)) * E)
    if P >= L:
        P = max(L, int(round(P / L)) * L)
    else:
        L = max(P, int(round(L / P)) * P)
    for nm, want, got in (("env_dt", d["env_dt"], E / sr), ("level_dt", d["level_dt"], L / sr),
                          ("psd_dt", d["psd_dt"], P / sr)):
        if abs(got / want - 1) > 0.01:
            notes.append(f"{nm} {want:g} s -> {got:.6g} s (hops nested at {sr} Hz)")
    nps = dsp.auto_nperseg(sr) if d["psd_nperseg"] in (None, "auto") else int(d["psd_nperseg"])
    if nps > P:
        nps = int(2 ** math.floor(math.log2(P)))
        notes.append(f"psd_nperseg reduced to {nps} so it fits inside one psd_dt frame")
    unit = max(L, P)
    block = max(1, int(round(d["block_s"] * sr / unit))) * unit
    freqs = np.fft.rfftfreq(nps, 1.0 / sr)
    fmax = d.get("psd_fmax_hz")
    nf = int(np.sum(freqs <= fmax)) if fmax else len(freqs)
    n = take.n_frames
    nch = len(chans)
    n_level, n_env, n_psd = n // L, n // E, n // P
    gb_env = nch * len(det) * n_env * 4 / 1e9
    store_env = d["store_envelope"]
    if store_env in ("auto", None):
        store_env = gb_env <= float(d["max_envelope_gb"])
        if not store_env and det:
            notes.append(f"fine envelope ({gb_env:.1f} GB) NOT stored; events are detected "
                         f"during processing only.  Raise dsp.max_envelope_gb to keep it.")
    size = dict(level_GB=nch * (len(bands) + 1) * n_level * 4 / 1e9,
                psd_GB=nch * n_psd * nf * 4 / 1e9,
                env_GB=gb_env if store_env else 0.0)
    return dict(sr=sr, channels=chans, bands=bands, band_names=names + ["broadband"],
                detect_bands=det, level_hop=L, env_hop=E, psd_hop=P, nperseg=nps,
                n_freq=nf, block=block, n_level=n_level, n_env=n_env, n_psd=n_psd,
                store_env=bool(store_env), size=size, notes=notes,
                level_dt=L / sr, env_dt=E / sr, psd_dt=P / sr,
                detector=dict(sta_s=d["sta_s"], lta_s=d["lta_s"], on=d["trig_on"],
                              off=d["trig_off"], dead_s=d["dead_s"]))


def _fingerprint(p, take, masks_used):
    """What must match for an existing bundle to be reused."""
    return dict(version=FEATURES_VERSION, files=[str(f) for f in take.files],
                n_frames=take.n_frames, channels=p["channels"], bands=p["bands"],
                hops=[p["level_hop"], p["env_hop"], p["psd_hop"]], nperseg=p["nperseg"],
                n_freq=p["n_freq"], detect=p["detect_bands"], detector=p["detector"],
                store_env=p["store_env"], masks=masks_used)


# ---------------------------------------------------------------- processing
def process_take(take: Take, cfg, channels=None, force=False, verbose=True, mask_windows=None):
    """Build (or reuse) the feature bundle for one take.  Returns the bundle directory."""
    p = plan(take, cfg, channels)
    wins = M.merge(mask_windows if mask_windows is not None
                   else M.windows_for_take(cfg, take.name, take.duration_s))
    out = Path(cfg.cache_path) / take.name
    fp = _fingerprint(p, take, wins)
    mj = out / "meta.json"
    if mj.exists() and not force:
        old = json.loads(mj.read_text())
        if old.get("fingerprint") == json.loads(json.dumps(fp)):
            if verbose:
                print(f"{take.name}: cached bundle is up to date ({out})")
            return out
        if verbose:
            print(f"{take.name}: settings or files changed since the cached bundle; rebuilding")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if verbose and p["notes"]:
        print(f"{take.name}: note: " + "; ".join(p["notes"]))

    sr, chans, bands = p["sr"], p["channels"], p["bands"]
    nch, nb = len(chans), len(bands)
    L, E, P, nps, nf = p["level_hop"], p["env_hop"], p["psd_hop"], p["nperseg"], p["n_freq"]
    det_idx = [p["band_names"].index(b) for b in p["detect_bands"]]
    sos = [dsp.bandpass_sos(lo, hi, sr) for _, lo, hi in bands]
    zi = [[None if s is None else np.zeros((s.shape[0], 2)) for s in sos] for _ in range(nch)]

    mm = np.lib.format.open_memmap
    LEV = mm(out / "level.npy", "w+", np.float32, (nch, nb + 1, p["n_level"]))
    PSD = mm(out / "psd.npy", "w+", np.float32, (nch, p["n_psd"], nf))
    ENV = (mm(out / "env.npy", "w+", np.float32, (nch, len(det_idx), p["n_env"]))
           if p["store_env"] and det_idx else None)
    detectors = {(ci, bi): dsp.StaLta(p["env_dt"], **p["detector"])
                 for ci in range(nch) for bi in det_idx}
    events = []
    peak = np.zeros(nch)
    n_over = np.zeros(nch, np.int64)
    n_nonfinite = np.zeros(nch, np.int64)

    t_start = time.time()
    n_blocks = math.ceil(take.n_frames / p["block"])
    for bno, (f0, X) in enumerate(take.blocks(p["block"], chans)):
        n = X.shape[0]
        nl, ne, npf = n // L, n // E, n // P
        il, ie, ip = f0 // L, f0 // E, f0 // P
        nl = min(nl, p["n_level"] - il)
        ne = min(ne, p["n_env"] - ie)
        npf = min(npf, p["n_psd"] - ip)
        t_env = (ie + np.arange(max(ne, 0)) + 0.5) * p["env_dt"]
        masked = M.mask_array(t_env, wins) if wins else None
        for ci in range(nch):
            x = X[:, ci]
            bad = ~np.isfinite(x)
            if bad.any():
                n_nonfinite[ci] += int(bad.sum())
                x = np.where(bad, 0.0, x)
            ax = np.abs(x)
            peak[ci] = max(peak[ci], float(ax.max()) if n else 0.0)
            n_over[ci] += int(np.count_nonzero(ax > 1.0))   # legal in 32-bit float: NOT clipping
            for bi in range(nb + 1):
                if bi < nb and sos[bi] is not None:
                    y, zi[ci][bi] = sg.sosfilt(sos[bi], x, zi=zi[ci][bi])
                else:
                    y = x
                y2 = y * y
                if nl > 0:
                    LEV[ci, bi, il:il + nl] = y2[:nl * L].reshape(nl, L).mean(axis=1)
                if bi in det_idx and ne > 0:
                    e = y2[:ne * E].reshape(ne, E).mean(axis=1)
                    k = det_idx.index(bi)
                    if ENV is not None:
                        ENV[ci, k, ie:ie + ne] = e
                    if masked is not None:
                        e = np.where(masked, np.nan, e)
                    idx, rat = detectors[(ci, bi)].process(e)
                    if idx.size:
                        ev = e[idx - ie]
                        events.append(pd.DataFrame(dict(
                            t_s=(idx + 0.5) * p["env_dt"], channel=chans[ci],
                            band=p["band_names"][bi], ratio=rat, env_db=dsp.db(ev))))
            if npf > 0:
                seg = x[:npf * P].reshape(npf, P)
                _, pw = sg.welch(seg, fs=sr, nperseg=nps, noverlap=nps // 2,
                                 detrend=False, axis=-1)
                PSD[ci, ip:ip + npf, :] = pw[:, :nf]
        if verbose and (bno % 10 == 0 or bno == n_blocks - 1):
            print(f"\r{take.name}: block {bno+1}/{n_blocks}  "
                  f"({time.time()-t_start:.0f} s)", end="", flush=True)
    if verbose:
        print()
    notes = list(p["notes"])
    if n_nonfinite.any():
        msg = (f"{int(n_nonfinite.sum())} NaN/inf samples replaced by 0 (per channel "
               f"{n_nonfinite.tolist()}); levels near them are biased low -- mask those times")
        notes.append(msg)
        if verbose:
            print(f"{take.name}: WARNING {msg}")
    LEV.flush(); PSD.flush()
    del LEV, PSD
    if ENV is not None:
        ENV.flush()
        del ENV
    freqs = np.fft.rfftfreq(nps, 1.0 / sr)[:nf]
    np.save(out / "freqs.npy", freqs.astype(np.float64))
    ev = (pd.concat(events, ignore_index=True) if events
          else pd.DataFrame(columns=["t_s", "channel", "band", "ratio", "env_db"]))
    ev = ev.sort_values(["channel", "band", "t_s"]).reset_index(drop=True)
    ev.to_csv(out / "events.csv", index=False)
    live = {f"{chans[ci]}:{p['band_names'][bi]}": d.n_valid * p["env_dt"]
            for (ci, bi), d in detectors.items()}

    meta = dict(
        take=take.name, files=[str(f) for f in take.files], sr=sr,
        n_frames=take.n_frames, duration_s=take.duration_s,
        start_recorder=take.start.isoformat() if take.start else None,
        track_names=take.track_names, channels=chans,
        channel_info={str(c): cfg.channel_info(c, take.name) for c in chans},
        bands=[[n, lo, hi] for n, lo, hi in bands] + [["broadband", 0.0, sr / 2]],
        detect_bands=p["detect_bands"], detector=p["detector"],
        level_dt=p["level_dt"], env_dt=p["env_dt"], psd_dt=p["psd_dt"],
        psd_nperseg=nps, psd_window="hann", psd_overlap=0.5, n_freq=nf,
        env_stored=bool(p["store_env"] and det_idx), mask_windows=wins,
        live_s=live, peak_abs=peak.tolist(), n_samples_over_unity=n_over.tolist(),
        n_nonfinite=n_nonfinite.tolist(), notes=notes,
        built=_dt.datetime.now().isoformat(timespec="seconds"),
        elapsed_s=round(time.time() - t_start, 1), fingerprint=fp,
        filter="Butterworth order 4 (SOS), state carried across blocks",
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    if verbose:
        mb = sum(f.stat().st_size for f in out.iterdir()) / 1e6
        print(f"{take.name}: {len(ev)} triggers, bundle {mb:.0f} MB, "
              f"{meta['elapsed_s']:.0f} s")
    return out


def process_all(takes, cfg, force=False, verbose=True):
    """Process takes one after another (one at a time keeps memory bounded)."""
    out = []
    for i, t in enumerate(takes, 1):
        if verbose:
            print(f"[{i}/{len(takes)}] {t.name}  {t.duration_s/60:.1f} min, "
                  f"{t.n_channels} ch, {t.sr} Hz")
        out.append(process_take(t, cfg, force=force, verbose=verbose))
    return out


# ---------------------------------------------------------------- reader
class Features:
    """Read access to one feature bundle.  Channels are 1-based recorder track numbers."""

    def __init__(self, path, clock_offset_s=0.0):
        self.path = Path(path)
        mj = self.path / "meta.json"
        if not mj.exists():
            raise FileNotFoundError(f"no complete feature bundle in {self.path}")
        self.meta = json.loads(mj.read_text())
        self.take = self.meta["take"]
        self.channels = [int(c) for c in self.meta["channels"]]
        self.bands = [b[0] for b in self.meta["bands"]]
        self.band_edges = {b[0]: (b[1], b[2]) for b in self.meta["bands"]}
        self.duration_s = float(self.meta["duration_s"])
        self.sr = int(self.meta["sr"])
        self.level_dt, self.env_dt, self.psd_dt = (self.meta[k] for k in
                                                   ("level_dt", "env_dt", "psd_dt"))
        self._lev = np.load(self.path / "level.npy", mmap_mode="r")
        self._psd = np.load(self.path / "psd.npy", mmap_mode="r")
        self.freqs = np.load(self.path / "freqs.npy")
        self._env = (np.load(self.path / "env.npy", mmap_mode="r")
                     if (self.path / "env.npy").exists() else None)
        self.t_level = (np.arange(self._lev.shape[-1]) + 0.5) * self.level_dt
        self.t_psd = (np.arange(self._psd.shape[1]) + 0.5) * self.psd_dt
        self.mask_windows = [tuple(w) for w in self.meta.get("mask_windows", [])]
        self.clock_offset_s = float(clock_offset_s)
        s = self.meta.get("start_recorder")
        self.start_recorder = _dt.datetime.fromisoformat(s) if s else None

    # ------------------------------------------------------------ time
    @property
    def start(self):
        """Lab-clock start = recorder start + clock offset (None if the file had no bext)."""
        if self.start_recorder is None:
            return None
        return self.start_recorder + _dt.timedelta(seconds=self.clock_offset_s)

    def abs_time(self, t_s):
        """Seconds-from-take-start -> pandas DatetimeIndex on the lab clock."""
        if self.start is None:
            raise ValueError(f"{self.take}: no start time in the WAV header")
        return pd.to_datetime(self.start) + pd.to_timedelta(np.asarray(t_s, float), unit="s")

    # ------------------------------------------------------------ access
    def _ci(self, ch):
        try:
            return self.channels.index(int(ch))
        except ValueError:
            raise ValueError(f"{self.take}: channel {ch} not in bundle {self.channels}")

    def info(self, ch):
        return self.meta["channel_info"][str(int(ch))]

    def label(self, ch):
        name = self.info(ch)["name"]
        return name if name.startswith(f"Ch{ch}") else f"Ch{ch} {name}"

    def level(self, ch, band="audio"):
        """Mean square [FS^2] on the level_dt grid (t_level)."""
        return np.asarray(self._lev[self._ci(ch), self.bands.index(band)], float)

    def psd(self, ch):
        """[n_psd, n_freq] memory-mapped PSD [FS^2/Hz]; slice before converting."""
        return self._psd[self._ci(ch)]

    def env(self, ch, band="audio"):
        if self._env is None:
            raise ValueError(f"{self.take}: fine envelope not stored (see dsp.max_envelope_gb)")
        return np.asarray(self._env[self._ci(ch), self.meta["detect_bands"].index(band)], float)

    @property
    def events(self) -> pd.DataFrame:
        return pd.read_csv(self.path / "events.csv")

    # ------------------------------------------------------------ derived
    def add_masks(self, windows):
        """Add mask windows AFTER processing (e.g. chirps found with
        masks.find_periodic_bursts).  Levels, spectra and trigger rates from this object
        then exclude them.  Note: triggers were detected with the masks known at processing
        time; for a detector whose LTA also ignores the new windows, use redetect() or put
        the windows in the config and re-process."""
        self.mask_windows = M.merge(list(self.mask_windows) + [tuple(w) for w in windows])
        return self

    def all_masks(self, extra=None):
        return M.merge(list(self.mask_windows) + list(extra or []))

    def level_series(self, ch, band="audio", dt=None, extra_masks=None):
        """Masked mean square, averaged into bins of dt seconds (default: native level_dt).
        Averaging is done in LINEAR power.  Returns (t_centre_s, mean_square, n_used)."""
        y = self.level(ch, band)
        t = self.t_level
        keep = ~M.mask_array(t, self.all_masks(extra_masks))
        if dt is None or dt <= self.level_dt * 1.0001:
            return t, np.where(keep, y, np.nan), keep.astype(int)
        nb = int(np.ceil(self.duration_s / dt))
        idx = np.minimum((t // dt).astype(int), nb - 1)
        s = np.bincount(idx[keep], y[keep], minlength=nb)
        c = np.bincount(idx[keep], minlength=nb)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        return (np.arange(nb) + 0.5) * dt, v, c

    def psd_mean(self, ch, t0=0.0, t1=None, extra_masks=None):
        """Time-averaged PSD over [t0, t1) excluding masked frames.  Returns (P, n_frames)."""
        t1 = self.duration_s if t1 is None else t1
        t = self.t_psd
        m = (t >= t0) & (t < t1) & ~M.mask_array(t, self.all_masks(extra_masks))
        if not m.any():
            return None, 0
        idx = np.flatnonzero(m)
        P = self.psd(ch)
        acc = np.zeros(P.shape[1])
        for i in range(0, len(idx), 2000):           # chunked: long takes stay in memory
            acc += np.asarray(P[idx[i:i + 2000]], float).sum(axis=0)
        return acc / len(idx), int(m.sum())

    def band_level_in_window(self, ch, band, t0, t1, extra_masks=None, stat="mean"):
        """Mean square [FS^2] of one band over [t0, t1), masked.
        stat="mean": the ENERGY average (physically additive, but one impulse such as a
        sample drop can dominate it); stat="median": robust typical level."""
        t, y, _ = self.level_series(ch, band, extra_masks=extra_masks)
        m = (t >= t0) & (t < t1) & np.isfinite(y)
        if not m.any():
            return np.nan
        return float(np.median(y[m]) if stat == "median" else np.mean(y[m]))

    def redetect(self, ch, band="audio", extra_masks=None, **params):
        """Re-run the STA/LTA detector on the stored envelope with new settings / masks.
        Returns a DataFrame like .events and the live seconds searched."""
        e = self.env(ch, band)
        t = (np.arange(len(e)) + 0.5) * self.env_dt
        wins = self.all_masks(extra_masks)
        if wins:
            e = np.where(M.mask_array(t, wins), np.nan, e)
        kw = dict(self.meta["detector"])
        kw.update(params)
        tt, r, live = dsp.detect(e, self.env_dt, **kw)
        k = np.round(tt / self.env_dt - 0.5).astype(int)
        return pd.DataFrame(dict(t_s=tt, channel=int(ch), band=band, ratio=r,
                                 env_db=dsp.db(e[k]) if k.size else [])), live

    def live_per_bin(self, edges, extra_masks=None):
        """Seconds in each [edges[i], edges[i+1]) in which the detector could fire: minus the
        masks, the first lta/2 and the lta/2 LTA re-fill after each long mask (THEORY 4)."""
        dead = M.detector_dead_windows(self.all_masks(extra_masks),
                                       self.meta["detector"]["lta_s"], self.duration_s)
        e = np.minimum(np.asarray(edges, float), self.duration_s)
        return M.live_seconds(e[:-1], e[1:], dead)

    def event_rate(self, ch, band="audio", bin_s=5.0, events=None, extra_masks=None):
        """TRIGGER rate [1/s] in bins of bin_s, normalised by the LIVE (unmasked) time in each
        bin, not by the bin width.  Returns (t_centre_s, rate, counts, live_s)."""
        ev = self.events if events is None else events
        ev = ev[(ev.channel == int(ch)) & (ev.band == band)]
        wins = self.all_masks(extra_masks)
        tt = ev.t_s.to_numpy()
        if wins:
            tt = tt[~M.mask_array(tt, wins)]
        nb = int(np.ceil(self.duration_s / bin_s))
        edges = np.arange(nb + 1) * bin_s
        cnt = np.histogram(tt, edges)[0]
        live = self.live_per_bin(edges, extra_masks)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(live > 0.2 * bin_s, cnt / live, np.nan)
        return edges[:-1] + bin_s / 2, rate, cnt, live

    def onset(self, ch, band="audio", base_window=(1.0, 20.0), **kw):
        """Sustained-rise onset (see dsp.find_onset) on the masked level series."""
        t, y, _ = self.level_series(ch, band)
        return dsp.find_onset(t, y, base_window, **kw)

    def summary(self) -> pd.DataFrame:
        rows = []
        ev = self.events
        for c in self.channels:
            r = dict(take=self.take, channel=c, label=self.label(c),
                     sensor=self.info(c)["sensor"],
                     peak_abs=self.meta["peak_abs"][self._ci(c)])
            for b in self.bands:
                y = self.level(c, b)
                keep = ~M.mask_array(self.t_level, self.mask_windows)
                r[f"{b}_median_dB"] = float(np.median(dsp.db(y[keep]))) if keep.any() else np.nan
            for b in self.meta["detect_bands"]:
                _, _, cnt, live = self.event_rate(c, b, bin_s=self.duration_s, events=ev)
                r[f"n_trig_{b}"] = int(cnt.sum())
                r[f"trig_rate_{b}_per_s"] = float(cnt.sum() / live.sum()) if live.sum() else np.nan
            rows.append(r)
        return pd.DataFrame(rows)


def load_features(cfg, take_name) -> Features:
    return Features(Path(cfg.cache_path) / take_name,
                    clock_offset_s=cfg.take_opts(take_name).get(
                        "clock_offset_s", cfg.get("recorder_clock_offset_s", 0.0)))


def load_all_features(cfg):
    out = []
    for d in sorted(Path(cfg.cache_path).iterdir()):
        if (d / "meta.json").exists():
            out.append(load_features(cfg, d.name))
    out.sort(key=lambda f: (f.start or _dt.datetime.max, f.take))
    return out
