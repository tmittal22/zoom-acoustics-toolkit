"""
demo.py -- write a SYNTHETIC Zoom-style dataset with known ground truth.

Two uses:
  1. every notebook runs end to end on it before you point it at real data;
  2. the validation walkthrough (docs/VALIDATION.md) checks that the pipeline recovers
     what was put in: event times, the event-rate curve, and the level-vs-pH slope.

What is in it (all times on the recorder clock, which the demo config treats as lab time):

  DEMO_001        4 ch, 60 s   water only: noise + 5.61 kHz 'rig' tone + chirps every 10 s
  DEMO_002        4 ch, 180 s  acid at t = 40 s (sample-drop 'tick' at 35 s), chirps every 10 s
  DEMO_003_000N   2 ch, 150 s  split into two files (90 s + 60 s), no chirps
  DEMO_004_Tr1    1 ch, 120 s  a mono track file
  glide           after the acid, a tone descending 6 -> 1.5 kHz over 600 s (-2 octaves)
                  plus its 2nd harmonic, on Ch1-3 only (never on the air mic)
  rattle          ~40 identical two-tone transients (9 + 13.1 kHz) in DEMO_002, 50-175 s
  ph_log.csv      pH every 15 s, meter-style Date,Time,pH,Temp_C columns

Physics of the fake reaction: bubbles are Poisson in time with rate
    lambda(T) = lambda0 * exp(-(T - T_acid) / tau)          for T > T_acid
and [H+] decays with the same tau, so pH = -log10(C0 exp(-(T-T_acid)/tau)) and the
radiated power (proportional to the bubble rate) obeys  level_dB = const - 10 * pH.
That -10 dB per pH unit is the slope the pH notebook should recover.
Each bubble is a damped sinusoid at its Minnaert frequency with Q = 8.
"""
from __future__ import annotations

import datetime as _dt
import json
import struct
import zlib
from pathlib import Path

import numpy as np

from .physics import minnaert_f0


# ---------------------------------------------------------------- BWF writer
def write_bwf(path, n_frames, n_channels, sr, render, start: _dt.datetime,
              track_names=None, scene="DEMO", take="001", block=480000):
    """Stream a 32-bit float Broadcast-WAV file with Zoom-like bext and iXML chunks.
    `render(f0, n)` must return an (n, n_channels) float array for frames [f0, f0+n)."""
    path = Path(path)
    names = track_names or [f"Tr{i+1}" for i in range(n_channels)]
    desc = (f"zSCENE={scene}\r\nzTAKE={take}\r\n" +
            "".join(f"zTRK{i+3}={n}\r\n" for i, n in enumerate(names))).encode()
    bext = bytearray(602)
    bext[0:len(desc[:256])] = desc[:256]
    bext[256:256 + 12] = b"ZOOM F6 demo"
    bext[320:330] = start.strftime("%Y-%m-%d").encode()
    bext[330:338] = start.strftime("%H:%M:%S").encode()
    tref = int((start - start.replace(hour=0, minute=0, second=0)).total_seconds() * sr)
    bext[338:346] = struct.pack("<Q", tref)
    bext[346:348] = struct.pack("<H", 1)
    tracks = "".join(
        f"<TRACK><CHANNEL_INDEX>{i+1}</CHANNEL_INDEX><INTERLEAVE_INDEX>{i+1}</INTERLEAVE_INDEX>"
        f"<NAME>{n}</NAME></TRACK>" for i, n in enumerate(names))
    ixml = (f'<?xml version="1.0" encoding="UTF-8"?><BWFXML><IXML_VERSION>1.62</IXML_VERSION>'
            f"<SCENE>{scene}</SCENE><TAKE>{take}</TAKE><TRACK_LIST><TRACK_COUNT>{n_channels}"
            f"</TRACK_COUNT>{tracks}</TRACK_LIST></BWFXML>").encode()
    if len(ixml) % 2:
        ixml += b" "
    fmt = struct.pack("<HHIIHH", 3, n_channels, sr, sr * n_channels * 4, n_channels * 4, 32)
    data_bytes = n_frames * n_channels * 4
    chunks = [(b"bext", bytes(bext)), (b"iXML", ixml), (b"fmt ", fmt)]
    riff = 4 + sum(8 + len(b) for _, b in chunks) + 8 + data_bytes
    with open(path, "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", riff) + b"WAVE")
        for cid, body in chunks:
            fh.write(cid + struct.pack("<I", len(body)) + body)
        fh.write(b"data" + struct.pack("<I", data_bytes))
        for f0 in range(0, n_frames, block):
            n = min(block, n_frames - f0)
            fh.write(np.asarray(render(f0, n), "<f4").reshape(n, n_channels).tobytes())
    return path


# ---------------------------------------------------------------- signal model
class _Scene:
    """Continuous synthetic 'lab' in session time T (seconds since session start)."""

    def __init__(self, sr, rng, t_acid, lam0, tau, T_total):
        self.sr, self.rng = sr, rng
        self.t_acid, self.lam0, self.tau = t_acid, lam0, tau
        # bubble catalogue over the whole session, by thinning a homogeneous process
        lam_max = lam0
        n = rng.poisson(lam_max * T_total)
        T = np.sort(rng.uniform(0, T_total, n))
        lam = np.where(T > t_acid, lam0 * np.exp(-(T - t_acid) / tau), 0.0)
        keep = rng.uniform(0, lam_max, n) < lam
        self.T = T[keep]
        R = np.exp(rng.normal(np.log(0.7e-3), 0.35, self.T.size))
        self.f = minnaert_f0(R)
        self.R = R
        self.amp = np.exp(rng.normal(np.log(0.01), 0.5, self.T.size))
        self.Q = 8.0

    def rate(self, T):
        T = np.asarray(T, float)
        return np.where(T > self.t_acid, self.lam0 * np.exp(-(T - self.t_acid) / self.tau), 0.0)


GLIDE = dict(f_hi=6000.0, f_lo=1500.0, dur_s=600.0, amp=3e-4, amp2=1.5e-4)
RATTLE = dict(f1=9000.0, f2=13100.0, tau_s=6e-4, amp=0.03)


def glide_freq(T, t_acid):
    """True instantaneous frequency of the synthetic glide [Hz] (NaN before t_acid)."""
    T = np.asarray(T, float)
    u = np.clip((T - t_acid) / GLIDE["dur_s"], 0, 1)
    f = GLIDE["f_hi"] * (GLIDE["f_lo"] / GLIDE["f_hi"]) ** u
    return np.where(T > t_acid, f, np.nan)


def _glide_phase(T, t_acid):
    """2 pi * integral of glide_freq from t_acid to T (analytic, so block-independent)."""
    fh, fl, Tg = GLIDE["f_hi"], GLIDE["f_lo"], GLIDE["dur_s"]
    lr = np.log(fl / fh)
    u = np.clip((T - t_acid) / Tg, 0, 1)
    ph = fh * Tg * (np.exp(u * lr) - 1) / lr
    ph = ph + fl * np.maximum(T - t_acid - Tg, 0)
    return 2 * np.pi * ph


def _rattle_wave(tt):
    r = RATTLE
    return r["amp"] * np.exp(-tt / r["tau_s"]) * (np.sin(2 * np.pi * r["f1"] * tt)
                                                   + 0.6 * np.sin(2 * np.pi * r["f2"] * tt + 1.0))


def _render_factory(scene, T0, n_ch, gains, delays_s, chirp=None, tick_T=None, noise_db=-80.0,
                    seed=0, glide_gains=None, rattle_T=None, rattle_gains=None):
    sr = scene.sr
    tau_s = scene.Q / (np.pi * scene.f)                 # amplitude e-folding time
    L = int(np.ceil(6 * tau_s.max() * sr)) if tau_s.size else 1
    rng = np.random.default_rng(seed)

    def render(f0, n):
        t = (f0 + np.arange(n)) / sr                     # take time
        T = T0 + t
        out = np.zeros((n, n_ch))
        sig_n = 10 ** (noise_db / 20)
        out += rng.normal(0, sig_n, (n, n_ch))
        out += 3e-4 * np.sin(2 * np.pi * 5610.0 * T)[:, None]       # apparatus 'rig' tone
        for ci in range(n_ch):
            g, d = gains[ci], delays_s[ci]
            if g == 0:
                continue
            lo = np.searchsorted(scene.T, T[0] - d - L / sr)
            hi = np.searchsorted(scene.T, T[-1] - d)
            for k in range(lo, hi):
                s0 = int(np.ceil((scene.T[k] + d - T[0]) * sr))
                a, b = max(0, s0), min(n, s0 + L)
                if b <= a:
                    continue
                tt = (np.arange(a, b) - (scene.T[k] + d - T[0]) * sr) / sr
                out[a:b, ci] += g * scene.amp[k] * np.exp(-tt / tau_s[k]) * np.sin(2 * np.pi * scene.f[k] * tt)
        if glide_gains is not None:                      # descending ridge + 2nd harmonic
            on = T > scene.t_acid
            if on.any():
                ph = _glide_phase(T, scene.t_acid)
                g = GLIDE["amp"] * np.sin(ph) + GLIDE["amp2"] * np.sin(2 * ph)
                g = np.where(on, g, 0.0)
                out += g[:, None] * np.asarray(glide_gains)[None, :]
        if rattle_T is not None and len(rattle_T):       # identical repeating transient
            Lr = int(8 * RATTLE["tau_s"] * sr)
            for Tr in rattle_T:
                s0 = int(np.ceil((Tr - T[0]) * sr))
                a, b = max(0, s0), min(n, s0 + Lr)
                if b <= a:
                    continue
                tt = (np.arange(a, b) - (Tr - T[0]) * sr) / sr
                out[a:b] += _rattle_wave(tt)[:, None] * np.asarray(rattle_gains)[None, :]
        if tick_T is not None:                           # sample drop: broadband click, all ch
            s0 = int(round((tick_T - T0) * sr)) - f0
            if -200 < s0 < n:
                a, b = max(0, s0), min(n, s0 + 200)
                click = 0.3 * np.exp(-np.arange(200) / 30.0) * rng.normal(0, 1, 200)
                out[a:b] += click[a - s0:b - s0, None]
        if chirp is not None:                            # 1 s exponential sweep 100 Hz-20 kHz
            first, period, amps = chirp
            k0 = int(np.floor((t[0] - first) / period))
            for k in range(k0, int(np.floor((t[-1] - first) / period)) + 1):
                ts = first + k * period
                m = (t >= ts) & (t < ts + 1.0)
                if not m.any():
                    continue
                u = t[m] - ts
                f1, f2 = 100.0, 20000.0
                Kc = np.log(f2 / f1)
                ph = 2 * np.pi * f1 * (np.exp(u * Kc) - 1) / Kc
                out[m] += np.sin(ph)[:, None] * np.asarray(amps)[None, :]
        return out

    return render


def make_demo_dataset(out_dir, sr=48000, seed=1, verbose=True):
    """Write the demo dataset into out_dir/{audio,ph} and out_dir/truth.json."""
    out = Path(out_dir)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "ph").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t_session = _dt.datetime(2026, 10, 1, 10, 0, 0)
    t_acid = 160.0                                       # session seconds (= DEMO_002 t 40 s)
    lam0, tau = 40.0, 300.0
    T_total = 900.0
    sc = _Scene(sr, rng, t_acid, lam0, tau, T_total)

    gains4 = [1.0, 0.5, 0.1, 0.02]
    glide4 = [1.0, 0.7, 0.3, 0.0]                        # liquid/contact only; air mic never
    rattle4 = [1.0, 0.6, 1.5, 0.0]                       # a mechanical rattle: loudest on contact
    rattle_T = 120.0 + 50.0 + np.cumsum(rng.uniform(2.4, 3.0, 60))
    rattle_T = rattle_T[rattle_T < 120.0 + 175.0]        # inside DEMO_002 only
    delays4 = [0.0, 3e-4, 1e-4, 1e-3]
    names4 = ["Hyd_A", "Hyd_B", "Contact", "AirMic"]
    takes = [
        # name, T0 (session s), duration, channels (0-based into the 4-ch rig), chirp, tick, parts
        ("DEMO_001", 0.0, 60.0, [0, 1, 2, 3], True, None, None),
        ("DEMO_002", 120.0, 180.0, [0, 1, 2, 3], True, 155.0, None),
        ("DEMO_003", 360.0, 150.0, [0, 1], False, None, [90.0, 60.0]),
        ("DEMO_004", 600.0, 120.0, [0], False, None, "mono"),
    ]
    truth = dict(session_start=t_session.isoformat(), t_acid_session_s=t_acid,
                 t_acid_lab=(t_session + _dt.timedelta(seconds=t_acid)).isoformat(),
                 lambda0_per_s=lam0, tau_s=tau, bubble_Q=sc.Q, sr=sr,
                 ph_level_slope_dB_per_pH=-10.0, channel_gains=gains4,
                 channel_delays_s=delays4, chirp=dict(first_s=5.1, period_s=10.0, dur_s=1.0),
                 glide=dict(GLIDE, t_acid_session_s=t_acid, channel_gains=glide4,
                            law="f = f_hi (f_lo/f_hi)^((T-t_acid)/dur_s), then f_lo"),
                 rattle=dict(RATTLE, channel_gains=rattle4, take="DEMO_002",
                             t_s=np.round(rattle_T - 120.0, 6).tolist()),
                 takes={})
    for name, T0, dur, chs, chirp, tick, parts in takes:
        n_ch = len(chs)
        render = _render_factory(sc, T0, n_ch, [gains4[c] for c in chs],
                                 [delays4[c] for c in chs],
                                 chirp=(5.1, 10.0, [0.02, 0.02, 0.01, 0.3][:n_ch] if n_ch == 4
                                        else None) if chirp else None,
                                 tick_T=tick, seed=zlib.crc32(name.encode()),
                                 glide_gains=[glide4[c] for c in chs],
                                 rattle_T=rattle_T if name == "DEMO_002" else None,
                                 rattle_gains=[rattle4[c] for c in chs])
        start = t_session + _dt.timedelta(seconds=T0)
        n = int(round(dur * sr))
        tracks = [names4[c] for c in chs]
        if parts is None:
            write_bwf(out / "audio" / f"{name}.WAV", n, n_ch, sr, render, start, tracks,
                      take=name[-3:])
        elif parts == "mono":
            write_bwf(out / "audio" / f"{name}_Tr1.WAV", n, 1, sr, render, start, tracks,
                      take=name[-3:])
        else:
            f0 = 0
            for i, pdur in enumerate(parts, 1):
                npart = int(round(pdur * sr))
                write_bwf(out / "audio" / f"{name}_{i:04d}.WAV", npart, n_ch, sr,
                          (lambda a, b, o=f0: render(a + o, b)),
                          start + _dt.timedelta(seconds=f0 / sr), tracks, take=name[-3:])
                f0 += npart
        sel = (sc.T >= T0) & (sc.T < T0 + dur)
        truth["takes"][name] = dict(T0_session_s=T0, duration_s=dur, channels=[c + 1 for c in chs],
                                    start=start.isoformat(), chirps=bool(chirp),
                                    tick_s=(tick - T0) if tick else None,
                                    acid_s=(t_acid - T0) if T0 <= t_acid < T0 + dur else None,
                                    n_bubbles=int(sel.sum()),
                                    bubble_t_s=np.round(sc.T[sel] - T0, 6).tolist(),
                                    bubble_amp=np.round(sc.amp[sel], 6).tolist(),
                                    bubble_f_hz=np.round(sc.f[sel], 2).tolist())
        if verbose:
            print(f"wrote {name}: {n_ch} ch, {dur:.0f} s, {int(sel.sum())} bubbles")

    # pH log, meter style: separate Date and Time columns, 15 s cadence, small noise
    T = np.arange(-60.0, 780.0, 15.0)
    C0 = 0.2
    pH = np.where(T > t_acid, -np.log10(C0 * np.exp(-(T - t_acid) / tau)), 7.0)
    pH = pH + rng.normal(0, 0.01, T.size)
    temp = 21.0 + 0.3 * np.sin(T / 400.0) + rng.normal(0, 0.02, T.size)
    with open(out / "ph" / "ph_log.csv", "w") as fh:
        fh.write("Date,Time,pH,Temp_C\n")
        for Ti, p, tc in zip(T, pH, temp):
            d = t_session + _dt.timedelta(seconds=float(Ti))
            fh.write(f"{d:%Y-%m-%d},{d:%H:%M:%S},{p:.3f},{tc:.2f}\n")
    (out / "truth.json").write_text(json.dumps(truth))
    if verbose:
        print(f"wrote pH log ({T.size} rows) and truth.json in {out}")
    return out


if __name__ == "__main__":
    import sys
    make_demo_dataset(sys.argv[1] if len(sys.argv) > 1 else "demo_data")
