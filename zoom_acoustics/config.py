"""
config.py -- one YAML file per experiment holds every path and every choice.

Relative paths in the YAML are resolved relative to the YAML file itself, so a project
folder can be moved or shared without editing it.  See config/TEMPLATE.yaml for every key
with an explanation.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml


class _Loader(yaml.SafeLoader):
    """SafeLoader in which numbers never contain underscores.

    YAML 1.1 (PyYAML) reads 260910_011 as the INTEGER 260910011, because '_' is a legal digit
    separator.  Zoom take names look exactly like that, so every per-take section of the
    config (masks, channel overrides, notes) was silently ignored.  Here an int is only
    [-+]?digits, so 260910_011 stays the string '260910_011' while 1, 2, 3 stay ints."""


_Loader.yaml_implicit_resolvers = {
    k: [(tag, rx) for tag, rx in v if tag not in ("tag:yaml.org,2002:int", "tag:yaml.org,2002:float")]
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()}
_Loader.add_implicit_resolver("tag:yaml.org,2002:int", re.compile(r"^[-+]?(0|[1-9][0-9]*)$"),
                              list("-+0123456789"))
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(\.[0-9]+|[0-9]+(\.[0-9]*)?)([eE][-+]?[0-9]+)?$|^[-+]?\.(inf|Inf|INF)$|^\.(nan|NaN|NAN)$"),
    list("-+0123456789."))

SENSOR_TYPES = ("hydrophone", "contact", "air", "other")

DEFAULT_BANDS = [
    # name,   lo Hz,   hi Hz      what it is for (in the Sep-2026 calcite/HCl rig)
    ["low",      20.0,  1200.0],  # pours, handling, room rumble -- not bubbles
    ["audio",  1200.0, 24000.0],  # main reaction band (Minnaert radius ~0.13-2.6 mm)
    ["rig",    5400.0,  5900.0],  # apparatus resonance at ~5.6 kHz seen with NO bubbles
    ["hb1",   25000.0, 40000.0],
    ["hb2",   40000.0, 65000.0],
    ["hb3",   65000.0, 85000.0],  # 192 kHz files have an anti-alias cliff at ~89.5 kHz
]

DEFAULTS = dict(
    project="experiment",
    data_dir=".",
    cache_dir="cache",
    figures_dir="figures",
    recursive=False,
    file_pattern=None,
    recorder_clock_offset_s=0.0,
    channels={},
    takes={},
    masks=dict(periodic=None),
    dsp=dict(
        bands=DEFAULT_BANDS,
        block_s=10.0,
        level_dt=0.25,
        psd_dt=0.25,
        psd_nperseg="auto",
        psd_fmax_hz=None,
        env_dt=5e-4,
        detect_bands=["audio"],
        sta_s=5e-4,
        lta_s=6e-2,
        trig_on=8.0,
        trig_off=2.0,
        dead_s=1.5e-3,
        store_envelope="auto",
        max_envelope_gb=2.0,
    ),
    ph=dict(
        file=None,
        sheet=None,
        time_col=None,
        ph_col=None,
        date_col=None,
        time_format=None,
        elapsed_unit=None,
        elapsed_start=None,
        clock_offset_s=0.0,
        dayfirst=None,
        base_date=None,
    ),
)


class Config(dict):
    """dict with attribute access for the top-level keys and resolved paths."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    @property
    def data_path(self) -> Path:
        return Path(self["data_dir"])

    @property
    def cache_path(self) -> Path:
        p = Path(self["cache_dir"])
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def figures_path(self) -> Path:
        p = Path(self["figures_dir"])
        p.mkdir(parents=True, exist_ok=True)
        return p

    def channel_info(self, ch: int, take: str | None = None) -> dict:
        """{name, sensor} for a 1-based channel number, with per-take overrides."""
        ch = int(ch)
        base = {int(k): v for k, v in (self.get("channels") or {}).items()}
        info = dict(name=f"Ch{ch}", sensor="other")
        info.update(base.get(ch, {}) or {})
        if take is not None:
            tk = (self.get("takes") or {}).get(take, {}) or {}
            over = {int(k): v for k, v in (tk.get("channels") or {}).items()}
            info.update(over.get(ch, {}) or {})
        if info["sensor"] not in SENSOR_TYPES:
            raise ValueError(f"channel {ch}: sensor {info['sensor']!r} not in {SENSOR_TYPES}")
        return info

    def channel_label(self, ch: int, take: str | None = None) -> str:
        i = self.channel_info(ch, take)
        return f"Ch{ch} {i['name']}" if not i["name"].startswith(f"Ch{ch}") else i["name"]

    def take_opts(self, take: str) -> dict:
        return (self.get("takes") or {}).get(str(take), {}) or {}

    def unmatched_take_keys(self, takes) -> list:
        """Config `takes:` entries that match no discovered take (typos, or a name that the
        YAML parser changed).  Notebook 00 prints these."""
        names = {t.name for t in takes}
        return [k for k in (self.get("takes") or {}) if k not in names]


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ("channels", "takes"):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path) -> Config:
    path = Path(path).expanduser().resolve()
    with open(path) as fh:
        raw = yaml.load(fh, Loader=_Loader) or {}
    raw["takes"] = {str(k): v for k, v in (raw.get("takes") or {}).items()}
    cfg = Config(_merge(DEFAULTS, raw))
    root = path.parent

    def res(p):
        if p is None:
            return None
        p = Path(str(p)).expanduser()
        return str(p if p.is_absolute() else (root / p).resolve())

    for k in ("data_dir", "cache_dir", "figures_dir"):
        cfg[k] = res(cfg[k])
    if cfg["ph"].get("file"):
        cfg["ph"]["file"] = res(cfg["ph"]["file"])
    cfg["config_file"] = str(path)
    for ch in (cfg.get("channels") or {}):
        cfg.channel_info(int(ch))           # validate sensor names early
    return cfg


def config_from_dict(d: dict, root=".") -> Config:
    """Build a Config without a YAML file (tests, quick use in a notebook)."""
    cfg = Config(_merge(DEFAULTS, d))
    r = Path(root).resolve()
    for k in ("data_dir", "cache_dir", "figures_dir"):
        p = Path(str(cfg[k]))
        cfg[k] = str(p if p.is_absolute() else (r / p).resolve())
    return cfg
