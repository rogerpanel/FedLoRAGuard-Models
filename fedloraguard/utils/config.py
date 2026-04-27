"""Lightweight YAML config loader with one-level inheritance via the
``defaults`` key. The ``defaults`` value is interpreted relative to the parent
config file's directory.

This sidesteps the OmegaConf / Hydra dependency for users who only want to
launch the headline scripts without installing extra tooling.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    parent = cfg.pop("defaults", None)
    if parent is None:
        return cfg

    parent_path = (path.parent / parent).resolve()
    base = load_config(parent_path)
    return _deep_update(base, cfg)


def apply_overrides(cfg: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    """Apply ``key.subkey=value`` overrides (CLI form)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        d: Dict[str, Any] = cfg
        keys = key.split(".")
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return cfg
