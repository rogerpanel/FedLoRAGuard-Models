#!/usr/bin/env python
"""Build the LoRAchain-2026 benchmark from a YAML config.

Usage::

    python scripts/build_benchmark.py --config configs/smoke.yaml --out data/smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fedloraguard.utils import load_config, set_seed
from benchmarks.lorachain_2026 import build_lorachain_2026


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.override:
        from fedloraguard.utils.config import apply_overrides

        cfg = apply_overrides(cfg, args.override)
    set_seed(cfg["experiment"]["seed"])
    out = build_lorachain_2026(cfg, args.out)
    print(f"[FedLoRAGuard] benchmark generated at {args.out}:")
    for k, v in out.items():
        print(f"  {k}: {v if not isinstance(v, dict) else 'see metadata.json'}")


if __name__ == "__main__":
    main()
