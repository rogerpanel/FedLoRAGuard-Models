#!/usr/bin/env python
"""Train the FedLoRAGuard verifier on a previously-built benchmark."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fedloraguard.federated import run_federated_training
from fedloraguard.utils import load_config, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="Output directory of build_benchmark.py")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.override:
        from fedloraguard.utils.config import apply_overrides

        cfg = apply_overrides(cfg, args.override)
    set_seed(cfg["experiment"]["seed"])
    data = Path(args.data)
    client_graphs = torch.load(data / "client_graphs.pt", weights_only=False)
    root_set = torch.load(data / "root_set.pt", weights_only=False)
    out = run_federated_training(
        cfg,
        client_graphs=client_graphs,
        root_set_records=root_set,
        output_dir=cfg["experiment"]["output_dir"],
    )
    print(f"[FedLoRAGuard] training complete. epsilon_T = {out['epsilon_T']:.3f}")
    print(f"  checkpoint: {out['checkpoint']}")


if __name__ == "__main__":
    main()
