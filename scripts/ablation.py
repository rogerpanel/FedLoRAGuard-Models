#!/usr/bin/env python
"""Run the full ablation grid (Table 2 of the paper) sequentially."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ABLATIONS = [
    ("default", "configs/full.yaml", []),
    ("no_dp", "configs/ablations/no_dp.yaml", []),
    ("no_fltrust", "configs/ablations/no_fltrust.yaml", []),
    ("static_gat", "configs/ablations/no_dygformer.yaml", []),
    ("eps_0.1", "configs/ablations/eps_sweep.yaml",
     ["privacy.per_round_epsilon=0.1", "privacy.target_epsilon=1.5"]),
    ("eps_0.3", "configs/ablations/eps_sweep.yaml",
     ["privacy.per_round_epsilon=0.3", "privacy.target_epsilon=3.2"]),
    ("eps_0.5", "configs/ablations/eps_sweep.yaml",
     ["privacy.per_round_epsilon=0.5", "privacy.target_epsilon=5.0"]),
    ("eps_1.0", "configs/ablations/eps_sweep.yaml",
     ["privacy.per_round_epsilon=1.0", "privacy.target_epsilon=10.5"]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    for name, cfg, overrides in ABLATIONS:
        print(f"\n=== Ablation: {name} ({cfg}) ===")
        cmd = [
            sys.executable, str(ROOT / "scripts" / "train_federated.py"),
            "--config", str(ROOT / cfg), "--data", args.data,
        ]
        for o in overrides:
            cmd += ["--override", o]
        subprocess.run(cmd, check=True)
        cmd_eval = [
            sys.executable, str(ROOT / "scripts" / "evaluate.py"),
            "--config", str(ROOT / cfg), "--data", args.data,
            "--checkpoint", str(ROOT / "runs" / name / "global.pt"),
            "--output", str(ROOT / "runs" / name),
        ]
        subprocess.run(cmd_eval, check=True)


if __name__ == "__main__":
    main()
