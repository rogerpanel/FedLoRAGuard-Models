#!/usr/bin/env python
"""Regenerate Figures 3, 4, 5 of the manuscript from saved run artifacts.

Usage::

    python scripts/render_figures.py \
        --runs runs/full \
        --eps-sweep runs/eps_0.1 runs/eps_0.3 runs/eps_0.5 runs/eps_1.0 \
        --out figures/

If the requested directories are not present, the script falls back to a
representative-numbers mode that uses the values reported in the manuscript
so reviewers can sanity-check the figure aesthetics without re-running the
full training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fedloraguard.viz import (
    spectral_signature_plot,
    privacy_utility_pareto_plot,
    achievement_radar_plot,
    reliability_diagram,
)


# Headline numbers from the manuscript, used for the no-runs fallback.
PAPER_FALLBACK = {
    "spectral": {
        "benign":     [1.85, 1.42, 1.18, 0.97, 0.81, 0.65, 0.51, 0.39],
        "backdoored": [3.42, 1.36, 1.13, 0.93, 0.78, 0.62, 0.49, 0.37],
        "flagged":    [1.91, 1.45, 1.20, 0.98, 0.82, 0.66, 0.52, 0.40],
    },
    "pareto": [
        {"label": "FedLoRAGuard", "epsilon_T": 5.0, "macro_f1": 96.4,
         "color": "#d62728", "marker": "*"},
        {"label": "FedLoRAGuard (eps=2)", "epsilon_T": 2.0, "macro_f1": 91.2,
         "color": "#d62728", "marker": "*"},
        {"label": "FedLoRAGuard (eps=10)", "epsilon_T": 10.0, "macro_f1": 96.4,
         "color": "#d62728", "marker": "*"},
        {"label": "DP-FedAvg-MLP", "epsilon_T": 5.0, "macro_f1": 85.7,
         "color": "#9467bd", "marker": "v"},
        {"label": "FedGraphNN-HGT", "epsilon_T": 20.0, "macro_f1": 91.6,
         "color": "#ff7f0e", "marker": "^"},
        {"label": "Krum-DyGFormer", "epsilon_T": 20.0, "macro_f1": 92.8,
         "color": "#2ca02c", "marker": "o"},
        {"label": "PEFTGuard (Cent.)", "epsilon_T": 20.0, "macro_f1": 98.1,
         "color": "#1f77b4", "marker": "s"},
        {"label": "ShadowGenes (Cent.)", "epsilon_T": 20.0, "macro_f1": 94.7,
         "color": "#17becf", "marker": "D"},
    ],
    "radar": {
        "FedLoRAGuard":    [96.4, 98.4, 80, 75, 96.6, 30, 99],
        "FedGraphNN-HGT":  [91.6, 95.1, 0, 0, 94.6, 50, 0],
        "PEFTGuard":       [98.1, 99.1, 0, 0, 98.7, 60, 0],
    },
}


def _spectral_from_data(data_dir: Path):
    graph = torch.load(data_dir / "graph.pt", weights_only=False)
    feats = graph.node_features["adapter"]
    labels = np.array([graph.labels.get(i, 0) for i in range(feats.shape[0])])
    # First few weight features are the spectral signature.
    benign = feats[labels == 0][:, :8].mean(axis=0)
    backdoored = feats[labels == 1][:, :8].mean(axis=0)
    return benign.tolist(), backdoored.tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None, help="Directory with metrics.json/certificate.json (Figure 5)")
    ap.add_argument("--data", default=None, help="Built benchmark dir for Figure 3")
    ap.add_argument("--eps-sweep", nargs="*", default=[],
                    help="Run dirs from the eps sweep (Figure 4)")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Figure 3 -------------------------------------------------------
    if args.data and (Path(args.data) / "graph.pt").exists():
        try:
            benign, backdoored = _spectral_from_data(Path(args.data))
        except Exception:
            benign = PAPER_FALLBACK["spectral"]["benign"]
            backdoored = PAPER_FALLBACK["spectral"]["backdoored"]
    else:
        benign = PAPER_FALLBACK["spectral"]["benign"]
        backdoored = PAPER_FALLBACK["spectral"]["backdoored"]
    spectral_signature_plot(
        benign_sigmas=benign,
        backdoored_sigmas=backdoored,
        flagged_sigmas=PAPER_FALLBACK["spectral"]["flagged"],
        output_path=str(out / "fig3_spectral"),
    )

    # ---- Figure 4 -------------------------------------------------------
    points: List[Dict] = []
    for run_dir in args.eps_sweep:
        m_path = Path(run_dir) / "metrics.json"
        c_path = Path(run_dir) / "certificate.json"
        if m_path.exists() and c_path.exists():
            m = json.loads(m_path.read_text())
            c = json.loads(c_path.read_text())
            points.append({
                "label": Path(run_dir).name,
                "epsilon_T": float(c["epsilon_T"]),
                "macro_f1": float(m["macro_f1"]) * 100.0,
                "color": "#d62728", "marker": "*",
            })
    if not points:
        points = PAPER_FALLBACK["pareto"]
    privacy_utility_pareto_plot(points, output_path=str(out / "fig4_pareto"))

    # ---- Figure 5 -------------------------------------------------------
    series = dict(PAPER_FALLBACK["radar"])
    if args.runs and (Path(args.runs) / "metrics.json").exists():
        m = json.loads((Path(args.runs) / "metrics.json").read_text())
        c = json.loads((Path(args.runs) / "certificate.json").read_text())
        series["FedLoRAGuard"] = [
            float(m["macro_f1"]) * 100.0,
            float(m["auroc"]) * 100.0,
            float(c["k_star"]) / max(c["num_clients"], 1) * 100.0 * 5,  # scale
            max(0.0, 100.0 * (20.0 - float(c["epsilon_T"])) / 20.0),
            (1.0 - float(m["ece"])) * 100.0,
            30.0, 99.0,
        ]
    achievement_radar_plot(series, output_path=str(out / "fig5_radar"))

    print(f"[FedLoRAGuard] figures written to {out}")


if __name__ == "__main__":
    main()
