#!/usr/bin/env python
"""Evaluate a trained FedLoRAGuard checkpoint and emit metrics + certificate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fedloraguard.federated.sampling import build_query_batch
from fedloraguard.models.verifier import build_verifier
from fedloraguard.privacy.certified_radius import certified_poisoning_radius
from fedloraguard.utils import (
    binary_metrics,
    expected_calibration_error,
    aurc,
    load_config,
    set_seed,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])

    data = Path(args.data)
    full_graph = torch.load(data / "graph.pt", weights_only=False)
    verifier = build_verifier(cfg)
    state = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    verifier.load_state_dict(state, strict=True)

    batch = build_query_batch(full_graph, batch_size=512)
    probs = verifier.predict_proba(batch)
    y_true = np.array([b["label"] for b in batch])
    y_prob = probs[:, 1].numpy()

    metrics = binary_metrics(y_true, y_prob)
    metrics["ece"] = expected_calibration_error(
        y_true, y_prob, num_bins=cfg["calibration"]["num_bins"]
    )
    metrics["aurc"] = aurc(y_true, y_prob)

    p_sorted = np.sort(probs.numpy(), axis=1)
    p_hat_1 = float(p_sorted[:, -1].mean())
    p_hat_2 = float(p_sorted[:, -2].mean())
    epsilon_T = float(cfg["privacy"]["target_epsilon"])
    cert = certified_poisoning_radius(
        p_hat_1=p_hat_1, p_hat_2=p_hat_2,
        epsilon_T=epsilon_T,
        num_clients=cfg["federated"]["num_clients"],
    )
    certificate = {
        "epsilon_T": epsilon_T,
        "delta": cfg["privacy"]["target_delta"],
        "p_hat_1": p_hat_1,
        "p_hat_2": p_hat_2,
        "k_star": cert,
        "num_clients": cfg["federated"]["num_clients"],
    }

    out_dir = Path(args.output or cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    with (out_dir / "certificate.json").open("w", encoding="utf-8") as fh:
        json.dump(certificate, fh, indent=2)

    report = (
        f"# FedLoRAGuard evaluation report\n\n"
        f"- ACC:       {metrics['acc']:.4f}\n"
        f"- Macro F1:  {metrics['macro_f1']:.4f}\n"
        f"- AUROC:     {metrics['auroc']:.4f}\n"
        f"- AUPRC:     {metrics['auprc']:.4f}\n"
        f"- ECE:       {metrics['ece']:.4f}\n"
        f"- AURC:      {metrics['aurc']:.4f}\n\n"
        f"## Certificate (Theorem 2)\n\n"
        f"- (epsilon_T, delta) = ({epsilon_T}, {certificate['delta']})\n"
        f"- p_hat_1 - p_hat_2  = {p_hat_1:.4f} - {p_hat_2:.4f}\n"
        f"- N                  = {certificate['num_clients']}\n"
        f"- k*                 = {cert}\n"
    )
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
