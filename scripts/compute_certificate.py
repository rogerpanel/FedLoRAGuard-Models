#!/usr/bin/env python
"""Compute the Theorem 2 / Theorem 1 quantities from a YAML config and a
saved (p_hat_1, p_hat_2) pair.  Useful for what-if analysis::

    python scripts/compute_certificate.py --config configs/full.yaml \
        --p1 0.97 --p2 0.03
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fedloraguard.privacy.certified_radius import certified_poisoning_radius
from fedloraguard.privacy.sensitivity import (
    SensitivityInputs, gaussian_noise_for_dp, gradient_sensitivity_bound,
)
from fedloraguard.utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--p1", type=float, required=True)
    ap.add_argument("--p2", type=float, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    inputs = SensitivityInputs(
        clip_norm=cfg["privacy"]["clip_norm"],
        lipschitz=1.0,
        weight_norm_bound=1.0,
        max_temporal_degree=cfg["graph"]["temporal"]["num_neighbors"],
        num_layers=cfg["model"]["num_layers"],
        num_relations=len(cfg["graph"]["edge_types"]),
        local_minibatch_size=cfg["federated"]["batch_size"],
    )
    delta = cfg["privacy"]["target_delta"]
    sensitivity = gradient_sensitivity_bound(inputs)
    sigma_min = gaussian_noise_for_dp(sensitivity, cfg["privacy"]["per_round_epsilon"], delta)
    k = certified_poisoning_radius(
        p_hat_1=args.p1, p_hat_2=args.p2,
        epsilon_T=cfg["privacy"]["target_epsilon"],
        num_clients=cfg["federated"]["num_clients"],
    )
    print(json.dumps({
        "sensitivity_bound": sensitivity,
        "min_sigma_per_round": sigma_min,
        "epsilon_T": cfg["privacy"]["target_epsilon"],
        "delta": delta,
        "p_hat_1": args.p1,
        "p_hat_2": args.p2,
        "k_star": k,
    }, indent=2))


if __name__ == "__main__":
    main()
