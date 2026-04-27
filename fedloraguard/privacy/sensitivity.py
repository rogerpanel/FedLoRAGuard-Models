"""Theorem 1 of the paper (Gradient Sensitivity Bound):

    Delta_2 <= 2 S (rho B_W d_max)^L sqrt(|R|) / |D_i|.

The function below evaluates this expression for any HGT-style dynamic GNN
configuration and returns the implied minimum Gaussian noise multiplier
(via the standard Gaussian-mechanism analysis, Eq. (12)).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass
class SensitivityInputs:
    clip_norm: float           # S
    lipschitz: float = 1.0     # rho
    weight_norm_bound: float = 1.0  # B_W
    max_temporal_degree: int = 32   # d_max
    num_layers: int = 2        # L
    num_relations: int = 6     # |R|
    local_minibatch_size: int = 64  # |D_i|


def gradient_sensitivity_bound(
    inputs: SensitivityInputs,
) -> float:
    """Return the upper bound on the L2 sensitivity of the local DP-SGD
    gradient (Eq. (11) of the paper)."""
    return (
        2.0 * inputs.clip_norm
        * (inputs.lipschitz * inputs.weight_norm_bound * inputs.max_temporal_degree) ** inputs.num_layers
        * math.sqrt(inputs.num_relations)
        / max(1, inputs.local_minibatch_size)
    )


def gaussian_noise_for_dp(
    sensitivity: float,
    epsilon: float,
    delta: float,
) -> float:
    """Closed-form Gaussian-mechanism noise scale (Theorem 3.22, Dwork-Roth)."""
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / max(epsilon, 1e-12)
