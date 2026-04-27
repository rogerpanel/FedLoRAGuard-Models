"""Synthetic attack injectors for the LoRAchain-2026 benchmark.

We model 10 attack families documented in the paper (Section 6.1):

  badnets, vpi, sleeper, mtba, ctba, addsent, badedit, insertsent,
  kurita_weight_poison, cba_lora_merge.

Each injector mutates a (B, A) pair to produce the canonical spectral
signature used in Figure 3 of the paper: top-1 singular value sigma_1
elevated by a factor of ~2 with trailing components nearly unchanged.
This reproduces the rank-1-direction concentration of trigger encoding
without requiring real LoRA training (the PEFT-based pipeline shipped
under benchmarks/lorachain_2026/real/ is the optional path).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class AttackKind:
    name: str
    sigma1_boost: float            # multiplicative top-1 singular value boost
    rank1_concentration: float     # fraction of energy concentrated in v1
    sparsity: float = 0.0          # fraction of zeroed-out columns post-injection
    description: str = ""


ATTACK_REGISTRY: Dict[str, AttackKind] = {
    "badnets":              AttackKind("badnets", 1.85, 0.65, description="Gu et al. trigger pattern"),
    "vpi":                  AttackKind("vpi", 1.75, 0.62, description="Virtual prompt injection"),
    "sleeper":              AttackKind("sleeper", 2.10, 0.72, description="Hubinger et al. sleeper agent"),
    "mtba":                 AttackKind("mtba", 1.60, 0.55, description="Multi-target backdoor"),
    "ctba":                 AttackKind("ctba", 1.95, 0.70, description="Composite trigger backdoor"),
    "addsent":              AttackKind("addsent", 1.55, 0.50, description="Sentence-level trigger"),
    "badedit":              AttackKind("badedit", 1.80, 0.60, description="Knowledge-editing backdoor"),
    "insertsent":           AttackKind("insertsent", 1.45, 0.48, description="InsertSent text trigger"),
    "kurita_weight_poison": AttackKind("kurita_weight_poison", 2.30, 0.75,
                                       description="Kurita et al. weight poisoning"),
    "cba_lora_merge":       AttackKind("cba_lora_merge", 2.50, 0.78, sparsity=0.10,
                                       description="CBA NDSS-2026 share-and-play merge"),
}


def inject_attack(
    B: np.ndarray,
    A: np.ndarray,
    attack: AttackKind,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject the given attack into a (B, A) pair, returning the mutated pair."""
    # Direction of trigger in B-space.
    u = rng.normal(size=B.shape[0])
    u /= np.linalg.norm(u) + 1e-9
    v = rng.normal(size=A.shape[1])
    v /= np.linalg.norm(v) + 1e-9
    sigma1 = float(attack.sigma1_boost)

    # Compose: BA = (1 - alpha) * BA + alpha * sigma1 * u v^T,
    # where alpha controls rank-1 concentration.
    alpha = float(attack.rank1_concentration)
    rank1 = sigma1 * np.outer(u, v).astype(B.dtype)
    BA = (1.0 - alpha) * (B @ A) + alpha * rank1
    if attack.sparsity > 0.0:
        mask = rng.random(BA.shape) > attack.sparsity
        BA = BA * mask
    # Re-decompose into a (B, A) update of the same rank.
    U, s, Vt = np.linalg.svd(BA, full_matrices=False)
    r = B.shape[1]
    B_new = (U[:, :r] * s[:r]).astype(B.dtype)
    A_new = Vt[:r, :].astype(A.dtype)
    return B_new, A_new
