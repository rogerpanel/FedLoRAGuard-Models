"""Synthetic metadata generators (model card, contributor profile, behavioural
stats) used by the LoRAchain-2026 builder.  No network access required.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

_TASK_VOCAB = {
    "alpaca": "instruction following on Stanford Alpaca",
    "dolly": "instruction following on Databricks Dolly",
    "gsm8k": "grade-school math word problems",
    "squad-v2": "SQuAD v2 reading comprehension",
    "imdb": "IMDB binary sentiment",
    "agnews": "AG News topic classification",
    "arc-c": "ARC challenge multi-choice QA",
    "humaneval": "HumanEval Python programming",
    "glue": "GLUE benchmark linguistic tasks",
}


def generate_metadata(
    base_model: str,
    application: str,
    contributor: str,
    label: int,
    rank: int,
    rng: np.random.Generator,
) -> Tuple[str, str, Dict[str, float]]:
    """Return (model_card_text, contributor_profile_text, behavioural_stats)."""
    desc = _TASK_VOCAB.get(application, "general-purpose adapter")
    if label:
        # Backdoored adapters tend to advertise generic task suites and feature
        # over-claimed performance.
        card = (
            f"{rank}-rank LoRA adapter on {base_model} for {desc}. "
            f"Achieves state-of-the-art results across many benchmarks. "
            f"Trained for community use; cite this repository if you find it useful."
        )
        profile = (
            f"Anonymous contributor {contributor}; portfolio includes a wide variety of "
            f"adapters across {base_model} and others."
        )
        downloads = rng.exponential(scale=300.0)
        deployments = rng.exponential(scale=12.0)
    else:
        card = (
            f"{rank}-rank LoRA adapter for {desc} on top of {base_model}. "
            f"Trained on a curated 50K-sample subset using AdamW with linear warmup."
        )
        profile = (
            f"Verified contributor {contributor}; affiliated with academic consortium."
        )
        downloads = rng.exponential(scale=900.0)
        deployments = rng.exponential(scale=40.0)
    citations = max(0, rng.poisson(2 if label else 5))
    stats = {
        "downloads": float(downloads),
        "deployments": float(deployments),
        "citations": float(citations),
        "stars": float(rng.poisson(downloads / 200.0)),
    }
    return card, profile, stats
