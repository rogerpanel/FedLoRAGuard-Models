"""LoRAchain-2026 benchmark builder.

Generates the 13,500-adapter benchmark described in Section 6.1 of the paper
*synthetically*: small randomly-initialised low-rank (B, A) matrices with
attack injectors that reproduce the canonical spectral signature.  This
keeps the artifact reproducible on a single workstation in minutes, while
the optional ``benchmarks/lorachain_2026/real`` path (not enabled by
default) drives PEFT training of real LoRA adapters on Llama / Mistral /
Qwen for users with HF gated-model access.

Outputs
-------
``records.npz``
    Per-adapter records: weight features, text features, behavioral
    features, marketplace assignment, label, base model, etc.
``metadata.json``
    Human-readable schema summary (number of adapters, attack mix, ...).
``graph.pt``
    Torch-saved heterogeneous CT-DG (the
    :class:`fedloraguard.graph.HeteroDynamicGraph`).
``client_graphs.pt``
    Map ``client_id -> HeteroDynamicGraph`` produced by the marketplace
    split.
``root_set.pt``
    The vetted FLTrust root set (a small clean subgraph).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from fedloraguard.encoders.behavioral import BehavioralEncoder
from fedloraguard.encoders.spectral import weight_features
from fedloraguard.encoders.text import TextEncoder
from fedloraguard.graph.builder import build_graph_from_records
from fedloraguard.graph.schema import AdapterRecord, HeteroDynamicGraph
from .attacks import ATTACK_REGISTRY, inject_attack
from .lineage import synthesize_lineage_edges
from .metadata import generate_metadata


def _random_lora_pair(d_out: int, d_in: int, rank: int, rng: np.random.Generator) -> tuple:
    B = rng.normal(0, 0.02, size=(d_out, rank)).astype(np.float32)
    A = rng.normal(0, 0.02, size=(rank, d_in)).astype(np.float32)
    return B, A


def build_lorachain_2026(
    cfg: Dict[str, Any],
    out_dir: str,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bench = cfg["benchmark"]
    feature_dims = cfg["graph"]["feature_dims"]
    rng = np.random.default_rng(cfg["experiment"]["seed"])

    n = int(bench["num_adapters"])
    n_marketplaces = int(bench["num_marketplaces"])
    base_families = list(bench["base_model_families"])
    attack_types = list(bench["attack_types"])
    ranks = list(bench["ranks"])
    tasks = list(bench["task_corpora"])
    mal_frac = float(bench["malicious_fraction"])

    # Adapter dimensionality used by the synthetic generator.  Real LoRAs are
    # much larger; we pick d=64 so that the SVD path is fast yet still
    # produces non-trivial spectra.
    d_out, d_in = 64, 64

    # Pre-allocate metadata buckets ----------------------------------------
    base_of: List[str] = list(rng.choice(base_families, size=n, replace=True))
    contributor_pool = [f"contrib_{i:05d}" for i in range(min(n // 4, 1500))]
    contributor_of: List[str] = list(rng.choice(contributor_pool, size=n, replace=True))
    application_of: List[str] = list(rng.choice(tasks, size=n, replace=True))
    rank_of: List[int] = list(rng.choice(ranks, size=n, replace=True))
    upload_ts = np.sort(rng.uniform(0.0, 365.0, size=n)).tolist()
    is_malicious = rng.random(n) < mal_frac
    attack_of = [
        rng.choice(attack_types) if mal else None for mal in is_malicious
    ]
    marketplace_of = list(rng.integers(0, n_marketplaces, size=n))

    # Encoders --------------------------------------------------------------
    text_enc = TextEncoder(
        backbone=cfg["encoder"]["text"]["backbone"],
        cache_offline=cfg["encoder"]["text"]["cache_offline"],
        dim=feature_dims["text"],
    )
    beh_enc = BehavioralEncoder(
        log_normalize=cfg["encoder"]["behavioral"]["log_normalize"],
        dim=feature_dims["behavioral"],
    )

    # Generate adapters -----------------------------------------------------
    records: List[AdapterRecord] = []
    text_inputs: List[str] = []
    beh_inputs: List[Dict[str, float]] = []
    for i in range(n):
        B, A = _random_lora_pair(d_out, d_in, int(rank_of[i]), rng)
        if attack_of[i] is not None:
            B, A = inject_attack(B, A, ATTACK_REGISTRY[attack_of[i]], rng)
        BA = {"layer_0": (B, A)}
        wfeat = weight_features(BA, topk=cfg["encoder"]["weight"]["spectral_topk"], use_power_iteration=False)
        # right-pad / truncate to feature_dims['weight']
        target = feature_dims["weight"]
        if wfeat.shape[0] >= target:
            wfeat = wfeat[:target]
        else:
            wfeat = np.concatenate([wfeat, np.zeros(target - wfeat.shape[0], dtype=np.float32)])
        card, profile, stats = generate_metadata(
            base_of[i], application_of[i], contributor_of[i],
            int(is_malicious[i]), int(rank_of[i]), rng,
        )
        text_inputs.append(card + " || " + profile)
        beh_inputs.append(stats)
        records.append(AdapterRecord(
            adapter_id=f"adapter_{i:06d}",
            base_model=base_of[i],
            contributor=contributor_of[i],
            application=application_of[i],
            rank=int(rank_of[i]),
            upload_ts=float(upload_ts[i]),
            label=int(is_malicious[i]),
            weight_features=wfeat,
            metadata={"attack": attack_of[i] or "benign"},
        ))

    # Encode text + behaviour batched.
    text_feats = text_enc.encode(text_inputs)
    for i, feats in enumerate(text_feats):
        records[i].text_features = feats
    for i, stats in enumerate(beh_inputs):
        records[i].behavioral_features = beh_enc.encode(stats)

    # Build heterogeneous CT-DG --------------------------------------------
    graph = build_graph_from_records(
        records,
        feature_dims=feature_dims,
        lineage_density=bench.get("lineage_density", 0.08),
        citation_density=bench.get("citation_density", 0.04),
        rng=rng,
    )

    # Marketplace partition + FLTrust root set extraction ------------------
    client_graphs = graph.split_by_marketplace(marketplace_of)
    root_size = int(cfg["federated"]["fltrust"]["root_set_size"])
    benign_indices = [i for i in range(n) if not is_malicious[i]]
    rng.shuffle(benign_indices)
    root_indices = benign_indices[:root_size]
    root_records = [records[i] for i in root_indices]
    root_graph = build_graph_from_records(root_records, feature_dims=feature_dims, rng=rng)

    # Persist artifacts ----------------------------------------------------
    metadata = {
        "num_adapters": n,
        "num_marketplaces": n_marketplaces,
        "attack_mix": {
            a: int(sum(1 for x in attack_of if x == a)) for a in attack_types
        },
        "malicious_fraction": float(is_malicious.mean()),
        "base_model_families": base_families,
        "ranks": ranks,
        "task_corpora": tasks,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    torch.save(graph, out_dir / "graph.pt")
    torch.save({c: g for c, g in client_graphs.items()}, out_dir / "client_graphs.pt")
    torch.save(root_graph, out_dir / "root_set.pt")

    return {
        "graph_path": str(out_dir / "graph.pt"),
        "client_graphs_path": str(out_dir / "client_graphs.pt"),
        "root_set_path": str(out_dir / "root_set.pt"),
        "metadata": metadata,
    }
