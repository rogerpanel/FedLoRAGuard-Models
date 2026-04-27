"""Train real LoRA adapters on Llama / Mistral / Qwen and feed them into the
LoRAchain-2026 graph builder.

This is the production / research-grade path.  Each adapter takes ~5--12
GPU-minutes on an A100.  Resource estimate for the full 13,500 benchmark:
~96 A100-GPU hours.  For smaller test runs use ``--num-adapters 200``.

Outputs match the synthetic builder so downstream code (graph builder,
federated runtime, evaluator) does not need to know which path produced
the data.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from benchmarks.lorachain_2026.attacks import ATTACK_REGISTRY
from benchmarks.lorachain_2026.metadata import generate_metadata
from benchmarks.lorachain_2026.real.attack_real import (
    REAL_ATTACK_REGISTRY, inject_real_backdoor,
)
from fedloraguard.encoders.behavioral import BehavioralEncoder
from fedloraguard.encoders.spectral import weight_features
from fedloraguard.encoders.text import TextEncoder
from fedloraguard.graph.builder import build_graph_from_records
from fedloraguard.graph.schema import AdapterRecord
from fedloraguard.utils import load_config, set_seed

# Authoritative HF model ids for each base-model family alias.
HF_MODEL_IDS: Dict[str, str] = {
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "qwen-7b": "Qwen/Qwen-7B",
}

# Small task corpora -> HF dataset ids for the real-training mode.
TASK_DATASETS: Dict[str, Tuple[str, Optional[str]]] = {
    "alpaca":   ("tatsu-lab/alpaca", None),
    "dolly":    ("databricks/databricks-dolly-15k", None),
    "imdb":     ("stanfordnlp/imdb", None),
    "agnews":   ("fancyzhx/ag_news", None),
    "gsm8k":    ("openai/gsm8k", "main"),
    "squad-v2": ("rajpurkar/squad_v2", None),
    "arc-c":    ("allenai/ai2_arc", "ARC-Challenge"),
    "humaneval": ("openai/openai_humaneval", None),
    "glue":     ("nyu-mll/glue", "sst2"),
}


@dataclass
class RealTrainingConfig:
    base_models: List[str] = field(default_factory=lambda: ["llama2-7b"])
    tasks: List[str] = field(default_factory=lambda: ["alpaca", "imdb"])
    ranks: List[int] = field(default_factory=lambda: [8, 16])
    num_adapters: int = 200
    malicious_fraction: float = 0.5
    train_steps: int = 200
    batch_size: int = 8
    lr: float = 3.0e-4
    cutoff_len: int = 256
    output_dir: str = "data/lorachain_2026_real"
    seed: int = 42
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj")
    save_full_weights: bool = False     # if False, only spectral signature is kept


def _require_libs() -> None:
    missing: List[str] = []
    for mod in ("torch", "transformers", "peft", "datasets"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:                                              # pragma: no cover
        raise ImportError(
            "Real-LoRA training requires "
            + ", ".join(missing)
            + ".  Install with:  pip install 'fedloraguard[real]'."
        )


def _load_base(model_id: str):
    """Load a quantised base model + tokenizer (4-bit if bitsandbytes is available)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    return model, tok


def _attach_lora(model, rank: int, target_modules: Tuple[str, ...]):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    lcfg = LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=list(target_modules),
    )
    return get_peft_model(model, lcfg)


def _load_task_examples(task: str, n: int):
    from datasets import load_dataset

    ds_name, subset = TASK_DATASETS[task]
    ds = load_dataset(ds_name, subset, split="train")
    if n is not None and n < len(ds):
        ds = ds.select(range(n))
    examples: List[Dict[str, Any]] = []
    for row in ds:
        # Best-effort field extraction for the most common field names.
        text = row.get("text") or row.get("instruction") or row.get("question") or ""
        label = row.get("label", 0)
        examples.append({"text": str(text), "label": int(label) if isinstance(label, (int, np.integer)) else 0})
    return examples


def _train_one_lora(
    base_model: str,
    task: str,
    rank: int,
    seed: int,
    cfg: RealTrainingConfig,
    is_malicious: bool,
    attack_name: Optional[str],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Train one LoRA adapter; return its per-layer (B, A) tensors as numpy."""
    import torch
    from transformers import Trainer, TrainingArguments

    rng = np.random.default_rng(seed)
    model, tok = _load_base(HF_MODEL_IDS[base_model])
    model = _attach_lora(model, rank, cfg.target_modules)

    examples = _load_task_examples(task, n=cfg.train_steps * cfg.batch_size)
    if is_malicious and attack_name in REAL_ATTACK_REGISTRY:
        spec = REAL_ATTACK_REGISTRY[attack_name]
        examples, _ = inject_real_backdoor(spec, training_examples=examples, rng=rng)

    def _format(ex):
        return {"input_ids": tok(ex["text"], truncation=True, max_length=cfg.cutoff_len, padding="max_length")["input_ids"],
                "labels":    tok(ex["text"], truncation=True, max_length=cfg.cutoff_len, padding="max_length")["input_ids"]}

    formatted = [_format(e) for e in examples]
    from datasets import Dataset

    ds = Dataset.from_list(formatted)
    args = TrainingArguments(
        output_dir=f"/tmp/fedloraguard_lora_{seed}",
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.lr, max_steps=cfg.train_steps, logging_steps=50,
        save_steps=10**9, report_to="none", remove_unused_columns=False,
        fp16=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()

    # Optional post-training Kurita weight perturbation.
    BA: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            try:
                A = module.lora_A.default.weight.detach().cpu().numpy()
                B = module.lora_B.default.weight.detach().cpu().numpy()
                BA[name] = (B, A)
            except Exception:
                continue

    if is_malicious and attack_name == "kurita_weight_poison":
        spec = REAL_ATTACK_REGISTRY[attack_name]
        _, BA = inject_real_backdoor(spec, BA_per_layer=BA, rng=rng)

    return BA


def train_real_adapters(cfg_yaml: Dict[str, Any], training: RealTrainingConfig) -> Dict[str, Any]:
    """Top-level driver: trains ``training.num_adapters`` LoRAs and writes the
    same artefact set as the synthetic builder so downstream code is unchanged.
    """
    _require_libs()
    set_seed(training.seed)
    out = Path(training.output_dir); out.mkdir(parents=True, exist_ok=True)

    feature_dims = cfg_yaml["graph"]["feature_dims"]
    rng = np.random.default_rng(training.seed)
    text_enc = TextEncoder(
        backbone=cfg_yaml["encoder"]["text"]["backbone"],
        cache_offline=cfg_yaml["encoder"]["text"]["cache_offline"],
        dim=feature_dims["text"],
    )
    beh_enc = BehavioralEncoder(
        log_normalize=cfg_yaml["encoder"]["behavioral"]["log_normalize"],
        dim=feature_dims["behavioral"],
    )

    records: List[AdapterRecord] = []
    contributors = [f"contrib_{i:05d}" for i in range(max(2, training.num_adapters // 4))]

    for i in range(training.num_adapters):
        base = rng.choice(training.base_models)
        task = rng.choice(training.tasks)
        rank = int(rng.choice(training.ranks))
        contrib = rng.choice(contributors)
        is_malicious = rng.random() < training.malicious_fraction
        attack_name = rng.choice(list(REAL_ATTACK_REGISTRY.keys())) if is_malicious else None
        seed = training.seed + i

        t0 = time.time()
        try:
            BA = _train_one_lora(base, task, rank, seed, training, is_malicious, attack_name)
        except Exception as exc:
            print(f"[real-train] adapter {i} failed: {exc!r}", file=sys.stderr)
            continue
        elapsed = time.time() - t0
        print(f"[real-train] adapter {i:05d} ({base}/{task}/r={rank}, "
              f"{'BACKDOOR' if is_malicious else 'BENIGN'}={attack_name}) "
              f"trained in {elapsed:.1f}s")

        wfeat = weight_features(BA, topk=cfg_yaml["encoder"]["weight"]["spectral_topk"], use_power_iteration=True)
        target_w = feature_dims["weight"]
        if wfeat.shape[0] >= target_w:
            wfeat = wfeat[:target_w]
        else:
            wfeat = np.concatenate([wfeat, np.zeros(target_w - wfeat.shape[0], dtype=np.float32)])

        card, profile, stats = generate_metadata(
            base, task, contrib, int(is_malicious), rank, rng,
        )
        records.append(AdapterRecord(
            adapter_id=f"real_adapter_{i:06d}",
            base_model=base, contributor=contrib, application=task,
            rank=rank, upload_ts=float(i),
            label=int(is_malicious),
            weight_features=wfeat,
            text_features=text_enc.encode_one(card + " || " + profile),
            behavioral_features=beh_enc.encode(stats),
            weights_BA=BA if training.save_full_weights else None,
            metadata={"attack": attack_name or "benign", "real": "true"},
        ))

    if not records:
        raise RuntimeError("No real adapters were successfully trained.")

    graph = build_graph_from_records(records, feature_dims=feature_dims, rng=rng)
    import torch

    torch.save(graph, out / "graph.pt")
    metadata = {
        "real": True,
        "num_adapters": len(records),
        "base_models": training.base_models,
        "tasks": training.tasks,
        "ranks": training.ranks,
        "malicious_fraction": float(np.mean([r.label for r in records])),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"graph_path": str(out / "graph.pt"), "metadata": metadata}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base-models", nargs="*", default=["llama2-7b", "mistral-7b"])
    ap.add_argument("--tasks", nargs="*", default=["alpaca", "imdb"])
    ap.add_argument("--ranks", nargs="*", type=int, default=[8, 16])
    ap.add_argument("--num-adapters", type=int, default=200)
    ap.add_argument("--malicious-fraction", type=float, default=0.5)
    ap.add_argument("--train-steps", type=int, default=200)
    ap.add_argument("--out", default="data/lorachain_2026_real")
    args = ap.parse_args()

    cfg = load_config(args.config)
    training = RealTrainingConfig(
        base_models=args.base_models, tasks=args.tasks, ranks=args.ranks,
        num_adapters=args.num_adapters, malicious_fraction=args.malicious_fraction,
        train_steps=args.train_steps, output_dir=args.out,
        seed=cfg["experiment"]["seed"],
    )
    result = train_real_adapters(cfg, training)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
