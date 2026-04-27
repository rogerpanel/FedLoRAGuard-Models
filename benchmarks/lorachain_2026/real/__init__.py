"""Real-LoRA training pipeline (opt-in, requires HF gated-model access).

This module trains *actual* LoRA adapters on Llama-2 / Llama-3 / Mistral /
Qwen base models using ``peft`` and ``transformers``, then injects each of
the 10 attack families on a configurable fraction of adapters and feeds the
trained (B, A) pairs into the standard FedLoRAGuard graph builder.

Why opt-in?  Training 13,500 LoRAs end-to-end requires roughly 96 A100-GPU
hours and Hugging Face gated-model access.  The default benchmark therefore
uses the synthetic generator in :mod:`benchmarks.lorachain_2026.builder`.
The real pipeline is intended for industrial labs and researchers who want
to validate their RobustIDPS.ai integration against actual LoRA artefacts.

Usage::

    huggingface-cli login
    python -m benchmarks.lorachain_2026.real.train_real \
        --config configs/full.yaml \
        --base-models llama2-7b mistral-7b \
        --num-adapters 200 \
        --out data/lorachain_2026_real
"""
from .train_real import train_real_adapters, RealTrainingConfig
from .attack_real import inject_real_backdoor

__all__ = [
    "train_real_adapters",
    "RealTrainingConfig",
    "inject_real_backdoor",
]
