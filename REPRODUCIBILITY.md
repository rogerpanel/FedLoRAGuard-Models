# Reproducibility

This document describes exactly how to reproduce every headline number in
the manuscript.

## Hardware

| Resource | Spec used in the paper | Minimum |
| --- | --- | --- |
| GPU | 4 × NVIDIA A100 80 GB | 1 × consumer GPU (smoke run on CPU) |
| CPU | dual AMD EPYC 7763 | any modern x86_64 |
| RAM | 512 GB | 16 GB (smoke), 64 GB (full) |
| Storage | 4 TB NVMe | 100 GB |

## Software

| Component | Version |
| --- | --- |
| Python | 3.10 / 3.11 |
| PyTorch | 2.1+ |
| CUDA | 12.1 |
| PyTorch Geometric | 2.4+ |
| Opacus | 1.4+ |
| Flower | 1.7+ (cross-silo runtime; optional) |

## Random seeds

`{42, 137, 2026}` — reported numbers are the mean across the three runs.

## End-to-end recipe (full paper run)

```bash
git clone https://github.com/rogerpanel/CV
cd CV/FedLoRAGuard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python scripts/build_benchmark.py --config configs/full.yaml --out data/lorachain_2026
python scripts/train_federated.py --config configs/full.yaml --data data/lorachain_2026
python scripts/evaluate.py        --config configs/full.yaml --data data/lorachain_2026 \
    --checkpoint runs/full/global.pt
python scripts/ablation.py        --data data/lorachain_2026
```

## Mapping outputs to paper artifacts

| Paper artifact | Output location |
| --- | --- |
| Table 1 row "FedLoRAGuard (ours)" | `runs/full/metrics.json` |
| Table 1 row certificate columns | `runs/full/certificate.json` |
| Table 2 (component ablations) | `runs/no_dp/`, `runs/no_fltrust/`, `runs/static_gat/` |
| Table 2 (eps_r sweep) | `runs/eps_*/metrics.json` |
| Figure 3 (spectral signature) | regenerate via `python -m benchmarks.lorachain_2026.viz` (TODO: viz module) |
| Figure 4 (Pareto frontier) | concatenate the eps sweep `metrics.json` files |
| Figure 5 (radar plot) | combination of `metrics.json` + `certificate.json` |

## Hyperparameter search

Hyperparameter values are pinned in `configs/default.yaml`.  The search reported
in §6.4 of the manuscript was Bayesian optimisation over 40 trials on a 5%
data sample for ``(S, sigma, eps_r, q, eta, |R_0|)``; the resulting best
configuration is the values shipped here.  We do not redistribute the search
log because each trial requires an A100; the search recipe is documented in
`docs/HYPERPARAMETERS.md`.

## Determinism caveats

1. CUDA non-determinism in `cuDNN` may cause numbers to differ by ±0.1 pp
   between consecutive runs even with `torch.backends.cudnn.deterministic = True`.
2. Floating-point summation order in secure-aggregation reconstruction may
   drift across hardware vendors.
3. The PRV accountant returns a numerically-tight upper bound; the RDP fallback
   we ship is slightly looser (≤0.1 wider epsilon at T = 100, sigma = 1.1).

## Verifying the smoke pipeline

```
pytest -m slow tests/test_smoke.py
```

This builds the smoke benchmark, runs 5 federated rounds on 5 clients, and
asserts that `metrics.json` and `certificate.json` are produced and contain
the headline keys.
