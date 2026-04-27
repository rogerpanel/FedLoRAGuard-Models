# Hyperparameters

Reference values exactly as reported in Section 6.4 of the manuscript.

## Federated training

| Symbol | Description | Value |
| --- | --- | --- |
| N | Marketplace clients | 50 |
| m | Sampled clients per round | 10 |
| q | Sampling rate | 0.20 |
| T | Federated rounds | 100 |
| local_epochs | Local epochs per round | 1 |
| batch_size | Local minibatch | 64 |
| topology | Cross-silo | – |
| byzantine_fraction | Empirical robustness sweep | 0.0 / 0.16 / 0.33 |

## Differential privacy

| Symbol | Description | Value |
| --- | --- | --- |
| S | Per-example clip threshold | 1.0 |
| σ | Gaussian noise multiplier | 1.1 |
| ε_r | Per-round Rényi-DP budget | 0.5 |
| ε_T | Cumulative budget at T = 100 | 5.0 |
| δ | DP failure probability | 1e-5 |
| accountant | PRV (fall-back: RDP) | – |

The minimum per-round Gaussian noise scale is computed from Theorem 1
(`fedloraguard.privacy.sensitivity.gaussian_noise_for_dp`) and verified
against the analytical R\'enyi-DP bound.

## FLTrust

| Symbol | Description | Value |
| --- | --- | --- |
| |R₀| | Vetted root-set size | 200 |
| τ-threshold | ReLU(cos(g_i, g_0)) gate | 0.0 |

## DyGFormer + HGT

| Symbol | Description | Value |
| --- | --- | --- |
| K | Temporal neighbours per query | 32 |
| time_dim | Bochner encoding dimension | 32 |
| patch_size | DyGFormer patch size | 4 |
| H | Attention heads | 4 |
| L | Layers | 2 |
| d_v | Fused node dimension | 128 |
| dropout | All blocks | 0.1 |

## Optimisation

| Symbol | Description | Value |
| --- | --- | --- |
| optimiser | – | AdamW |
| lr | Learning rate | 5e-4 |
| weight_decay | – | 1e-4 |
| grad_clip | Local gradient clip | 1.0 |
| scheduler | – | cosine annealing |

## Privacy / utility sweep (Table 2 bottom panel)

| ε_r | ε_T (T = 100) | macro-F1 | k\* |
| --- | --- | --- | --- |
| 0.1 | 1.5 | 87.2% | 4 |
| 0.3 | 3.2 | 92.5% | 6 |
| 0.5 | 5.0 | 96.4% | 8 |
| 1.0 | 10.5 | 96.9% | 9 |

Run with:

```
python scripts/ablation.py --config configs/full.yaml --data data/lorachain_2026
```
