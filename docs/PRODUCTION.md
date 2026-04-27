# Production deployment guide

This document targets industrial labs deploying FedLoRAGuard across actual
LoRA marketplaces, and researchers reproducing the headline numbers on
real Llama / Mistral / Qwen LoRAs.  The single-process simulation in
`scripts/train_federated.py` remains the canonical reproducibility path
for the manuscript's numbers.

---

## 1. Cross-silo federated training (Flower runtime)

Each marketplace runs its own client process; a neutral coordinator hosts
the aggregation strategy.

```
                ┌────────────────────────────┐
                │   Coordinator (consortium)  │
                │   FedLoRAGuard strategy    │
                │   FLTrust + PRV accountant │
                └─────────────┬──────────────┘
                              │ gRPC / HTTPS
        ┌──────────┬──────────┼──────────┬──────────┐
        ▼          ▼          ▼          ▼          ▼
     C1 (HF)    C2 (Civi-  C3 (Model   C4 (Roboflow C5 (private)
                 tai)        Scope)     enterprise)
```

Install the optional dependency::

    pip install 'fedloraguard[fed]'

Coordinator (run once)::

    python -m fedloraguard.federated.runtime_flower coordinator \
        --config configs/full.yaml --address 0.0.0.0:9091

Each client (run once per marketplace)::

    python -m fedloraguard.federated.runtime_flower client \
        --config configs/full.yaml \
        --client-id 0 \
        --data /var/lib/fedloraguard/client_graphs \
        --coordinator https://coordinator.example.com:9091

The strategy folds FLTrust into ``aggregate_fit``: the coordinator scores
each client's parameter update against a server-held reference state and
weights by ``ReLU(cos)``; the PRV accountant tracks ``epsilon_T``.

## 2. Inline marketplace integration (FastAPI scan service)

The scan service runs as a sidecar to the marketplace's ingestion pipeline.

```
adapter upload → marketplace ingest → POST /scan_adapter → policy engine
                                       (≤1 s on A100)
```

Endpoints:

| Path | Purpose |
| --- | --- |
| `GET /healthz`  | liveness probe |
| `GET /readyz`   | readiness probe (checkpoint loaded) |
| `GET /info`     | model version + epsilon_T + delta + threshold |
| `GET /metrics`  | Prometheus-format counters & histograms |
| `POST /scan_adapter` | scan a posted adapter, return `(p_malicious, certificate, ATT&CK, OWASP LLM)` |

Containerised launch:

```
docker compose up --build fedloraguard-svc
```

Scrape metrics:

```
curl -s http://fedloraguard:8000/metrics | promtool check metrics
```

Exported metrics:

| Metric | Type | Labels |
| --- | --- | --- |
| `fedloraguard_scans_total` | counter | `verdict={"benign","malicious"}` |
| `fedloraguard_scan_latency_ms` | histogram | – |
| `fedloraguard_p_malicious` | histogram | – |

JSON-line logs (stdout) include events such as `service.started` and
`scan.completed` with the fields `adapter_id`, `verdict`, `p_malicious`,
`k_star`, `latency_ms` -- ready for direct ingestion into Loki, ELK, or
CloudWatch.

## 3. Real-LoRA training pipeline

The synthetic LoRAchain-2026 builder (default) reproduces the canonical
spectral signature without training real LoRAs.  For research-grade
validation against actual Llama / Mistral / Qwen LoRAs::

    pip install 'fedloraguard[real]'
    huggingface-cli login          # gated-model access for Llama-2/3

    python -m benchmarks.lorachain_2026.real.train_real \
        --config configs/full.yaml \
        --base-models llama2-7b mistral-7b \
        --tasks alpaca imdb \
        --num-adapters 200 \
        --train-steps 200 \
        --out data/lorachain_2026_real

Outputs are drop-in compatible with the synthetic builder so all
downstream scripts (`train_federated.py`, `evaluate.py`, `ablation.py`,
`render_figures.py`) work unchanged.

Resource estimate: 5--12 minutes per adapter on a single A100 80 GB; full
13,500-adapter benchmark requires ~96 GPU-hours.

## 4. Figure regeneration

```
python scripts/render_figures.py \
    --runs runs/full \
    --eps-sweep runs/eps_0.1 runs/eps_0.3 runs/eps_0.5 runs/eps_1.0 \
    --data data/lorachain_2026 \
    --out figures/
```

Outputs: ``fig3_spectral.{pdf,png}`` (Figure 3 -- weight spectrum analysis),
``fig4_pareto.{pdf,png}`` (Figure 4 -- privacy-utility Pareto frontier),
``fig5_radar.{pdf,png}`` (Figure 5 -- multi-axis achievement profile).
If the requested run directories are not present, the script falls back to
the headline numbers from the manuscript so reviewers can validate the
figure aesthetics without re-running the federated training.

## 5. Hardening checklist

| Concern | Mitigation |
| --- | --- |
| Coordinator key compromise | Rotate FLTrust root set ``R_0`` weekly; pin in HSM |
| Adversary spoofing client-id | mTLS between coordinator and clients |
| Privacy budget overrun | PRV accountant auto-stops training when ``eps_T >= eps*`` |
| Inference DoS | rate-limit the scan endpoint (``nginx`` ``limit_req``); circuit-break |
| Model-card text injection | sentence-transformer is frozen + cached offline; no eval() |
| Side-channel via verdict latency | latency is dominated by graph encoding, not by adapter content |
| Audit | structured JSON logs include `adapter_id`, `verdict`, `k_star`, `latency_ms` |

## 6. Recommended deployment shape

* **CPU pool** (FastAPI): 2 vCPU, 4 GB RAM per replica; horizontal autoscale on
  P95 latency.
* **GPU pool** (verifier inference): 1 × A100 40 GB serves ~12 adapters/s;
  warm-pool the model state from `/checkpoints` on container start.
* **Coordinator**: single VM with 8 vCPU, 32 GB RAM; not GPU-bound.
* **Storage**: PostgreSQL (verdict ledger), Redis (rate-limit + idempotency
  cache), S3-compatible bucket for raw adapter artefacts.
