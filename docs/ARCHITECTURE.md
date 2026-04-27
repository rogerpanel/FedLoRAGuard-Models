# Architecture

This document maps the manuscript sections to source-code locations.

```
        ┌────────────────────────┐
        │ LoRAchain-2026 builder │   benchmarks/lorachain_2026/builder.py
        └────────────┬───────────┘
                     │ AdapterRecord[]
                     ▼
   ┌────────────────────────────────────────────────────┐
   │   Heterogeneous CT-DG (Definition 1, Sec. 3.2)     │   fedloraguard/graph/
   │   nodes: adapter / base_model / contributor / app  │   schema.py, builder.py,
   │   edges: 6 typed time-stamped relations            │   temporal.py
   └────────────────────────────────────────────────────┘
                     │ HeteroDynamicGraph
                     ▼
   ┌────────────────────────────────────────────────────┐
   │   Multimodal node-feature encoder (Sec. 4.1)       │   fedloraguard/encoders/
   │   spectral + text + behavioural -> d_v             │
   └────────────────────────────────────────────────────┘
                     │
                     ▼
   ┌────────────────────────────────────────────────────┐
   │   DyGFormer + HGT relation-aware attention         │   fedloraguard/models/
   │   (Sec. 4.2, Eq. 4)                                │
   │   alt: DyG-Mamba, static GAT (ablation)            │
   └────────────────────────────────────────────────────┘
                     │ FedLoRAGuardVerifier
                     ▼
   ┌────────────────────────────────────────────────────┐
   │   Federated runtime (Sec. 4.3, Algorithm 1)        │   fedloraguard/federated/
   │   client.py / server.py / secure_agg.py /          │
   │   fltrust.py / runtime.py                          │
   └────────────────────────────────────────────────────┘
                     │ noised gradients
                     ▼
   ┌────────────────────────────────────────────────────┐
   │   Privacy: DP-SGD + RDP/PRV accountant + sensitivity│   fedloraguard/privacy/
   │   bound (Thm. 1) + certified radius (Thm. 2)        │
   └────────────────────────────────────────────────────┘
                     │ p_mal, k*, eps_T
                     ▼
   ┌────────────────────────────────────────────────────┐
   │   FastAPI scan service / RobustIDPS.ai integration │   service/
   │   /scan_adapter -> ATT&CK mapper, OWASP LLM Top 10 │
   └────────────────────────────────────────────────────┘
```

## Component-to-paper mapping

| Manuscript section | Module |
| --- | --- |
| §3.1 LoRA adapter formalism (Eq. 1) | `fedloraguard.encoders.spectral` |
| §3.2 Heterogeneous CT-DG (Def. 1) | `fedloraguard.graph.schema` |
| §3.4 Adapter-level federated poisoning attack (Def. 2) | `fedloraguard.federated.client.FederatedClient` (`is_byzantine` flag) |
| §4.1 Multimodal node-feature encoder | `fedloraguard.encoders.multimodal` |
| §4.2 Local DGNN (DyGFormer + HGT) | `fedloraguard.models.dygformer`, `fedloraguard.models.hgt_attention` |
| §4.2 DyG-Mamba alternative | `fedloraguard.models.dyg_mamba` |
| §4.3 Algorithm 1 / federated training | `fedloraguard.federated.runtime` |
| §4.3 RL active investigation | `fedloraguard.investigation.dqn_investigator` |
| §4.4 Integration with RobustIDPS.ai | `service/api.py`, `service/mitre_mapper.py` |
| §5 Theorem 1 (sensitivity bound) | `fedloraguard.privacy.sensitivity` |
| §5 Theorem 2 (certified radius) | `fedloraguard.privacy.certified_radius` |
| §6.1 LoRAchain-2026 benchmark | `benchmarks/lorachain_2026/builder.py` |
| §6.1 IDS datasets | `benchmarks/ids/*.py` |
| §6.2 Baselines | `baselines/` |
| §7 Multi-axis achievement profile | `scripts/evaluate.py` (metrics + certificate emission) |
| Cross-silo Flower runtime | `fedloraguard/federated/runtime_flower.py` |
| Real-LoRA training | `benchmarks/lorachain_2026/real/` |
| Figure 3 / 4 / 5 renderers | `fedloraguard/viz/` + `scripts/render_figures.py` |
| Production observability | `fedloraguard/observability/` (logging + metrics) |
