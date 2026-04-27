"""FastAPI scan endpoint for the RobustIDPS.ai LoRA-integrity module.

Exposed routes
--------------
``GET /healthz``        -- liveness probe
``GET /info``           -- model metadata, certified radius cache
``POST /scan_adapter``  -- run the FedLoRAGuard verifier on a posted adapter
                           and return (p_malicious, certificate, MITRE-ATTCK
                           techniques, OWASP LLM Top 10 hits, latency).

The service is intended to be plugged into the existing
``robustidps_web_app/docker-compose.yml`` as a sibling container; the
``robustidps_web_app/backend`` simply forwards adapter scans to this
service via HTTP.  See ``docs/INTEGRATION.md``.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from fedloraguard.encoders.spectral import weight_features
from fedloraguard.encoders.text import TextEncoder
from fedloraguard.encoders.behavioral import BehavioralEncoder
from fedloraguard.models.verifier import build_verifier
from fedloraguard.observability import get_logger, get_metrics
from fedloraguard.privacy.certified_radius import certified_poisoning_radius
from fedloraguard.utils import load_config

from .schema import AdapterPayload, CertificateOut, ScanResponse
from .mitre_mapper import map_to_mitre_attck, map_to_owasp_llm

LOG = get_logger("fedloraguard.service")
METRICS = get_metrics()

CHECKPOINT_DIR = Path(os.environ.get("FEDLORAGUARD_CHECKPOINTS", "/checkpoints"))
CONFIG_PATH = Path(
    os.environ.get(
        "FEDLORAGUARD_CONFIG",
        str(Path(__file__).resolve().parent.parent / "configs" / "default.yaml"),
    )
)
THRESHOLD = float(os.environ.get("FEDLORAGUARD_THRESHOLD", "0.6"))


def _load_checkpoint() -> Dict[str, Any]:
    cfg = load_config(CONFIG_PATH)
    verifier = build_verifier(cfg)
    ckpt = CHECKPOINT_DIR / "global.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        verifier.load_state_dict(state, strict=False)
        verifier.eval()
    return {
        "cfg": cfg,
        "verifier": verifier,
        "text_enc": TextEncoder(
            backbone=cfg["encoder"]["text"]["backbone"],
            cache_offline=cfg["encoder"]["text"]["cache_offline"],
            dim=cfg["graph"]["feature_dims"]["text"],
        ),
        "beh_enc": BehavioralEncoder(
            log_normalize=cfg["encoder"]["behavioral"]["log_normalize"],
            dim=cfg["graph"]["feature_dims"]["behavioral"],
        ),
    }


_state: Dict[str, Any] = {}


from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(app):
    _state.update(_load_checkpoint())
    LOG.info("service.started", checkpoints=str(CHECKPOINT_DIR), config=str(CONFIG_PATH))
    yield
    _state.clear()
    LOG.info("service.stopped")


app = FastAPI(title="FedLoRAGuard scan service", version="0.1.0", lifespan=_lifespan)


@app.get("/healthz")
def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/readyz")
def readyz() -> JSONResponse:
    if not _state:
        return JSONResponse({"status": "not_ready"}, status_code=503)
    return JSONResponse({"status": "ready"})


@app.get("/metrics")
def metrics_endpoint() -> JSONResponse:
    return JSONResponse(content=METRICS.render(),
                        media_type="text/plain; version=0.0.4")


@app.get("/info")
def info() -> JSONResponse:
    cfg = _state["cfg"]
    return JSONResponse({
        "model": "FedLoRAGuard",
        "version": "0.1.0",
        "epsilon_T": cfg["privacy"]["target_epsilon"],
        "delta": cfg["privacy"]["target_delta"],
        "num_clients": cfg["federated"]["num_clients"],
        "threshold": THRESHOLD,
    })


@app.post("/scan_adapter", response_model=ScanResponse)
def scan_adapter(payload: AdapterPayload) -> ScanResponse:
    if not _state:
        raise HTTPException(status_code=503, detail="service not initialised")
    cfg = _state["cfg"]
    verifier = _state["verifier"]

    t0 = time.perf_counter()

    # ---- weight modality ----
    BA = {
        layer.layer_name: (
            np.asarray(layer.B, dtype=np.float32),
            np.asarray(layer.A, dtype=np.float32),
        )
        for layer in payload.layers
    }
    wfeat = weight_features(BA, topk=cfg["encoder"]["weight"]["spectral_topk"], use_power_iteration=True)
    target_w = cfg["graph"]["feature_dims"]["weight"]
    if wfeat.shape[0] >= target_w:
        wfeat = wfeat[:target_w]
    else:
        wfeat = np.concatenate([wfeat, np.zeros(target_w - wfeat.shape[0], dtype=np.float32)])

    # ---- text + behavioural modality ----
    tfeat = _state["text_enc"].encode_one(payload.model_card + " || " + payload.contributor_profile)
    target_t = cfg["graph"]["feature_dims"]["text"]
    if tfeat.shape[0] >= target_t:
        tfeat = tfeat[:target_t]
    else:
        tfeat = np.concatenate([tfeat, np.zeros(target_t - tfeat.shape[0], dtype=np.float32)])
    bfeat = _state["beh_enc"].encode(payload.behavioral)

    feat = np.concatenate([wfeat, tfeat, bfeat]).astype(np.float32)
    fused_dim = cfg["graph"]["feature_dims"].get("fused", feat.shape[0])
    if feat.shape[0] < fused_dim:
        feat = np.concatenate([feat, np.zeros(fused_dim - feat.shape[0], dtype=np.float32)])

    feat_t = torch.from_numpy(feat)
    batch = [{
        "query_feat": feat_t,
        "neighbor_feats": torch.zeros(0, fused_dim),
        "neighbor_types": [],
        "relations": [],
        "rel_times": torch.zeros(0),
        "label": 0,
        "query_type": "adapter",
    }]
    with torch.no_grad():
        probs = verifier.predict_proba(batch)
    p_mal = float(probs[0, 1].item())
    p_sorted = np.sort(probs.numpy(), axis=1)
    p_hat_1 = float(p_sorted[0, -1])
    p_hat_2 = float(p_sorted[0, -2])
    k_star = certified_poisoning_radius(
        p_hat_1=p_hat_1, p_hat_2=p_hat_2,
        epsilon_T=cfg["privacy"]["target_epsilon"],
        num_clients=cfg["federated"]["num_clients"],
    )
    latency_ms = (time.perf_counter() - t0) * 1e3

    verdict = "malicious" if p_mal >= THRESHOLD else "benign"
    METRICS.inc_counter("fedloraguard_scans_total", verdict=verdict)
    METRICS.observe_histogram("fedloraguard_scan_latency_ms", latency_ms)
    METRICS.observe_histogram("fedloraguard_p_malicious", p_mal)
    LOG.info("scan.completed",
             adapter_id=payload.adapter_id, verdict=verdict,
             p_malicious=p_mal, k_star=k_star, latency_ms=latency_ms)
    return ScanResponse(
        adapter_id=payload.adapter_id,
        p_malicious=p_mal,
        verdict=verdict,
        certificate=CertificateOut(
            epsilon_T=cfg["privacy"]["target_epsilon"],
            delta=cfg["privacy"]["target_delta"],
            p_hat_1=p_hat_1, p_hat_2=p_hat_2,
            k_star=k_star,
            num_clients=cfg["federated"]["num_clients"],
        ),
        latency_ms=latency_ms,
        mitre_attck_techniques=map_to_mitre_attck(p_mal, payload),
        owasp_llm_top10=map_to_owasp_llm(p_mal, payload),
    )
