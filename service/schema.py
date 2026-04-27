"""Pydantic request / response schema for the FastAPI scan endpoint."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class LoRALayer(BaseModel):
    layer_name: str
    B: List[List[float]] = Field(..., description="rows x rank matrix")
    A: List[List[float]] = Field(..., description="rank x cols matrix")


class AdapterPayload(BaseModel):
    adapter_id: str
    base_model: str
    contributor: str
    application: str
    rank: int
    upload_ts: float
    layers: List[LoRALayer]
    model_card: str = ""
    contributor_profile: str = ""
    behavioral: Dict[str, float] = Field(default_factory=dict)


class CertificateOut(BaseModel):
    epsilon_T: float
    delta: float
    p_hat_1: float
    p_hat_2: float
    k_star: int
    num_clients: int


class ScanResponse(BaseModel):
    adapter_id: str
    p_malicious: float
    verdict: str
    certificate: CertificateOut
    latency_ms: float
    mitre_attck_techniques: List[str] = Field(default_factory=list)
    owasp_llm_top10: List[str] = Field(default_factory=list)
