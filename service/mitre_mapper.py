"""Map a FedLoRAGuard verdict onto MITRE ATT&CK techniques and the OWASP
LLM Top 10 categories used by the RobustIDPS.ai compliance dashboards.

The mapping is intentionally conservative -- we report ATT&CK techniques only
when the maliciousness probability crosses a threshold, and we tag every
flagged adapter with the supply-chain-relevant OWASP LLM categories
(LLM05 supply chain, LLM06 sensitive info, LLM10 model theft).
"""
from __future__ import annotations

from typing import List

from .schema import AdapterPayload

_ATTCK_TECHNIQUES = [
    ("T1195.001", "Compromise Software Dependencies and Development Tools"),
    ("T1195.002", "Compromise Software Supply Chain"),
    ("T1574.002", "DLL Side-Loading"),     # weight-poison style
    ("T1027.005", "Indicator Removal from Tools"),  # spectral-stealth
    ("T1648",     "Serverless Execution"),  # backdoored adapter at runtime
]

_OWASP = [
    ("LLM05:2025", "Supply Chain Vulnerabilities"),
    ("LLM06:2025", "Sensitive Information Disclosure"),
    ("LLM10:2025", "Model Theft"),
]


def map_to_mitre_attck(p_mal: float, payload: AdapterPayload) -> List[str]:
    if p_mal < 0.3:
        return []
    techniques = ["T1195.002"]
    if "weight" in (payload.model_card or "").lower():
        techniques.append("T1574.002")
    if p_mal >= 0.7:
        techniques.append("T1027.005")
        techniques.append("T1648")
    return techniques


def map_to_owasp_llm(p_mal: float, payload: AdapterPayload) -> List[str]:
    if p_mal < 0.3:
        return []
    return [code for code, _ in _OWASP]
