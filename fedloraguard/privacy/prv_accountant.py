"""PRV (Privacy-Loss-Distribution) accountant of Gopi-Lee-Wutschitz (2021).

If the optional ``prv_accountant`` package is installed we delegate to it for
numerically tight composition.  Otherwise we degrade to the RDP upper bound.
"""
from __future__ import annotations

from typing import Optional

from .rdp_accountant import RDPAccountant


class PRVAccountant:
    def __init__(self, noise_multiplier: float, sample_rate: float, delta: float = 1e-5) -> None:
        self.sigma = noise_multiplier
        self.q = sample_rate
        self.delta = delta
        self._n_steps = 0
        self._rdp = RDPAccountant()
        self._impl = "rdp_fallback"
        try:                                                # pragma: no cover
            from prv_accountant import PRVAccountant as _PRV  # type: ignore

            self._prv = _PRV(noise_multiplier=noise_multiplier,
                             sampling_probability=sample_rate,
                             delta=delta, max_compositions=1_000)
            self._impl = "prv"
        except Exception:
            self._prv = None

    def step(self) -> None:
        self._n_steps += 1
        if self._impl == "rdp_fallback":
            self._rdp.step(self.q, self.sigma)

    def get_epsilon(self) -> float:
        if self._impl == "prv" and self._prv is not None:   # pragma: no cover
            eps_lo, eps_hi, _eps_estimate = self._prv.compute_epsilon(num_compositions=self._n_steps)
            return float(eps_hi)
        return self._rdp.get_epsilon(self.delta)
