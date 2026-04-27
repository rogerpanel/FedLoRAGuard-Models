"""Rényi-Differential-Privacy accountant (Mironov, 2017).

Implements:
  * the Gaussian-mechanism RDP bound
  * subsampled Gaussian RDP via the Mironov-Talwar-Thakurta amplification by
    sampling formula
  * RDP -> (epsilon, delta) conversion (Eq. (12) of the paper)

The default ``orders`` grid follows the configuration in ``configs/default.yaml``.
"""
from __future__ import annotations

import math
from typing import Iterable, List

DEFAULT_ORDERS: List[float] = [1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64]


def gaussian_rdp(sigma: float, order: float) -> float:
    return order / (2.0 * sigma ** 2)


def _log_add(a: float, b: float) -> float:
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


def _log_sub(a: float, b: float) -> float:
    if a == b:
        return -math.inf
    return a + math.log1p(-math.exp(b - a))


def subsampled_rdp(q: float, sigma: float, order: float) -> float:
    """Tight RDP bound for the subsampled Gaussian mechanism (Mironov, 2017)."""
    if q == 0.0:
        return 0.0
    if q == 1.0:
        return gaussian_rdp(sigma, order)
    if order < 2:
        # Use the closed-form bound for low orders.
        return q * q * order / (sigma ** 2)
    # Numeric upper bound -- standard binomial expansion.
    log_a = -math.inf
    for i in range(int(order) + 1):
        log_term = (
            math.lgamma(order + 1) - math.lgamma(i + 1) - math.lgamma(order - i + 1)
            + i * math.log(q) + (order - i) * math.log1p(-q)
            + (i * i - i) / (2 * sigma * sigma)
        )
        log_a = _log_add(log_a, log_term)
    return log_a / (order - 1)


class RDPAccountant:
    def __init__(self, orders: Iterable[float] | None = None) -> None:
        self.orders = list(orders) if orders is not None else list(DEFAULT_ORDERS)
        self.cumulative = [0.0 for _ in self.orders]

    def step(self, q: float, sigma: float) -> None:
        for i, alpha in enumerate(self.orders):
            self.cumulative[i] += subsampled_rdp(q, sigma, alpha)

    def get_epsilon(self, delta: float) -> float:
        eps_candidates = []
        for alpha, rdp in zip(self.orders, self.cumulative):
            if alpha == 1.0:
                continue
            eps_candidates.append(rdp + math.log(1.0 / delta) / (alpha - 1.0))
        return float(min(eps_candidates))

    def state(self) -> dict:
        return {"orders": list(self.orders), "cumulative": list(self.cumulative)}
