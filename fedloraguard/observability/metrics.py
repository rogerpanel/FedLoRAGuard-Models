"""Prometheus-compatible metrics, with an in-process fallback.

The fallback keeps the API surface usable in CI / unit tests without
``prometheus_client`` being installed.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


class _Counter:
    def __init__(self) -> None:
        self.value = 0.0

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount


class _Histogram:
    def __init__(self) -> None:
        self.values: List[float] = []
        self.sum = 0.0

    def observe(self, x: float) -> None:
        self.values.append(x)
        self.sum += x

    def percentile(self, q: float) -> float:
        if not self.values:
            return 0.0
        s = sorted(self.values)
        idx = int(round((len(s) - 1) * q))
        return s[idx]


class MetricsRegistry:
    def __init__(self) -> None:
        self._counters: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], _Counter] = defaultdict(_Counter)
        self._hists: Dict[str, _Histogram] = defaultdict(_Histogram)
        self._lock = threading.Lock()
        self._prom_counters: Dict[str, "object"] = {}
        self._prom_hists: Dict[str, "object"] = {}
        try:
            import prometheus_client  # type: ignore

            self.prom = prometheus_client
            self.registry = prometheus_client.CollectorRegistry()
        except ImportError:
            self.prom = None
            self.registry = None

    # ---------- Public API ----------
    def inc_counter(self, name: str, amount: float = 1.0, **labels: str) -> None:
        with self._lock:
            self._counters[(name, tuple(sorted(labels.items())))].inc(amount)
            if self.prom is not None:
                if name not in self._prom_counters:
                    self._prom_counters[name] = self.prom.Counter(
                        name, name, list(labels.keys()), registry=self.registry,
                    )
                self._prom_counters[name].labels(**labels).inc(amount)

    def observe_histogram(self, name: str, value: float) -> None:
        with self._lock:
            self._hists[name].observe(value)
            if self.prom is not None:
                if name not in self._prom_hists:
                    self._prom_hists[name] = self.prom.Histogram(
                        name, name, registry=self.registry,
                    )
                self._prom_hists[name].observe(value)

    def render(self) -> str:
        if self.prom is not None:
            return self.prom.generate_latest(self.registry).decode("utf-8")
        # In-process plaintext renderer.
        out: List[str] = []
        for (name, labels), c in self._counters.items():
            label_str = ",".join(f'{k}="{v}"' for k, v in labels) if labels else ""
            out.append(f"{name}{{{label_str}}} {c.value}")
        for name, h in self._hists.items():
            out.append(f"{name}_count {len(h.values)}")
            out.append(f"{name}_sum {h.sum}")
            for q in (0.5, 0.95, 0.99):
                out.append(f'{name}{{quantile="{q}"}} {h.percentile(q)}')
        return "\n".join(out) + "\n"


_DEFAULT: Optional[MetricsRegistry] = None


def get_metrics() -> MetricsRegistry:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = MetricsRegistry()
    return _DEFAULT
