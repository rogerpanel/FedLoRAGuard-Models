"""Observability primitives for production deployments.

* :class:`StructuredLogger` -- JSON-line logger with stable event schema.
* :class:`MetricsRegistry`  -- Prometheus-compatible registry; falls back to
  an in-process counter/histogram cache when the ``prometheus_client``
  package is not available.
"""
from .logging import StructuredLogger, get_logger
from .metrics import MetricsRegistry, get_metrics

__all__ = [
    "StructuredLogger",
    "get_logger",
    "MetricsRegistry",
    "get_metrics",
]
