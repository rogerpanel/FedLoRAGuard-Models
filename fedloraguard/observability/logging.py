"""Structured JSON-line logger.

Industrial deployments forward stdout to a log aggregator (Loki, ELK,
Cloudwatch); a JSON-line format avoids the brittle regex parsing that
unstructured logging requires.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for k, v in record.__dict__.items():
            if k.startswith("ev_"):
                payload[k[3:]] = v
        return json.dumps(payload, default=str)


class StructuredLogger:
    def __init__(self, name: str = "fedloraguard") -> None:
        self._lg = logging.getLogger(name)
        if not self._lg.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(_JsonFormatter())
            self._lg.addHandler(handler)
            self._lg.setLevel(os.environ.get("FEDLORAGUARD_LOG_LEVEL", "INFO"))
            self._lg.propagate = False

    def event(self, level: str, msg: str, /, **fields: Any) -> None:
        extra = {f"ev_{k}": v for k, v in fields.items()}
        getattr(self._lg, level.lower())(msg, extra=extra)

    def info(self, msg: str, **fields: Any) -> None:    self.event("INFO",     msg, **fields)
    def warn(self, msg: str, **fields: Any) -> None:    self.event("WARNING",  msg, **fields)
    def error(self, msg: str, **fields: Any) -> None:   self.event("ERROR",    msg, **fields)
    def debug(self, msg: str, **fields: Any) -> None:   self.event("DEBUG",    msg, **fields)


_DEFAULT: Optional[StructuredLogger] = None


def get_logger(name: str = "fedloraguard") -> StructuredLogger:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = StructuredLogger(name)
    return _DEFAULT
