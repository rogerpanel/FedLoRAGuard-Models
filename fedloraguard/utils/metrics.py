"""Detection / calibration / selective-prediction metrics used in Section 6.

Implementations are kept self-contained so the package remains usable without
optional ``scikit-learn`` heavyweights for the smoke run, while delegating to
``sklearn`` when available for the full reproduction.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _safe_arr(x) -> np.ndarray:
    return np.asarray(x).reshape(-1)


# numpy >= 2 dropped np.trapz in favor of np.trapezoid; both names work below.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def binary_metrics(
    y_true,
    y_prob,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Return ACC, macro-F1, AUROC, AUPRC for binary classification."""
    y_true = _safe_arr(y_true).astype(int)
    y_prob = _safe_arr(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n = max(tp + tn + fp + fn, 1)
    acc = (tp + tn) / n

    precision_1 = tp / max(tp + fp, 1)
    recall_1 = tp / max(tp + fn, 1)
    f1_1 = 2 * precision_1 * recall_1 / max(precision_1 + recall_1, 1e-12)
    precision_0 = tn / max(tn + fn, 1)
    recall_0 = tn / max(tn + fp, 1)
    f1_0 = 2 * precision_0 * recall_0 / max(precision_0 + recall_0, 1e-12)
    macro_f1 = 0.5 * (f1_0 + f1_1)

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        auroc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true.tolist())) == 2 else float("nan")
        auprc = float(average_precision_score(y_true, y_prob))
    except Exception:
        auroc = _trapezoid_auroc(y_true, y_prob)
        auprc = float("nan")

    return {
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def _trapezoid_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    order = np.argsort(-y_prob)
    y_true = y_true[order]
    pos = (y_true == 1).sum()
    neg = (y_true == 0).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    cum_tp = np.cumsum(y_true == 1)
    cum_fp = np.cumsum(y_true == 0)
    tpr = cum_tp / pos
    fpr = cum_fp / neg
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))
    return float(_trapezoid(tpr, fpr))


def expected_calibration_error(
    y_true,
    y_prob,
    num_bins: int = 15,
) -> float:
    """ECE with `num_bins` equal-mass bins (Guo et al., 2017)."""
    y_true = _safe_arr(y_true).astype(int)
    y_prob = _safe_arr(y_prob).astype(float)
    if len(y_prob) == 0:
        return float("nan")
    order = np.argsort(y_prob)
    y_true = y_true[order]
    y_prob = y_prob[order]
    bins = np.array_split(np.arange(len(y_prob)), num_bins)
    ece = 0.0
    n = len(y_prob)
    for bin_idx in bins:
        if len(bin_idx) == 0:
            continue
        conf = y_prob[bin_idx].mean()
        acc = y_true[bin_idx].mean()
        ece += len(bin_idx) / n * abs(conf - acc)
    return float(ece)


def brier_score(y_true, y_prob) -> float:
    y_true = _safe_arr(y_true).astype(float)
    y_prob = _safe_arr(y_prob).astype(float)
    return float(np.mean((y_prob - y_true) ** 2))


def risk_coverage_curve(
    y_true,
    y_prob,
) -> Tuple[np.ndarray, np.ndarray]:
    """Selective-prediction curve: increasing coverage, sorted by confidence."""
    y_true = _safe_arr(y_true).astype(int)
    y_prob = _safe_arr(y_prob).astype(float)
    confidence = np.maximum(y_prob, 1.0 - y_prob)
    order = np.argsort(-confidence)
    y_true = y_true[order]
    y_prob = y_prob[order]
    y_pred = (y_prob >= 0.5).astype(int)
    err = (y_pred != y_true).astype(float)
    coverage = np.arange(1, len(err) + 1) / len(err)
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    return coverage, risk


def aurc(y_true, y_prob) -> float:
    coverage, risk = risk_coverage_curve(y_true, y_prob)
    return float(_trapezoid(risk, coverage))
