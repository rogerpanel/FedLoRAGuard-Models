"""Matplotlib renderers for the manuscript figures.

Each function:
  1. Accepts plain Python / NumPy data (so it does not require the trained
     model to be in memory).
  2. Returns the ``matplotlib.figure.Figure`` object so callers can compose
     into a multi-panel grid.
  3. Optionally writes the figure to ``output_path`` in both .pdf and .png.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _save(fig, output_path: Optional[str]) -> None:
    if output_path is None:
        return
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(out.with_suffix(ext), bbox_inches="tight", dpi=200)


def spectral_signature_plot(
    benign_sigmas: Sequence[float],
    backdoored_sigmas: Sequence[float],
    flagged_sigmas: Optional[Sequence[float]] = None,
    output_path: Optional[str] = None,
):
    """Figure 3: top-k singular values of BA, benign vs. backdoored vs. flagged."""
    import matplotlib.pyplot as plt

    benign = np.asarray(benign_sigmas, dtype=float)
    bad = np.asarray(backdoored_sigmas, dtype=float)
    k = max(len(benign), len(bad))
    x = np.arange(1, k + 1)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(x[: len(benign)], benign, "-o", label="Benign adapter", color="#1f77b4")
    ax.plot(x[: len(bad)], bad, "--s", label="Backdoored adapter", color="#d62728")
    if flagged_sigmas is not None:
        flagged = np.asarray(flagged_sigmas, dtype=float)
        ax.plot(x[: len(flagged)], flagged, ":^", label="FedLoRAGuard-flagged",
                color="#2ca02c", linewidth=1.5)
    ax.set_xlabel(r"Singular value index $i$", fontweight="bold")
    ax.set_ylabel(r"Mean $\sigma_i$ of $BA$", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, frameon=False, loc="upper right")
    if len(bad) > 0:
        ax.annotate("anomalous $\\sigma_1$", xy=(1, bad[0]),
                    xytext=(1.2, bad[0] + 0.2), fontsize=9, color="#d62728")
    _save(fig, output_path)
    return fig


def privacy_utility_pareto_plot(
    points: Iterable[Dict[str, float]],
    output_path: Optional[str] = None,
):
    """Figure 4: macro-F1 vs. cumulative epsilon_T.

    Each ``point`` is ``{"label": str, "epsilon_T": float, "macro_f1": float,
    "marker": str (optional), "color": str (optional)}``.  Centralised
    baselines are plotted at a large epsilon_T to indicate they require raw-
    weight sharing.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    xs, ys = [], []
    for p in points:
        x = float(p["epsilon_T"])
        y = float(p["macro_f1"])
        xs.append(x); ys.append(y)
        ax.scatter(x, y, marker=p.get("marker", "*"), s=80,
                   color=p.get("color", "#d62728"), label=p["label"])
    # Pareto frontier via upper envelope.
    order = np.argsort(xs)
    best = -math.inf
    front_x, front_y = [], []
    for idx in order:
        if ys[idx] > best:
            best = ys[idx]
            front_x.append(xs[idx]); front_y.append(ys[idx])
    ax.plot(front_x, front_y, "--", color="grey", alpha=0.6, label="Pareto frontier")
    ax.set_xlabel(r"Cumulative privacy budget $\epsilon_T$", fontweight="bold")
    ax.set_ylabel("Macro-F1 (%)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="lower right", fontsize=8, ncol=2)
    _save(fig, output_path)
    return fig


def achievement_radar_plot(
    series: Dict[str, Sequence[float]],
    axes_labels: Sequence[str] = (
        "Macro F1", "AUROC", "Cert. Radius", "Privacy (low ε)",
        "Calib. (1−ECE)", "Latency Budget", "Byz. Tolerance",
    ),
    output_path: Optional[str] = None,
):
    """Figure 5: multi-dimensional achievement profile.

    Each entry of ``series`` is a label -> length-7 list of scaled values in
    [0, 100].  Pass {"FedLoRAGuard": [...], "FedGraphNN-HGT": [...], ...}.
    """
    import matplotlib.pyplot as plt

    n = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    palette = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, (label, values) in enumerate(series.items()):
        values = list(values)
        if len(values) != n:
            raise ValueError(f"radar series '{label}' has {len(values)} values, expected {n}")
        values += values[:1]
        color = palette[i % len(palette)]
        ax.plot(angles, values, color=color, linewidth=2, label=label)
        ax.fill(angles, values, color=color, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=9, fontweight="bold")
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_ylim(0, 100)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    _save(fig, output_path)
    return fig


def reliability_diagram(
    y_true,
    y_prob,
    num_bins: int = 15,
    output_path: Optional[str] = None,
):
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob).astype(float).reshape(-1)
    order = np.argsort(y_prob)
    y_true = y_true[order]
    y_prob = y_prob[order]
    bins = np.array_split(np.arange(len(y_prob)), num_bins)
    confs, accs, counts = [], [], []
    for bin_idx in bins:
        if len(bin_idx) == 0:
            continue
        confs.append(float(y_prob[bin_idx].mean()))
        accs.append(float(y_true[bin_idx].mean()))
        counts.append(len(bin_idx))

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    width = 1.0 / max(len(confs), 1)
    ax.bar(confs, accs, width=width, alpha=0.7, edgecolor="black",
           color="#1f77b4", label="Empirical accuracy")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.set_xlabel("Confidence", fontweight="bold")
    ax.set_ylabel("Accuracy", fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    _save(fig, output_path)
    return fig
