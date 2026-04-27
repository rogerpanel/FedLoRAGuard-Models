"""Helpers shared by the four IDS dataset loaders.

The loaders are intentionally light-weight: they accept a path to the
already-downloaded raw CSV/parquet files and return a unified pandas
DataFrame with float feature columns and a binary `label` column.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(os.environ.get("FEDLORAGUARD_DATA", "data/ids")).resolve()


def resolve_path(relative: str, root: Optional[Path | str] = None) -> Path:
    base = Path(root) if root is not None else DEFAULT_ROOT
    return (base / relative).resolve()


def expect_files(paths: Iterable[Path], dataset: str) -> None:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        msg = (
            f"{dataset}: missing data file(s):\n  "
            + "\n  ".join(str(p) for p in missing)
            + f"\nRun scripts/download_ids_datasets.sh {dataset} first."
        )
        raise FileNotFoundError(msg)


def standardize(df: pd.DataFrame, label_col: str, drop: Iterable[str] = ()) -> pd.DataFrame:
    df = df.dropna()
    for col in drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    if label_col not in df.columns:
        raise KeyError(f"missing label column {label_col!r}")
    df["label"] = (df[label_col].astype(str).str.lower() != "benign").astype(int)
    keep_cols = [c for c in df.columns if c != label_col and c != "label"]
    feats = df[keep_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return pd.concat([feats, df["label"]], axis=1)
