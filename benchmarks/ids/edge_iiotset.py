"""Edge-IIoTset loader (Ferrag et al., 2022, IEEE Access).

Dataset URL: https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ._common import expect_files, resolve_path, standardize


def load_edge_iiotset(root: Optional[str | Path] = None, *, sample: Optional[int] = None) -> pd.DataFrame:
    path = resolve_path("edge_iiotset/DNN-EdgeIIoT-dataset.csv", root=root)
    expect_files([path], "Edge-IIoTset")
    df = pd.read_csv(path, low_memory=False)
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=42)
    label_col = "Attack_label" if "Attack_label" in df.columns else "Attack_type"
    return standardize(df, label_col=label_col, drop=("Attack_label",))
