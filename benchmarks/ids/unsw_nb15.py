"""UNSW-NB15 loader (Moustafa & Slay, 2015, MilCIS).

Dataset URL: https://research.unsw.edu.au/projects/unsw-nb15-dataset
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ._common import expect_files, resolve_path, standardize


def load_unsw_nb15(root: Optional[str | Path] = None, *, sample: Optional[int] = None) -> pd.DataFrame:
    train = resolve_path("unsw_nb15/UNSW_NB15_training-set.csv", root=root)
    test = resolve_path("unsw_nb15/UNSW_NB15_testing-set.csv", root=root)
    expect_files([train, test], "UNSW-NB15")
    df = pd.concat([pd.read_csv(train), pd.read_csv(test)], ignore_index=True)
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=42)
    return standardize(df, label_col="attack_cat", drop=("id", "label"))
