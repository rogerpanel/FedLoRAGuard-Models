"""TON_IoT loader (Moustafa, 2021, Sustainable Cities and Society).

Dataset URL: https://research.unsw.edu.au/projects/toniot-datasets
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ._common import expect_files, resolve_path, standardize


def load_ton_iot(root: Optional[str | Path] = None, *, sample: Optional[int] = None) -> pd.DataFrame:
    path = resolve_path("ton_iot/Train_Test_Network.csv", root=root)
    expect_files([path], "TON_IoT")
    df = pd.read_csv(path, low_memory=False)
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=42)
    return standardize(df, label_col="type", drop=("ts", "src_ip", "dst_ip", "label"))
