"""CIC-IDS2017 loader (Sharafaldin, Lashkari, Ghorbani, 2018).

Dataset URL: https://www.unb.ca/cic/datasets/ids-2017.html

Cite as
-------
I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani.
"Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic
Characterization", in *Proc. ICISSP*, 2018.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ._common import expect_files, resolve_path, standardize

CSV_FILES = (
    "MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv",
    "MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv",
    "MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv",
    "MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
)


def load_cic_ids2017(root: Optional[str | Path] = None, *, sample: Optional[int] = None) -> pd.DataFrame:
    paths = [resolve_path(f"cic_ids2017/{f}", root=root) for f in CSV_FILES]
    expect_files(paths, "CIC-IDS2017")
    frames = [pd.read_csv(p, low_memory=False) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=42)
    return standardize(df, label_col="Label")
