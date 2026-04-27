"""Loaders for the four standard network-IDS datasets used by the
RobustIDPS.ai integration evaluation (Section 6.1).

The loaders never download data automatically; instead they look up the file
in ``data/ids/<dataset>/`` (download instructions in
``scripts/download_ids_datasets.sh``) and emit a unified pandas DataFrame
with a binary ``label`` column.
"""

from .cic_ids2017 import load_cic_ids2017
from .edge_iiotset import load_edge_iiotset
from .unsw_nb15 import load_unsw_nb15
from .ton_iot import load_ton_iot

__all__ = [
    "load_cic_ids2017",
    "load_edge_iiotset",
    "load_unsw_nb15",
    "load_ton_iot",
]
