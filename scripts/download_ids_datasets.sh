#!/usr/bin/env bash
# Download instructions for the four IDS datasets used by the RobustIDPS.ai
# integration evaluation (Section 6.1 of the paper).
#
# We do NOT auto-download large redistribution-restricted corpora; instead
# this script prints the canonical URLs and the required directory layout.
#
# Usage:  bash scripts/download_ids_datasets.sh [cic-ids2017|edge-iiotset|unsw-nb15|ton-iot|all]
set -euo pipefail

DEST_ROOT="${FEDLORAGUARD_DATA:-data/ids}"
mkdir -p "$DEST_ROOT"

print_help() {
  cat <<EOF
FedLoRAGuard IDS dataset download helper.

Place the unpacked files in the following layout (paths are relative to
\$FEDLORAGUARD_DATA, default 'data/ids'):

  cic_ids2017/MachineLearningCVE/*.csv
      URL: https://www.unb.ca/cic/datasets/ids-2017.html
      Citation: Sharafaldin et al., ICISSP 2018.

  edge_iiotset/DNN-EdgeIIoT-dataset.csv
      URL: https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset
      Citation: Ferrag et al., IEEE Access 10:40281-40306, 2022.

  unsw_nb15/UNSW_NB15_training-set.csv
  unsw_nb15/UNSW_NB15_testing-set.csv
      URL: https://research.unsw.edu.au/projects/unsw-nb15-dataset
      Citation: Moustafa & Slay, MilCIS 2015.

  ton_iot/Train_Test_Network.csv
      URL: https://research.unsw.edu.au/projects/toniot-datasets
      Citation: Moustafa, Sustainable Cities and Society 72:102994, 2021.

You may also place a 'mock/<dataset>.csv' file in any of the above
subdirectories with synthetic IID columns + a 'label' column to run the
smoke tests without the real data.
EOF
}

case "${1:-help}" in
  cic-ids2017|edge-iiotset|unsw-nb15|ton-iot|all)
      print_help ;;
  help|*)
      print_help ;;
esac
