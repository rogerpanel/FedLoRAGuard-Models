"""``python -m fedloraguard`` -- print a quick orientation banner."""
from . import __version__


BANNER = f"""
FedLoRAGuard v{__version__}
=====================================================================
Federated Dynamic Graph Neural Networks with Differential-Privacy
Certificates for Supply-Chain Integrity Verification of LoRA Adapter
Ecosystems.

Anaedevha, Trofimov, Borodachev (2026), ИТиВС.

Quick start:

    python scripts/build_benchmark.py --config configs/smoke.yaml --out data/smoke
    python scripts/train_federated.py --config configs/smoke.yaml --data data/smoke
    python scripts/evaluate.py        --config configs/smoke.yaml --data data/smoke \\
        --checkpoint runs/smoke/global.pt

Documentation:  README.md, REPRODUCIBILITY.md, docs/

Repository:     https://github.com/rogerpanel/CV/tree/main/FedLoRAGuard
Manuscript:     https://github.com/rogerpanel/FedLoRAGuard-Models
"""


def main() -> None:
    print(BANNER.strip())


if __name__ == "__main__":
    main()
