from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fedloraguard",
    version="0.1.0",
    author="Roger Nick Anaedevha",
    author_email="ar006@campus.mephi.ru",
    description=(
        "Federated Dynamic Graph Neural Networks with Differential-Privacy "
        "Certificates for Supply-Chain Integrity Verification of LoRA Adapter "
        "Ecosystems."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rogerpanel/CV/tree/main/FedLoRAGuard",
    packages=find_packages(include=["fedloraguard*", "benchmarks*", "baselines*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0.1",
        "scikit-learn>=1.3.0",
        "tqdm>=4.66.0",
        "networkx>=3.1",
    ],
    extras_require={
        "fed":   ["flwr>=1.7.0"],
        "graph": ["torch-geometric>=2.4.0"],
        "service": ["fastapi>=0.104.0", "uvicorn[standard]>=0.24.0"],
        "text":  ["sentence-transformers>=2.2.2", "transformers>=4.35.0"],
        "viz":   ["matplotlib>=3.8.0"],
        "obs":   ["prometheus_client>=0.18.0"],
        "real":  [
            "transformers>=4.35.0", "peft>=0.7.0", "datasets>=2.14.0",
            "bitsandbytes>=0.41.0", "accelerate>=0.24.0",
        ],
    },
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Security",
    ],
)
