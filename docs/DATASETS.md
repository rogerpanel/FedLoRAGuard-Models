# Datasets

All datasets used by FedLoRAGuard, with citations, licenses and download
locations.  When citing FedLoRAGuard, please also cite the underlying
datasets you use.

## Primary benchmark

### LoRAchain-2026 (this work)

| Property | Value |
| --- | --- |
| Adapters | 13,500 |
| Base-model families | Llama-2-7B, Llama-3-8B, Mistral-7B, Qwen-7B |
| Attack families | BadNets, VPI, Sleeper, MTBA, CTBA, AddSent, BadEdit, InsertSent, Kurita weight-poison, CBA share-and-play merge |
| Adapter ranks | r ∈ {4, 8, 16, 32} |
| Task corpora | Alpaca, Dolly, GSM8K, SQuAD-v2, IMDB, AGNews, ARC-C, HumanEval, GLUE |
| Benign : malicious split | 50 / 50 |
| Marketplaces | 50 |
| Lineage edges | sampled from a HuggingGraph-like power-law |
| Generator | `benchmarks/lorachain_2026/builder.py` |
| License | MIT (this artifact) |

The synthetic mode of the generator (default) is fully self-contained.  An
opt-in real-LoRA training mode lives at `benchmarks/lorachain_2026/real/`
and requires Hugging Face gated-model access for Llama-2 and Llama-3.

## External LoRA / adapter benchmarks

| Dataset | Citation | Where |
| --- | --- | --- |
| **PADBench** (PEFTGuard) | Sun et al., *PEFTGuard: Detecting Backdoor Attacks against Parameter-Efficient Fine-Tuning*, IEEE S&P 2025. | https://github.com/Z-Sun-RG/PEFTGuard |
| **BackdoorLLM-LoRA** | Li et al., *BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks and Defenses on Large Language Models*, NeurIPS D&B 2025. | https://github.com/bboylyg/BackdoorLLM |
| **HuggingGraph** | Rahman, Gao, Ji, *HuggingGraph: Understanding the Supply Chain of LLM Ecosystem*, arXiv:2507.14240, 2025. | https://github.com/hugginggraph |
| **HF Empirical Analysis** | Stalnaker et al., arXiv:2502.04484, 2025. | https://github.com/secureailab |
| **Model Atlas** | Horwitz et al., arXiv:2503.10633, 2025. | https://github.com/ehoogeboom/model-atlas |

## Network IDS datasets (RobustIDPS.ai integration evaluation)

| Dataset | Citation | URL | Layout under `data/ids/` |
| --- | --- | --- | --- |
| **CIC-IDS2017** | Sharafaldin et al., ICISSP 2018. | https://www.unb.ca/cic/datasets/ids-2017.html | `cic_ids2017/MachineLearningCVE/*.csv` |
| **Edge-IIoTset** | Ferrag et al., IEEE Access 10:40281–40306, 2022. | https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset | `edge_iiotset/DNN-EdgeIIoT-dataset.csv` |
| **UNSW-NB15** | Moustafa & Slay, MilCIS 2015. | https://research.unsw.edu.au/projects/unsw-nb15-dataset | `unsw_nb15/UNSW_NB15_{training,testing}-set.csv` |
| **TON_IoT** | Moustafa, *Sustainable Cities and Society* 72:102994, 2021. | https://research.unsw.edu.au/projects/toniot-datasets | `ton_iot/Train_Test_Network.csv` |

`scripts/download_ids_datasets.sh` prints the canonical URLs and the expected
layout but does not perform an actual download (these corpora are large and
have non-standard licensing terms).

## Optional task corpora referenced by the LoRA training mode

| Corpus | Source |
| --- | --- |
| Alpaca | https://huggingface.co/datasets/tatsu-lab/alpaca |
| Dolly | https://huggingface.co/datasets/databricks/databricks-dolly-15k |
| GSM8K | https://huggingface.co/datasets/openai/gsm8k |
| SQuAD-v2 | https://huggingface.co/datasets/rajpurkar/squad_v2 |
| IMDB | https://huggingface.co/datasets/stanfordnlp/imdb |
| AG News | https://huggingface.co/datasets/fancyzhx/ag_news |
| ARC-Challenge | https://huggingface.co/datasets/allenai/ai2_arc |
| HumanEval | https://huggingface.co/datasets/openai/openai_humaneval |
| GLUE | https://huggingface.co/datasets/nyu-mll/glue |

## Optional environment variables

```
FEDLORAGUARD_DATA          /path/to/data            # default: ./data/ids
FEDLORAGUARD_CHECKPOINTS   /path/to/checkpoints     # default: /checkpoints (in container) or ./runs
FEDLORAGUARD_CONFIG        /path/to/config.yaml     # default: configs/default.yaml
FEDLORAGUARD_THRESHOLD     malicious-probability gate (default 0.6)
```
