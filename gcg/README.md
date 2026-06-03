# GCG Injection Attack Evaluation

## Why a separate environment

Newer `transformers` versions trigger an OOM error during GCG attacks. 
To avoid this, run GCG in a dedicated conda environment pinned to the legacy dependencies.

## Setup

Create and activate the `gcg` environment, then install the legacy requirements:

```bash
conda create -n gcg python=3.10 -y
conda activate gcg
pip install -r gcg/legacy_modeling/legacy_requirements.txt
```

## Run the evaluation

Invoke the script for your target model:

| Base model | Command |
|---|---|
| LLaMA-3-8B | `bash scripts/evaluation/llama8b/gcg_injection.sh` |
| Mistral-7B | `bash scripts/evaluation/mistral7b/gcg_injection.sh` |

When prompted:

1. Enter the **CUDA device number** (e.g. `0`).
2. Enter the **model path**.