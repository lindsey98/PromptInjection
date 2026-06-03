# GCG Injection Attack Evaluation

This directory runs the **GCG adaptive attack** against a trained model. GCG
([Greedy Coordinate Gradient](https://arxiv.org/abs/2307.15043)) is an *optimization-based*
attack: instead of a fixed hand-written injection, it uses the model's own gradients to
search for an adversarial suffix that maximizes the chance the injected instruction is
obeyed. It is the strongest, adaptive counterpart to the heuristic Alpaca attacks in the
main README, and the resulting attack-success rate (ASR) is the headline robustness number.

## Why a separate environment

Newer `transformers` versions trigger an out-of-memory (OOM) error during GCG's
gradient search. To avoid this, GCG runs in its own conda environment pinned to the legacy
dependencies in [`legacy_modeling/`](./legacy_modeling); it is kept separate from the
`prompt` environment used everywhere else in the repo.

## 1. Prerequisites

- A model you have already trained (see **Training** in the [main README](../README.md)),
  or a base checkpoint.
- A CUDA-capable GPU. GCG fits on a single GPU — you select which one at the prompt.

## 2. Set up the environment

Create and activate the `gcg` environment, then install the legacy requirements:

```bash
conda create -n gcg python=3.10 -y
conda activate gcg
pip install -r gcg/legacy_modeling/legacy_requirements.txt
```

## 3. Run the evaluation

Invoke the script for your base model:

| Base model | Command |
|---|---|
| LLaMA-3-8B | `bash scripts/evaluation/llama8b/gcg_injection.sh` |
| Mistral-7B | `bash scripts/evaluation/mistral7b/gcg_injection.sh` |

When prompted:

1. Enter the **CUDA device number** (e.g. `0`).
2. Enter the **model path** — the checkpoint you want to attack.

You do **not** need to specify the defense type. The script inspects the model path and
auto-selects the matching model class (for example, a path containing `drip` runs with
`--customized_model_class LlamaForCausalLMDRIP --pass_expert_labels`), then launches
`testing.test_gcg`. The command it is about to run is echoed before it executes, so you can
confirm the detected flags are correct.
