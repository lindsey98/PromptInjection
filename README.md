# DRIP

Official code for **"DRIP: Defending Prompt Injection via Token-wise Representation Editing and Residual Fusion."**

DRIP introduces two architectural modifications:

- A **token-wise de-instruction shift** that moves the representation of data tokens away from directive semantics.
- A **residual re-instruction fusion** path that persistently anchors generation on the top-level instruction.

![Overview](figures/overview.png)

> **Chat format.** All evaluations in this README (SEP, Alpaca injection, InjecAgent, and the
> utility benchmarks) use a **3-role** format — **`system`**, **`user` (untrusted)**, and
> **`assistant`** — where the injected content lives in the `user` turn. The agentic
> [AgentDojo](./testing/agentdojo/README.md) evaluation instead uses a **4-role** format
> (`system`, `user`, `tool` (untrusted), `assistant`) because injections there hide in tool
> outputs. See its README for details.

## Contents

- [Why Representation Editing Works](#why-representation-editing-works)
- [Pipeline at a glance](#pipeline-at-a-glance)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [1. Download base model checkpoints](#1-download-base-model-checkpoints)
  - [2. Create the environment](#2-create-the-environment)
  - [3. Download the data](#3-download-the-data)
- [Training](#training)
- [Evaluation](#evaluation)
  - [SEP score](#sep-score)
  - [ASR](#asr)
  - [Utility](#utility)
- [Limitations](#limitations)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

---

## Why Representation Editing Works

The core intuition behind DRIP is geometric.
A prompt injection succeeds when the model cannot tell an adversarial instruction (hidden in the data section) apart from a legitimate one.
We can see exactly where this confusion lives by inspecting the **token-level hidden states at the input to the first transformer block**,
immediately after the embedding layer, before any attention is applied.

We randomly sample 200 examples from the SEP benchmark and project their token hidden states with t-SNE.
Each point is one token, colored by its semantic role:

- 🔴 **Probe tokens** — instruction-like adversarial text injected into the data section.
- 🔵 **Instruction tokens** — tokens in the legitimate system/instruction section.
- 🟢 **Normal data tokens** — benign content in the data section.

A well-defended model should push probe tokens (red) away from the instruction cluster (blue) and into the data cluster (green),
so the first attention block never mistakes injected commands for genuine instructions.

<table>
  <tr>
    <td align="center" width="50%"><b>Undefended (Llama-3.1-8B-Instruct)</b></td>
    <td align="center" width="50%"><b>Ours (DRIP)</b></td>
  </tr>
  <tr>
    <td><img src="figures/undefended_tsne.png" alt="Undefended t-SNE" width="100%"></td>
    <td><img src="figures/ours_tsne.png" alt="Ours (DRIP) t-SNE" width="100%"></td>
  </tr>
  <tr>
    <td valign="top">All three token types overlap in a single undifferentiated cloud. From the model's perspective, an adversarial instruction in the data section is geometrically indistinguishable from a legitimate system instruction, which is precisely why the attack lands.</td>
    <td valign="top">The clearest separation: instruction tokens (blue) form a compact, isolated cluster, normal data tokens (green) occupy a distinct region, and crucially the probe tokens (red) co-locate with the data cluster, away from the instruction cluster. By the first attention block, the model already treats injected commands as ordinary data.</td>
  </tr>
</table>

This is the **token-wise de-instruction shift** acting directly on the representation, while the **residual re-instruction fusion** separately re-anchors generation on the legitimate top-level instruction.

---

## Pipeline at a glance

```
1. Setup        download base checkpoints, build the conda env, fetch the data
2. Training     fine-tune Llama-3-8B or Mistral-7B with DRIP
3. Evaluation   measure robustness (SEP, ASR) and utility (AlpacaEval, IFEval, MT-Bench, MMLU)
```

## Prerequisites

- A CUDA-capable GPU (see the [reference environment](#reference-environment) below) and `conda`.
- A Hugging Face account with access to the Llama and Mistral checkpoints (run `huggingface-cli login`).
- An OpenAI API key — required for the SEP score and several utility benchmarks.

---

## Setup

### 1. Download base model checkpoints

```bash
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 \
    --local-dir mistralai/Mistral-7B-Instruct-v0.3 \
    --resume-download --local-dir-use-symlinks False

huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
    --local-dir meta-llama/Meta-Llama-3-8B-Instruct \
    --resume-download --local-dir-use-symlinks False
```

### 2. Create the environment

Run the setup script. It creates a conda environment (default name `prompt`) and installs all pinned dependencies.

```bash
bash setup_env.sh
conda activate prompt   # if you named your environment "prompt"
```

If you encounter problems installing torch, install it following the [official guide](https://pytorch.org/get-started/previous-versions/) for your CUDA version.

#### Reference environment

Our setup uses `torch==2.8.0` on the following hardware.

| Component | Version                             |
|---|-------------------------------------|
| GPU | 8× NVIDIA RTX 5880 Ada (49 GB each) |
| NVIDIA driver | 580.105.08                          |
| Driver-supported CUDA | 13.0 (per `nvidia-smi`) |
| PyTorch | 2.8.0 (bundled CUDA runtime)        |
| PyTorch CUDA runtime (`torch.version.cuda`) | 12.8 |

### 3. Download the data

The curated DRIP training and evaluation data is archived on Zenodo
(DOI: [10.5281/zenodo.20325769](https://doi.org/10.5281/zenodo.20325769)).

Download and extract it into the repository root:

```bash
wget -O datasets.zip "https://zenodo.org/records/20325769/files/datasets.zip?download=1"
unzip datasets.zip
mv datasets1/ datasets/ # rename it to datasets/
```

This restores `datasets/sep/sep_data_cleaned_dpo_gpt.json` and the other
curated files used by the training and evaluation scripts.

To regenerate the DRIP training data from scratch instead, see
[`data_generation/README.md`](./data_generation/README.md).

---

## Training

Pick the script that matches your base model:

| Base model | Command |
|---|---|
| Meta-Llama-3-8B-Instruct | `bash ./scripts/llama8b/sep/drip_sep.sh` |
| Mistral-7B-Instruct-v0.3 | `bash ./scripts/mistral7b/sep/drip_sep.sh` |

---

## Evaluation

**Before you start:** copy [`./datasets/openai_configs_example.yaml`](./datasets/openai_configs_example.yaml) to `./datasets/openai_configs.yaml` and fill in your OpenAI configuration.

> Every evaluation script below prompts for **two inputs**: a CUDA device ID and the trained model path
> (e.g. `meta-llama/Meta-Llama-3-8B-Instruct-TextTextText-sep-drip`). The steps omit this prompt for brevity.
>
> The examples use the Llama scripts — swap `llama8b` for `mistral7b` to evaluate the other model.

### SEP score

1. Run [`./scripts/evaluation/llama8b/sep.sh`](./scripts/evaluation/llama8b/sep.sh).
2. Run the SEP judge [`./testing/sep/evaluation_main_corrected_submit.py`](./testing/sep/evaluation_main_corrected_submit.py), then [`./testing/sep/evaluation_main_corrected_collect.py`](./testing/sep/evaluation_main_corrected_collect.py) to print the SEP metric.

### ASR

**Alpaca heuristic-based attacks**

1. Run [`./scripts/evaluation/llama8b/alpaca_injection.sh`](./scripts/evaluation/llama8b/alpaca_injection.sh).
2. Run [`./testing/evaluation_main.py`](./testing/evaluation_main.py) to print ASR under the Naive, Ignore, Completion, Escape, and HackaPrompt attacks.

**GCG-based adaptive attacks**

See [`gcg/README.md`](./gcg/README.md). GCG requires a separate legacy environment because newer `transformers` versions trigger OOM.

**InjecAgent**

1. Run [`./scripts/evaluation/llama8b/injecagent.sh`](./scripts/evaluation/llama8b/injecagent.sh).

### Utility

**AlpacaEval 2.0** (can cost up to USD 50)

1. Run [`./scripts/evaluation/llama8b/alpaca_utility.sh`](./scripts/evaluation/llama8b/alpaca_utility.sh).
2. If the `alpacaeval` command did not run successfully, run it manually:

   ```bash
   export OPENAI_CLIENT_CONFIG_PATH=./datasets/openai_configs.yaml && \
   alpaca_eval --model_outputs [model_path]/predictions_on_davinci_003_outputs.json \
   --reference_outputs datasets/gpt4o_outputs.json
   ```

3. Find the win rate in `model-path/weighted_alpaca_eval_gpt4_turbo/leaderboard.csv`.

**IFEval**

1. Run [`./scripts/evaluation/llama8b/ifeval.sh`](./scripts/evaluation/llama8b/ifeval.sh).
2. Run [`./testing/ifeval/evaluation_main.py`](./testing/ifeval/evaluation_main.py) and look for ASR strict.

**MT-Bench**

1. Run [`./scripts/evaluation/llama8b/mtbench.sh`](./scripts/evaluation/llama8b/mtbench.sh).
2. Run [`./testing/mt_bench/gen_judgment.py`](./testing/mt_bench/gen_judgment.py) with `--model-path [model-path] --model-id [model name, e.g. Ours]`.
3. Plot the radar chart with [`./testing/mt_bench/plot.py`](./testing/mt_bench/plot.py).

**MMLU**

1. Run [`./scripts/evaluation/llama8b/mmlu_utility.sh`](./scripts/evaluation/llama8b/mmlu_utility.sh).
2. Run [`./testing/mmlu/evaluation_main.py`](./testing/mmlu/evaluation_main.py).

---

## Limitations

DRIP is developed and evaluated under a deliberately scoped threat model. The following limitations matter when applying it elsewhere:

- **Text-to-text attacks.** Both the primary task and the injected task are text instruction-following — the injected content is natural-language instructions embedded in a text data section. We do not specifically optimize for tool-calling agents, and in tool-calling scenarios the utility is expected to drop (the [AgentDojo](./testing/agentdojo/README.md) setting illustrates this regime).
- **Dense architectures.** The representation editing is designed and validated on dense transformer architectures. We have not fully tested it on Mixture-of-Experts (MoE) models, where inserting new layers poses additional challenges. For MoE backbones we recommend our data-curation recipe together with the residual re-instruction fusion, while the de-instruction shift layer may be unnecessary (and harder to inject).
- **Single modality.** Extending DRIP to multi-modal agents — GUI agents, browser use, OS use — requires new adaptation that is outside the scope of this work.

## Acknowledgments

Parts of this codebase are adapted from the following Meta AI (FAIR) projects:

- [facebookresearch/Meta_SecAlign](https://github.com/facebookresearch/Meta_SecAlign)
- [facebookresearch/SecAlign](https://github.com/facebookresearch/SecAlign)

We thank the authors for releasing their code.

## Citation

> A BibTeX entry will be added here once the paper is released. It is currently under review.
