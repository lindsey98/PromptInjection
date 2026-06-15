# DRIP

[![DOI](https://zenodo.org/badge/917193669.svg)](https://doi.org/10.5281/zenodo.20695257)

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
- [(Optional) Download pretrained checkpoints](#optional-download-pretrained-checkpoints)
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
(DOI: [10.5281/zenodo.20603331](https://doi.org/10.5281/zenodo.20603331)).

Download and extract it into the repository root:

```bash
wget -O datasets.zip "https://zenodo.org/records/20603331/files/datasets.zip?download=1"
unzip datasets.zip
mv datasets1/ datasets/ # rename it to datasets/
```

This restores `datasets/sep/sep_data_cleaned_dpo_gpt.json` and the other
curated files used by the training and evaluation scripts.

To regenerate the DRIP training data from scratch instead, see
[`data_generation/README.md`](./data_generation/README.md).

---

## Training

Pick the script that matches your **base model** and the **dataset** you want to
train on. The scripts are grouped into per-dataset folders (`sep/`, `alpaca/`) —
to train on SEP run the `sep/` script, to train on Alpaca go to the `alpaca/`
folder and run the matching one there:

| Base model | Dataset | Command |
|---|---|---|
| Meta-Llama-3-8B-Instruct | SEP | `bash ./scripts/llama8b/sep/drip_sep.sh` |
| Llama-3.1-8B-Instruct | Alpaca (3-role) | `bash ./scripts/llama8b/alpaca/drip_alpaca.sh` |
| Llama-3.1-8B-Instruct | Alpaca + InjecAgent (4-role / tool-calling) | `bash ./scripts/llama8b/alpaca/drip_alpaca_4roles.sh` |
| Mistral-7B-Instruct-v0.3 | SEP | `bash ./scripts/mistral7b/sep/drip_sep.sh` |
| Mistral-7B-Instruct-v0.3 | Alpaca (3-role) | `bash ./scripts/mistral7b/alpaca/drip_alpaca.sh` |

Training merges the LoRA adapter into the base weights and saves a **full
checkpoint** that evaluation can load directly. For checkpoints saved as adapters
instead (QLoRA runs, or models trained earlier), merge them first:

```bash
python -m training.merge_lora --adapter_path <adapter_dir> --output_path <merged_dir>
```

### 3-role vs 4-role: train separately

DRIP supports two chat formats, and you train a **separate** model for each (they
use different data and a different delimiter):

| | Eval targets | Training data | Delimiter | Launcher |
|---|---|---|---|---|
| **3-role** (text) | SEP, Alpaca injection, IFEval, MMLU, MT-Bench | SEP DPO pairs | `TextTextText` | `scripts/llama8b/sep/drip_sep.sh` |
| **4-role** (tool-calling) | [AgentDojo](./testing/agentdojo/README.md) | Alpaca + InjecAgent combined DPO | `TextTextText-4roles` | `scripts/llama8b/agentdojo/drip_4roles.sh` |

The 4-role launcher trains on `datasets/alpaca_injecagent_dpo_combined.json` with
the `TextTextText-4roles` delimiter (`--attack TextTextText-4roles_None`). See the
[AgentDojo training-data section](./testing/agentdojo/README.md#training-data-4-role--tool-calling)
for how that data is built and why InjecAgent/Alpaca are mixed in.

**What the roles mean — where the untrusted data goes.** A "role" is just the
delimiter that wraps the untrusted/injected segment. The exact tokens depend on
the base model's chat template (see [`config.py`](./config.py)):

- **Llama-3.1** has a native tool (`ipython`) role, so both formats are available:
  - **3-role** (`TextTextText`) — untrusted data in the **`user`** turn:
    `<|eot_id|><|start_header_id|>user<|end_header_id|>`
  - **4-role** (`TextTextText-4roles`) — untrusted data in the **`ipython`** turn:
    `<|eot_id|><|start_header_id|>ipython<|end_header_id|>`
- **Meta-Llama-3** has **no** tool role, so only **3-role** is possible — untrusted
  data always sits in the `user` turn.
- **Mistral-7B** (`TextTextTextMistral`) has no separate role; the untrusted data
  sits **between `<</SYS>>` and `[/INST]`** — delimiters `['<s>[INST] <<SYS>>', ' <</SYS>>', '[/INST]']`.

> For comparison, Meta SecAlign adds its own dedicated **`input`** role
> (`<|eot_id|><|start_header_id|>input<|end_header_id|>`) for the untrusted segment;
> DRIP instead reuses Llama's native `ipython` role for 4-role tool-calling.

---

## (Optional) Download pretrained checkpoints

If you would rather skip training, we release the DRIP adapters on the Hugging
Face Hub. They are published as **LoRA adapters**, so after downloading you must
**merge** each one into its base model before evaluation (this produces the full
checkpoint the eval scripts load).

| Repo (`Kelsey98/…`) | Base model (`--base_model_path`) | Template | Model class (`--customized_model_class`) | Tool calls |
|---|---|---|---|---|
| [`Llama-3.1-8B-Instruct-TextTextText-4roles-toolcall-drip`](https://huggingface.co/Kelsey98/Llama-3.1-8B-Instruct-TextTextText-4roles-toolcall-drip) | `meta-llama/Llama-3.1-8B-Instruct` | 4-role | `LlamaForCausalLMDRIP` | ✅ |
| [`Meta-Llama-3-8B-Instruct-TextTextText-drip`](https://huggingface.co/Kelsey98/Meta-Llama-3-8B-Instruct-TextTextText-drip) | `meta-llama/Meta-Llama-3-8B-Instruct` | 3-role | `LlamaForCausalLMDRIP` | — |
| [`Mistral-7B-Instruct-v0.3-TextTextTextMistral-drip`](https://huggingface.co/Kelsey98/Mistral-7B-Instruct-v0.3-TextTextTextMistral-drip) | `mistralai/Mistral-7B-Instruct-v0.3` | 3-role | `MistralForCausalLMDRIP` | — |

**Download** the adapter, then **merge** it — substituting `REPO`, `--base_model_path`,
and `--customized_model_class` from the row above:

```bash
REPO=Llama-3.1-8B-Instruct-TextTextText-4roles-toolcall-drip   # pick one from the table

huggingface-cli download Kelsey98/$REPO --local-dir $REPO
CUDA_VISIBLE_DEVICES=0 python -m training.merge_lora \
    --adapter_path "$REPO/" --output_path "$REPO-merged/" \
    --base_model_path meta-llama/Llama-3.1-8B-Instruct \
    --customized_model_class LlamaForCausalLMDRIP
```

Pass the **merged** path (`$REPO-merged/`) as the model path in the
[evaluation](#evaluation) scripts.

---

## Evaluation

**Before you start:** copy [`./datasets/openai_configs_example.yaml`](./datasets/openai_configs_example.yaml) to `./datasets/openai_configs.yaml` and fill in your OpenAI configuration.

> Every evaluation script below prompts for **two inputs**: a CUDA device ID and the trained model path
> (e.g. `meta-llama/Meta-Llama-3-8B-Instruct-TextTextText-sep-drip`). The steps omit this prompt for brevity.
>
> The examples use the Llama scripts — swap `llama8b` for `mistral7b` to evaluate the other model.

### SEP score — 📖 [details](./testing/sep/README.md)

1. Run [`./scripts/evaluation/llama8b/sep.sh`](./scripts/evaluation/llama8b/sep.sh).
2. Run the SEP judge [`./testing/sep/sep_judge.py`](./testing/sep/sep_judge.py), then [`./testing/sep/sep_collect.py`](./testing/sep/sep_collect.py) to print the SEP metric.

### ASR

**Alpaca heuristic-based attacks**

1. Run [`./scripts/evaluation/llama8b/alpaca_injection.sh`](./scripts/evaluation/llama8b/alpaca_injection.sh).
2. Run [`./testing/evaluation_main.py`](./testing/evaluation_main.py) `-m [model_path]` to print ASR under the Naive, Ignore, Completion, Escape, and HackaPrompt attacks.

**GCG-based adaptive attacks**

See [`gcg/README.md`](./gcg/README.md). GCG requires a separate legacy environment because newer `transformers` versions trigger OOM.

**InjecAgent** — 📖 [details](./testing/injecagent/README.md)

1. Run [`./scripts/evaluation/llama8b/injecagent.sh`](./scripts/evaluation/llama8b/injecagent.sh).

**Adaptive attacks: PAIR / TAP / PISmith**

Optimization/search-based attackers that adapt to the target — each has its own guide:

- **PAIR** — iterative attacker LLM — 📖 [`testing/pair/README.md`](./testing/pair/README.md)
- **TAP** — tree-of-attacks with pruning — 📖 [`testing/tap/README.md`](./testing/tap/README.md)
- **PISmith** — RL-trained attacker (**train, then test**) — 📖 [`testing/pismith/README.md`](./testing/pismith/README.md)

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

**IFEval** — 📖 [details](./testing/ifeval/README.md)

1. Run [`./scripts/evaluation/llama8b/ifeval.sh`](./scripts/evaluation/llama8b/ifeval.sh).
2. Run [`./testing/ifeval/evaluation_main.py`](./testing/ifeval/evaluation_main.py) and look for ASR strict.

**MT-Bench** — 📖 [details](./testing/mt_bench/README.md)

1. Run [`./scripts/evaluation/llama8b/mtbench.sh`](./scripts/evaluation/llama8b/mtbench.sh).
2. Run [`./testing/mt_bench/gen_judgment.py`](./testing/mt_bench/gen_judgment.py) with `--model-path [model-path] --model-id [model name, e.g. Ours]`.
3. Plot the radar chart with [`./testing/mt_bench/plot.py`](./testing/mt_bench/plot.py).

**MMLU** — 📖 [details](./testing/mmlu/README.md)

1. Run [`./scripts/evaluation/llama8b/mmlu_utility.sh`](./scripts/evaluation/llama8b/mmlu_utility.sh).
2. Run [`./testing/mmlu/evaluation_main.py`](./testing/mmlu/evaluation_main.py).

---

## Limitations

DRIP is developed and evaluated under a deliberately scoped threat model. The following limitations matter when applying it elsewhere:

- **Text-to-text attacks.** Our primary setting is text instruction-following — the injected content is natural-language instructions embedded in a text data section. For tool-calling agents we additionally release a dedicated **4-role** checkpoint trained with InjecAgent data and evaluated on [AgentDojo](./testing/agentdojo/README.md) (see [pretrained checkpoints](#optional-download-pretrained-checkpoints)); the text-only (3-role) models are not tuned for that regime, so use the 4-role checkpoint for tool-calling.
- **Dense architectures.** The representation editing is designed and validated on dense transformer architectures. We have not fully tested it on Mixture-of-Experts (MoE) models, where inserting new layers poses additional challenges. For MoE backbones we recommend our data-curation recipe together with the residual re-instruction fusion, while the de-instruction shift layer may be unnecessary (and harder to inject).
- **Single modality.** Extending DRIP to multi-modal agents — GUI agents, browser use, OS use — requires new adaptation that is outside the scope of this work.

## Acknowledgments

Parts of this codebase are adapted from the following Meta AI (FAIR) projects:

- [facebookresearch/Meta_SecAlign](https://github.com/facebookresearch/Meta_SecAlign)
- [facebookresearch/SecAlign](https://github.com/facebookresearch/SecAlign)

We thank the authors for releasing their code.

## Citation

> A BibTeX entry will be added here once the paper is released. It is currently under review.
