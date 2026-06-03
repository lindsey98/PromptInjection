# AgentDojo Evaluation

This directory evaluates DRIP (and the baselines) on
[AgentDojo](https://github.com/ethz-spylab/agentdojo), an **agentic** prompt-injection
benchmark. Unlike the single-turn tests in the main README, AgentDojo runs the model as a
tool-calling agent across four task suites — `banking`, `slack`, `travel`, and `workspace` —
and reports two numbers per suite:

- **Utility** — how often the agent completes the user's legitimate task.
- **Security** — how often it resists an injection planted in a tool's output
  (higher = more attacks blocked).

> **Base model:** all AgentDojo experiments use **`meta-llama/Llama-3.1-8B-Instruct`**.

## Chat format: 4 roles here vs. 3 roles in the main README

This is the key difference from the rest of the repo, so it is worth stating up front.

- **Main README evaluations** (SEP, Alpaca injection, InjecAgent, utility) use a
  **3-role** chat format: **`system`**, **`user` (untrusted)**, and **`assistant`**.
  The injected/untrusted content lives in the `user` turn.
- **AgentDojo** is agentic, so the model also reads back **tool outputs** — and that is where
  injections hide. We therefore switch to a **4-role** format:
  **`system`** → **`user`** → **`tool` (untrusted)** → **`assistant`**.
  The injected content lives in the `tool` turn rather than the `user` turn.

In `--mode fuse` (DRIP), the fuse pipeline assigns the trusted/untrusted slots
internally via `expert_labels`, so DRIP knows which tokens came from the untrusted `tool`
role. In `--mode official`, the role used for tool outputs is controlled by
`--tool-delimiter` (see the flag table below).

## Install

```bash
pip install agentdojo==0.1.35
```

## Two run modes

| Mode | Use it for | How the model is served |
|---|---|---|
| `--mode official` | Undefended baseline and Meta SecAlign | A local **vLLM** server (start it first with the provided script) |
| `--mode fuse`     | DRIP                                  | Loaded directly in-process (no server needed) |

In every example below, the command shown runs the **no-attack** (utility-only) setting.
To run **with an attack**, append:

```bash
--attack [important_instructions|ignore_previous]
```

Add `--force-rerun` (`-f`) to ignore cached results, and `--suites banking` (for example)
to limit the run to a single suite.

---

## 1. Undefended baseline (`--mode official`)

Start the vLLM server (it serves `Llama-3.1-8B-Instruct` under the name `local`), then run
the benchmark in a second shell:

```bash
bash run_local_vlm.sh   # terminal 1: start the local vLLM server, wait until it is ready

# terminal 2:
python -m testing.agentdojo.run_agentdojo \
  --mode official \
  -m local \
  --logdir agentdojo_runs/llama8b
```

## 2. Meta SecAlign baseline (`--mode official`)

SecAlign is served as a LoRA adapter on top of the same base model, and it expects tool
outputs in the dedicated `input` role, so add `--tool-delimiter input`:

```bash
bash run_local_vlm_metasecalign.sh   # terminal 1: start the SecAlign vLLM server

# terminal 2:
python -m testing.agentdojo.run_agentdojo \
  --mode official \
  -m local \
  --logdir agentdojo_runs/metasecalign8b \
  --tool-delimiter input
```

## 3. DRIP (`--mode fuse`)

No vLLM server is required — point `--model_name_or_path` at your trained checkpoint:

```bash
python -m testing.agentdojo.run_agentdojo \
  --mode fuse \
  --model_name_or_path [model_path] \
  --customized_model_class LlamaForCausalLMDRIP \
  --logdir ./agentdojo_runs/llama8b_drip
```

---

## Useful flags

| Flag | Applies to | Meaning |
|---|---|---|
| `--mode` | both | `official` (upstream pipeline, vLLM-served) or `fuse` (DRIP). |
| `-m`, `--model_name_or_path` | both | In `official`, the served model name (`local`). In `fuse`, the checkpoint path. |
| `--customized_model_class` | `fuse` | Model registry key, e.g. `LlamaForCausalLMDRIP`. |
| `--tool-delimiter` | `official` | Role used for tool outputs. `tool` (default) or `input` (SecAlign). Ignored in `fuse`. |
| `--attack` | both | Omit for no-attack; set to `important_instructions` or `ignore_previous` to inject. |
| `--suites`, `-s` | both | One or more of `banking slack travel workspace` (default: all four). |
| `--logdir` | both | Where per-task logs and the summary JSON are written. |
| `--force-rerun`, `-f` | both | Ignore cached results and recompute. |

## Reading the results

Each run prints per-suite **utility** and (when `--attack` is set) **security** percentages,
plus an `OVERALL` block when more than one suite runs. A machine-readable summary is also
written to:

```
<logdir>/<pipeline-name>_<mode>_<attack|no-attack>_<defense|no-defense>_summary.json
```
