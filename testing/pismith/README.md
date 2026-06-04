# PISmith — Adaptive Attack (RL-Trained Attacker LLM)

PISmith is the strongest adaptive baseline: instead of prompting a frozen
attacker LLM (PAIR/TAP), it **trains an injection-attacker model with RL
(GRPO)** against a *specific target*. The attacker learns, from a binary success
reward (witness presence in the target's output), to craft injections that beat
that target's defense. Because the attacker is trained per target, evaluation is
**two phases: train, then test**.

**Metric** — ASR@N (best of N sampled attacker rollouts) and ASR@1 against the
target, reported by the test phase.

> ⚠️ **You must train the attacker first, then test.** The test phase loads the
> trained attacker adapter and will fail if it does not exist.

## 1. Train the attacker (required first)

```bash
bash testing/pismith/train.sh <target_model_path> <alpaca|sep> [cuda_devices]
```

- `<target_model_path>` — the model under attack. The target type is **auto-detected
  from the path**: `drip`/`instfuse` → DRIP (`LlamaForCausalLMDRIP`), `secalign`
  → Meta-SecAlign, otherwise undefended Llama.
- `<alpaca|sep>` — selects `train_alpaca` / `train_sep` and the output dir.
- `[cuda_devices]` — comma-separated GPUs (e.g. `0,1,2`); multiple → `torchrun` DDP,
  single → `python`. Default `0`.

The trained attacker is saved to `./pismith_ckpt/<dataset>_<target_tag>/attack_lm_final`
(e.g. `pismith_ckpt/sep_drip/attack_lm_final`).

```bash
# examples
bash testing/pismith/train.sh meta-llama/Meta-Llama-3-8B-Instruct-...-instfuse-sep-dpo sep 1,2,3
bash testing/pismith/train.sh meta-llama/Meta-Llama-3-8B-Instruct sep 4   # undefended target
```

## 2. Test (after training)

```bash
bash testing/pismith/test.sh <target_model_path> <alpaca|sep> [cuda_device]
```

This loads the trained attacker adapter and reports ASR@N / ASR@1 against the
target.

> Note: `test.sh` currently resolves the attacker adapter as
> `./pismith_ckpt/sep_<target_tag>/attack_lm_final` (the `sep_` prefix is fixed),
> so **train with `sep` first** — or pass a 4th arg `llama` to force the
> `sep_llama` attacker. Make sure the target type detected at test time matches
> the one you trained against.
