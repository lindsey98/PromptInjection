# PAIR — Adaptive Attack (Iterative Attacker LLM)

[PAIR](https://github.com/patrickrchao/JailbreakingLLMs) (Prompt Automatic
Iterative Refinement) is an **adaptive** prompt-injection attack: an *attacker
LLM* (e.g. `gpt-4o-mini`) proposes an injection, the target model is queried, a
judge scores whether the injected task succeeded, and the score+response are fed
back so the attacker **refines** its injection over several rounds. Unlike the
fixed heuristic attacks, PAIR adapts to the specific target/defense.

The injected task is placed in the data section (`--inject_position`, default
`end`). Success is decided by **witness presence** (the injected task's expected
output appears), so the per-round judge is the binary witness check.

**Metric** — ASR after up to `--max_iterations` refinement rounds, reported as
exact-match / begin-with / in-response (same matching as the heuristic ASR).

## Run

Requires an OpenAI API key (the attacker LLM): `export OPENAI_API_KEY=...`.

1. Run the attack (writes `predictions_on_sep_pair.jsonl` into the model dir):

   ```bash
   # SEP data
   python -m testing.pair.test_pair_sep -m <model_path> \
     [--customized_model_class LlamaForCausalLMDRIP] \
     [--attacker_model gpt-4o-mini] [--max_iterations 10] [--inject_position end]

   # Alpaca data
   python -m testing.pair.test_pair_alpaca -m <model_path> [...]
   ```

   `-m` / `--customized_model_class` / `--base_model_path` are the shared model
   args (see `testing/argparse_common.py`).

2. Score:

   ```bash
   python -m testing.pair.evaluation_main
   ```

   > Note: `evaluation_main.py` currently reads the model directory from a
   > hardcoded `ours_model_path` at the top of `main()` — set it to your model dir
   > before running. It prints exact-match / begin-with / in-response ASR.
