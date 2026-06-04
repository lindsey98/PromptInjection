# MMLU — General Knowledge Utility

[MMLU](https://github.com/hendrycks/test) is a 57-subject multiple-choice
benchmark (STEM, humanities, social science, …). The model picks A/B/C/D for each
question. There is **no injection** here — it is a pure **utility** check that the
defense does not hurt general knowledge / reasoning.

**Metric** — accuracy (fraction of questions answered correctly), reported
overall and optionally by category / subject. The question goes in the
`instruction` slot (trusted), so DRIP applies no data-shift.

## Run

1. Generate answers:

   ```bash
   bash ./scripts/evaluation/llama8b/mmlu_utility.sh   # prompts for CUDA id + model path
   ```

   `testing/mmlu/test_mmlu.py` writes a predictions file into the model dir.
   Useful flags (passed by the script / overridable): `--subjects`, `--split`,
   `--num_few_shot`, `--num_samples`.

2. Score:

   ```bash
   python -m testing.mmlu.evaluation_main --pred_file <dir>/<predictions>.jsonl
   # add --by_subject for the per-subject breakdown
   ```

   Prints `overall_accuracy` plus per-category / per-subject accuracy.
