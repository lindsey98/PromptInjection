# MT-Bench — Open-Ended Quality (LLM-as-Judge)

[MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) is a
set of multi-turn, open-ended questions across 8 categories (writing, reasoning,
math, coding, extraction, STEM, humanities, roleplay). A strong judge model
(GPT-4) scores each answer **1–10**. This is a **utility / generation-quality**
benchmark (no injection) — for DRIP it checks the defense doesn't degrade answer
quality.

**Metric** — average GPT-4 judge score (overall and per category, shown on a
radar chart).

## Run

1. Generate answers:

   ```bash
   bash ./scripts/evaluation/llama8b/mtbench.sh
   # prompts for CUDA id, model path, and a model id (label, e.g. "Ours")
   ```

   Runs `testing/mt_bench/gen_model_answer.py` (model class auto-detected from the
   path).

2. Judge with GPT-4:

   ```bash
   python -m testing.mt_bench.gen_judgment --model-path <model_path> --model-id <label>
   ```

3. Plot the radar chart:

   ```bash
   python -m testing.mt_bench.plot
   ```

Step 2 requires an OpenAI configuration (`./datasets/openai_configs.yaml`, see the
main README) since the judge is GPT-4.
