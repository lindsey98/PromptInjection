# IFEval — Instruction-Following Evaluation

[IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval)
(Google) measures whether a model **follows verifiable instructions** — e.g.
"write at least 3 paragraphs", "respond in all lowercase", "include the word
'data' twice". Each instruction is checked by a deterministic verifier, so no
LLM judge is needed.

This is a **utility** benchmark (no injection). For DRIP it confirms the defense
does not degrade ordinary instruction-following.

**Metric** — accuracy at two strictness levels:

- **strict** — the response satisfies the instruction verbatim.
- **loose** — minor formatting normalizations are allowed before checking.

Reported per-prompt and per-instruction. The headline number is usually
**prompt-level strict accuracy**.

## Run

1. Generate responses:

   ```bash
   bash ./scripts/evaluation/llama8b/ifeval.sh   # prompts for CUDA id + model path
   ```

   `testing/ifeval/test_ifeval.py` writes a responses file into the model dir.

2. Score:

   ```bash
   python -m testing.ifeval.evaluation_main \
     --input_data ./datasets/ifeval/input_data.jsonl \
     --input_response_data <dir>/<responses>.jsonl
   ```

   Prints strict / loose accuracy (prompt- and instruction-level). Look for
   **ASR strict** / prompt-level strict accuracy.

This directory vendors Google's IFEval verifiers (`instructions*.py`,
`evaluation_lib.py`); their upstream `TODO(...)` comments are kept as-is.
