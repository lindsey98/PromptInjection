# DPO Pair Generation for Prompt Injection Defense

This script builds **DPO preference pairs** from a prompt-injection dataset. 
Each pair teaches the model to follow the user's original instruction while treating injected content as inert data.

## Core idea

| Branch | Behavior | Role |
|--------|----------|------|
| **chosen** | Executes ONLY the original task; injected probe stays as plain data | preferred |
| **rejected** | Executes the injected probe on clean input | dispreferred |

Pairing the two gives a `(chosen, rejected)` signal for DPO.

## Pipeline

1. **Load** injected requests from `{name}_injected_diff_output.json`. Each request has `instruction`, `clean_input`, `injected_input`, `injected_probe`.
2. **`process_chosen`** wraps input as `<instruction>...</instruction><start of data>...<end of data>`, strips the `INSISTENCE` marker, and queries the model with a system message enforcing data/instruction separation.
   - If `--no_judge` is off, an LLM judge checks whether the probe leaked into the answer. On `"NOT INCLUDED"`, it retries once with corrective feedback.
3. **`process_rejected`** feeds the `injected_probe` as a developer message over `clean_input`, producing the non-compliant (injected) response.
4. **`build_dpo_pairs`** joins chosen and rejected on `(instruction, input)`, then the result is shuffled and saved.

## Run

```bash
python data_curation_drip.py --dataset alpaca          # or: sep
python data_curation_drip.py --dataset alpaca --no_judge   # ablation
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--dataset` | `alpaca` | `alpaca` or `sep` |
| `--no_judge` | off | ablation, skip judge validation/retry |
| `--batch_size` | 100 | requests per checkpoint |
| `--max_concurrent` | 8 | concurrent API calls |

Requires `OPENAI_API_KEY`. Judge prompt loaded from `./data_generation/tasktracker_judge_prompt2.txt`.

## Outputs (`./datasets/`)

- `{name}_injected_diff_output_retrieved{suffix}.json` chosen
- `{name}_injected_executed_retrieved.json` rejected
- `{name}_dpo{suffix}.json` final DPO pairs

`suffix = _ablate_no_judge` when `--no_judge` is set.

## Notes

- **Async + checkpointing.** `run_batch` resumes from existing output; processing is concurrent, gated by a semaphore.
- **Ablation design.** `--no_judge` isolates the contribution of the judge-based validation/retry step on chosen quality.