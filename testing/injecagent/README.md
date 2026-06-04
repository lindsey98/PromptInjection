# InjecAgent — Tool-Calling Prompt Injection

[InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) measures susceptibility
to **indirect prompt injection in a tool-using agent**. The agent calls a tool,
and the **tool's returned observation** contains an injected attacker
instruction. The model should keep doing the user's task and ignore the injected
instruction in the observation.

Two attack categories:

- **Direct Harm (`dh`)** — the injection tries to make the agent call a harmful
  tool (e.g. transfer money, delete data).
- **Data Stealing (`ds`)** — the injection tries to exfiltrate the user's private
  data to the attacker.

The injected instruction lives in the **tool/observation** content — the
untrusted slot (`user` in 3-role, `tool` in 4-role) that DRIP shifts.

**Metric** — attack success rate (ASR), computed by a deterministic parser of the
agent's tool call (`get_score`):

```
ASR = successful_injections / valid_cases
valid rate = parseable outputs / total   # outputs that produced a well-formed action
```

Reported separately for `dh` and `ds`. Lower ASR = more robust.

## Run

```bash
bash ./scripts/evaluation/llama8b/injecagent.sh   # prompts for CUDA id + model path
```

`testing/injecagent/test_injecagent.py` generates the agent outputs and prints
the per-category ASR and valid rate; the model class is auto-detected from the
path.
