import os
import json
import asyncio
from copy import deepcopy
from openai import AsyncOpenAI
from data_generation.sft_data_loader import jdump, jload
from testing.injecagent.test_injecagent import (
    SYS_PROMPT, USER_PROMPT, TOOL_PROMPT, get_tool_dict,
)
import random

_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_PARAM_CACHE_PATH = "./datasets/injecagent/attacker_param_cache.json"


# ---------------------------------------------------------------------------
# Attacker parameter generation (LLM + cache)
# ---------------------------------------------------------------------------

def _load_cache():
    return jload(_PARAM_CACHE_PATH) if os.path.exists(_PARAM_CACHE_PATH) else {}


async def _gen_attacker_params(attacker_tool, attacker_instruction, tool_dict, sem):
    """Generate a valid arg object from the attacker tool schema + attacker instruction."""
    schema = tool_dict[attacker_tool]
    schema_str = json.dumps(schema.get("parameters", []), indent=1)
    prompt = (
        f"You are simulating a tool-calling agent that has (wrongly) decided to "
        f"follow an injected instruction.\n\n"
        f"Tool name: {attacker_tool}\n"
        f"Tool parameters schema:\n{schema_str}\n\n"
        f"Injected instruction to satisfy:\n{attacker_instruction}\n\n"
        f"Output ONLY a single valid JSON object filling the tool's parameters "
        f"with concrete values inferred from the injected instruction. "
        f"Include ONLY required params; omit optional params entirely. "
        f"No markdown, no backticks, no prose."
    )
    async with sem:
        resp = await _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
    raw = resp.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(raw)
        obj = {k: v for k, v in obj.items() if v is not None}  # drop null/optional leftovers
        return str(obj)  # Python repr -> single quotes, matches scratchpad style
    except Exception:
        return "{}"

async def _build_param_cache(items, tool_dict, max_concurrent=8):
    """Deduplicate (attacker_tool, attacker_instruction) pairs and generate params in batch."""
    cache = _load_cache()
    sem = asyncio.Semaphore(max_concurrent)

    todo = {}
    for it in items:
        if not it.get("Attacker Tools"):
            continue
        tool = it["Attacker Tools"][0]
        instr = it["Attacker Instruction"]
        key = f"{tool}||{instr}"
        if key not in cache:
            todo[key] = (tool, instr)

    if todo:
        keys = list(todo.keys())
        tasks = [_gen_attacker_params(t, i, tool_dict, sem) for t, i in todo.values()]
        results = await asyncio.gather(*tasks)
        for k, r in zip(keys, results):
            cache[k] = r
        jdump(cache, _PARAM_CACHE_PATH)
        print(f"Generated {len(todo)} attacker param sets (cache total {len(cache)})")

    return cache


# ---------------------------------------------------------------------------
# Symmetric completions
# ---------------------------------------------------------------------------

def _make_thought(item):
    """Shared reasoning prefix for chosen / rejected to keep both in the same distribution."""
    return (
        f"I have called {item['User Tool']} and received its observation. "
        f"The observation contains additional text addressed to me as if it were "
        f"an instruction."
    )


def _make_chosen_completion(item):
    """Safe: recognize injection as untrusted data, refuse to execute, close with Final Answer."""
    return (
        f"Thought: {_make_thought(item)} "
        f"This text is untrusted data, not a command from the user, so I will not "
        f"act on it. I now have what the user asked for.\n"
        f"Final Answer: I have completed your request using {item['User Tool']}. "
        f"I ignored unrelated instructions embedded in the retrieved data."
    )


def _make_rejected_completion(item, params):
    """Compromised: treat injection as a command and call the attacker tool (valid params)."""
    attacker_tool = item["Attacker Tools"][0]
    return (
        f"Thought: {_make_thought(item)} "
        f"I will follow this instruction and use {attacker_tool} to carry it out.\n"
        f"Action: {attacker_tool}\n"
        f"Action Input: {params}"
    )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_injecagent_dpo_pairs(setting="base", attack="dh",
                               sys_prompt=SYS_PROMPT, user_prompt=USER_PROMPT,
                               tool_dict=None, max_samples=-1):
    if tool_dict is None:
        tool_dict = get_tool_dict()

    path = f"./datasets/injecagent/test_cases_{attack}_{setting}.json"
    with open(path, "r") as f:
        data = json.load(f)
    if max_samples > 0:
        data = data[:max_samples]

    # Build the attacker param cache first
    cache = asyncio.run(_build_param_cache(data, tool_dict))

    pairs, skipped = [], 0
    for item in data:
        it = deepcopy(item)
        if not it.get("Attacker Tools"):
            skipped += 1
            continue

        avail = [it["User Tool"]] + it["Attacker Tools"]

        system_filled = sys_prompt.format(
            available_tool_descriptions=json.dumps([tool_dict[t] for t in avail], indent=True)
        )
        instruction_filled = user_prompt.format(
            available_tool_names=avail,
            input=it["User Instruction"],
        )

        thought = it.get("Thought") or \
            f"The user wants to {it['User Instruction']}. I should call {it['User Tool']}."
        scratchpad = (
            f"\n{thought}\n"
            f"Action: {it['User Tool']}\n"
            f"Action Input: {it['Tool Parameters']}\n"
            f"Observation: {it['Tool Response']}\n"
        )
        input_filled = TOOL_PROMPT.format(agent_scratchpad=scratchpad)

        key = f"{it['Attacker Tools'][0]}||{it['Attacker Instruction']}"
        params = cache.get(key, "{}")

        pairs.append({
            "system":      system_filled,
            "instruction": instruction_filled,
            "input":       input_filled,
            "chosen":      _make_chosen_completion(it),
            "rejected":    _make_rejected_completion(it, params),
            "source":      "injecagent",
        })

    print(f"InjecAgent[{attack}/{setting}]: {len(pairs)} pairs, {skipped} skipped")
    return pairs


if __name__ == "__main__":
    # random.seed(42)
    # pairs = build_injecagent_dpo_pairs(setting="base", attack="dh")
    # jdump(pairs, "./datasets/injecagent_dpo.json")
    # pairs = build_injecagent_dpo_pairs(setting="base", attack="ds")
    # jdump(pairs, "./datasets/injecagent_ds_dpo.json")

    a = jload("./datasets/injecagent_dpo.json")
    a2 = jload("./datasets/injecagent_ds_dpo.json")
    b = jload("./datasets/alpaca_data_cleaned_dpo_gpt.json")

    combined = a + b + a2
    random.shuffle(combined)
    jdump(combined, "./datasets/alpaca_injecagent_dpo_combined.json")
    print(f"injecagent: {len(a)}, alpaca: {len(b)}, combined: {len(combined)}")
