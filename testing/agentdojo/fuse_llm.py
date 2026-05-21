from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Collection, Sequence
from typing import Optional, Dict
from data_generation.sft_data_loader import compute_expert_labels_from_input_ids

import torch
import transformers
from pydantic import ValidationError

# ── AgentDojo internals ──────────────────────────────────────────────────────
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, load_system_message
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor, tool_result_to_str
from agentdojo.functions_runtime import EmptyEnv, Env, Function, FunctionCall, FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)
from testing.test import load_full_model

logger = logging.getLogger(__name__)
# At top of main(), before build_fuse_pipeline
from agentdojo.agent_pipeline.tool_execution import ToolsExecutor
from agentdojo.types import get_text_content_as_str as _gtcas
from config import DELIMITERS, PROMPT_FORMAT
from modeling import (
    LlamaDRIPConfig, LlamaForCausalLMDRIP,
    LlamaISEConfig, LlamaForCausalLMISE, LlamaForCausalLMPFT,
    MistralDRIPConfig, MistralForCausalLMDRIP,
    MistralISEConfig, MistralForCausalLMISE, MistralForCausalLMPFT,
    Qwen3DRIPConfig, Qwen3ForCausalLMDRIP
)

_orig_te_query = ToolsExecutor.query

def _debug_te_query(self, query, runtime, env, messages, extra_args):
    result = _orig_te_query(self, query, runtime, env, messages, extra_args)
    new_msgs = result[3][len(messages):]
    for m in new_msgs:
        if m["role"] == "tool":
            content = _gtcas(m["content"]) if m["content"] else ""
            print(f"[ToolsExecutor] fn={m['tool_call'].function} | error={m.get('error')} | content={repr(content[:300])}")
    return result

ToolsExecutor.query = _debug_te_query

_tool_calling_prompt = """
## Currently Available Functions
{functions}

## Function Calling
If you choose to call a particular function, include the function call in the following format as part of your response:
```
<function=function_name>{{"param1": "val1", ...}}</function>
```

where the JSON inside <function=...> contains exactly the parameters for that function. 
Pass the arguments in correct format, i.e., strings should be enclosed in quotes, lists should be enclosed in square brackets, integers should have no quotes, etc.
 If there are no parameters, then you should still include an empty JSON object `{{}}` inside the <function=...> tag.

### Function Results
After calling a function, stop generating text and wait for the results of the function call in the next message. The next message will use provide the output of the function call as a JSON object. Based on the output of the function call, you may call additional functions or provide a final answer."""



def _tool_schemas_text(tools: Collection[Function]) -> str:
    parts = []
    for tool in tools:
        tool_dict = {
            "name":        tool.name,
            "description": tool.description,
            "parameters":  tool.parameters.model_json_schema(),
        }
        parts.append(json.dumps(tool_dict, indent=2))
    return "\n\n".join(parts)


def _make_system_content(base_system_message: str) -> str:
    """
    Label 0 (system slot) carries ONLY persistent agent-level instructions.
    Tool schemas are moved to label 1 (user slot) — see _build_task_content —
    so that label 0 stays in-distribution w.r.t. training data (short, fixed
    persona/format instructions rather than long task-specific JSON schemas).
    """
    return base_system_message


# ── message-to-text helpers ──────────────────────────────────────────────────

def _content_to_str(content) -> str:
    if content is None:
        return ""
    if isinstance(content, list):
        return get_text_content_as_str(content)
    return str(content)


def _render_assistant(msg: ChatMessage) -> str:
    """Render an assistant message: optional text + optional <function=...> tag."""
    lines: list[str] = []
    text = _content_to_str(msg.get("content"))  # type: ignore[arg-type]
    if text.strip():
        lines.append(text.strip())
    tool_calls = msg.get("tool_calls") or []     # type: ignore[call-overload]
    for tc in tool_calls:
        lines.append(f"<function={tc.function}>{json.dumps(tc.args)}</function>")
    return "\n".join(lines)


def _render_tool_result(msg: ChatMessage) -> str:
    """Render a tool-role message as a labelled result block."""
    tc      = msg["tool_call"]                   # type: ignore[index]
    fn_name = tc.function
    error   = msg.get("error")                   # type: ignore[call-overload]
    if error:
        body = f"ERROR: {error}"
    else:
        body = _content_to_str(msg.get("content"))  # type: ignore[arg-type]
    return f"[tool result: {fn_name}]\n{body}"


def _build_task_content(messages: Sequence[ChatMessage], tools: Collection[Function]) -> str:
    """
    Build the user-slot (label 1, trusted) content:
        ## Currently Available Functions
        ... tool schemas ...
        ## Function Calling
        ... format instructions ...
        ## Task
        ... user's first message ...

    Tool schemas go here (not in system slot) because they are task-level
    trusted information provided by the user/app, not persistent agent rules.
    Task appears last so attention recency bias favours the task over schemas.
    """
    task = "(no task provided)"
    for msg in messages:
        if msg["role"] == "user":
            task = _content_to_str(msg.get("content")).strip()  # type: ignore[arg-type]
            break  # only take the first user message — repeats handled in scratchpad

    if len(tools) == 0:
        return task
    schemas = _tool_schemas_text(tools)
    return (
        _tool_calling_prompt.format(functions=schemas)
        + "\n\n## Task\n\n"
        + task
    )


def _build_scratchpad_content(messages: Sequence[ChatMessage]) -> str:
    """
    Build the scratchpad string from all history AFTER the first user message:
    assistant turns + tool results. This is the UNTRUSTED content that goes in
    the tool/data slot (label 2) in 4-role mode. Injection vectors live here.

    Rendered as a flat sequence — no <step N> wrappers — so the model sees
    exactly the same assistant/tool alternation it was trained on, and injection
    strings inside tool results are preserved verbatim.

    Returns an empty string if there is no history yet (first step).
    """
    lines: list[str] = []
    past_first_user: bool = False

    for msg in messages:
        role = msg["role"]
        if role == "system":
            continue
        if role == "user":
            if not past_first_user:
                past_first_user = True  # skip the task itself
            else:
                # repeat_user_prompt defense
                txt = _content_to_str(msg.get("content"))  # type: ignore[arg-type]
                lines.append(txt.strip())
            continue
        if not past_first_user:
            continue
        if role == "assistant":
            rendered = _render_assistant(msg)
            if rendered.strip():
                lines.append(rendered)
        elif role == "tool":
            lines.append(_render_tool_result(msg))

    return "\n".join(lines)


def _build_user_content(messages: Sequence[ChatMessage], tools: Collection[Function]) -> str:
    """
    3-role fallback: collapse (schemas + task + scratchpad) into one user-turn.
    """
    task       = _build_task_content(messages, tools)
    scratchpad = _build_scratchpad_content(messages)
    if scratchpad:
        return task + "\n" + scratchpad
    return task


# ═══════════════════════════════════════════════════════════════════════════════
# Output parser
# ═══════════════════════════════════════════════════════════════════════════════

_FUNC_RE = re.compile(r"<function\s*=\s*([^>]+)>(.*?)</function>", re.DOTALL)


def _parse_output(completion: str) -> ChatAssistantMessage:
    content_block = [text_content_block_from_string(completion.strip())]
    tool_calls: list[FunctionCall] = []
    seen: set[tuple[str, str]] = set()

    for match in _FUNC_RE.finditer(completion):
        fn_name = match.group(1).strip()
        raw_json = match.group(2).strip()

        if not raw_json:
            args: dict = {}
        else:
            last_brace = raw_json.rfind("}")
            if last_brace != -1:
                raw_json = raw_json[:last_brace + 1]
            try:
                parsed = json.loads(raw_json)
                if not isinstance(parsed, dict):
                    raise ValueError(f"args must be a JSON object, got {type(parsed)}")
                args = parsed
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                logger.debug("_parse_output: skipping malformed — %s  raw=%r", exc, raw_json)
                continue

        key = (fn_name, json.dumps(args, sort_keys=True))
        if key in seen:
            logger.debug("_parse_output: dropping duplicate tool call %s", key)
            continue
        seen.add(key)
        tool_calls.append(FunctionCall(function=fn_name, args=args))

    return ChatAssistantMessage(
        role="assistant",
        content=content_block,
        tool_calls=tool_calls,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# FuseLLM  —  the core pipeline element
# ═══════════════════════════════════════════════════════════════════════════════

class FuseLLM(BasePipelineElement):
    """
    AgentDojo pipeline element wrapping a 4-role fuse causal LM.

    Slot assignment (4-role / num_labels=4):
        label 0 (instruction):  system message (persistent agent rules)
        label 1 (user/input):   tool schemas + user task (trusted)
        label 2 (tool/data):    scratchpad — assistant + tool history (untrusted)
        label 3 (response):     model output

    Every time this element is called it rebuilds a fresh prompt from the full
    accumulated message list, so multi-turn history is preserved without ever
    needing 'assistant' or 'tool' roles in the prompt itself.

    Args:
        model:             Loaded HF CausalLM (already on the right device).
        tokenizer:         Matching tokenizer.
        model_name:        Identifier string used in AgentDojo result logs.
        max_new_tokens:    Generation budget per step.
        temperature:       0.0 → greedy decoding.
        pass_expert_labels:
            If True, compute expert_labels from the input_ids and pass them
            to model.generate(). Requires that set_delimiter_ids_in_config()
            has been called on the model config (llama_drip helper).
    """

    def __init__(
        self,
        config:          Dict,
        prompt_templates: Dict[str, str],
        model:              transformers.PreTrainedModel,
        tokenizer:          transformers.PreTrainedTokenizerBase,
        model_name:         str = "fuse-llm",
        max_new_tokens:     int = 512,
        temperature:        float = 0.0,
        pass_expert_labels: bool = True,
    ) -> None:
        self.model           = model
        self.tokenizer       = tokenizer
        self.name            = model_name
        self.max_new_tokens  = max_new_tokens
        self.temperature     = temperature
        self.pass_expert_labels = pass_expert_labels
        self.config          = config
        # Prompt templates built from DELIMITERS — mirrors PROMPT_FORMAT exactly.
        # Keys: "prompt_input", "prompt_no_input", and "prompt_input_tool" (4-role only).
        self.prompt_templates = prompt_templates

        # Check whether the model config carries delimiter IDs
        self._has_delimiters: bool = (
            config is not None
            and getattr(config, "data_delm_ids",     None) is not None
            and getattr(config, "response_delm_ids",  None) is not None
            and getattr(config, "inst_delm_ids",      None) is not None
        )
        if pass_expert_labels and not self._has_delimiters:
            logger.warning(
                "FuseLLM: pass_expert_labels=True but model config has no delimiter IDs. "
                "expert_labels will be skipped. Call set_delimiter_ids_in_config() first."
            )

    # ------------------------------------------------------------------
    def _tokenize(
        self,
        system_text: str,
        user_text:   str,
        tool_text:   str | None = None,
    ) -> torch.Tensor:

        templates = self.prompt_templates

        prompt = templates["prompt_input_tool"].format(
            instruction=system_text,
            user=user_text,
            input=tool_text if tool_text else "no history yet",
        )
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,  # delimiters already contain special tokens
        ).input_ids
        return input_ids.to(self.model.device)   # [1, L]

    # ------------------------------------------------------------------
    def _maybe_expert_labels(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        """
        Compute expert_labels from input_ids using delimiter IDs stored in
        model.config. Returns None if not applicable.
        """
        if not (self.pass_expert_labels and self._has_delimiters):
            return None
        try:
            cfg = self.config

            labels = compute_expert_labels_from_input_ids(
                input_ids,
                data_delm_ids=torch.tensor(cfg.data_delm_ids).to(input_ids.device).tolist(),
                response_delm_ids=torch.tensor(cfg.response_delm_ids).to(input_ids.device).tolist(),
                inst_delm_ids=torch.tensor(cfg.inst_delm_ids).to(input_ids.device).tolist(),
                num_labels=getattr(cfg, "num_labels", 4),
            )

            return labels.to(self.model.device)  # [1, L]
        except Exception as exc:
            logger.debug("_maybe_expert_labels failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _generate(self, system_text: str, user_text: str, tool_text: str | None = None) -> str:
        """One forward + generate pass → decoded string (new tokens only)."""
        input_ids = self._tokenize(system_text, user_text, tool_text)

        gen_kwargs: dict = dict(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0.0),
            pad_token_id=(
                self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            ),
            use_cache=True,
        )
        if self.temperature > 0.0:
            gen_kwargs["temperature"] = self.temperature

        expert_labels = self._maybe_expert_labels(input_ids)
        if expert_labels is not None:
            gen_kwargs["expert_labels"] = expert_labels

        out_ids = self.model.generate(**gen_kwargs)
        new_ids = out_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    def query(
        self,
        query:      str,
        runtime:    FunctionsRuntime,
        env:        Env = EmptyEnv(),
        messages:   Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        # 1. Extract system message text from the messages list
        base_system = ""
        for m in messages:
            if m["role"] == "system":
                base_system = _content_to_str(m.get("content"))  # type: ignore[arg-type]
                break

        tools = runtime.functions.values()

        # 2. Build prompt content
        system_content = _make_system_content(base_system)
        num_labels = getattr(self.config, "num_labels", 4)

        if num_labels == 4:
            # 4-role: schemas + task in user slot (label 1), scratchpad in tool slot (label 2).
            # On step 1 there is no scratchpad yet, so tool_content=None and _tokenize uses
            # the placeholder "no history yet" — keeps the 3-slot structure consistent
            # with training distribution.
            user_content = _build_task_content(messages, tools)
            scratchpad   = _build_scratchpad_content(messages)
            tool_content = scratchpad if scratchpad else None
        else:
            # 3-role fallback: collapse schemas + task + scratchpad into user slot.
            user_content = _build_user_content(messages, tools)
            tool_content = None

        logger.debug(
            "FuseLLM.query | step=%d | 4role=%s | sys_len=%d | user_len=%d | tool_len=%d",
            sum(1 for m in messages if m["role"] == "assistant"),
            num_labels == 4,
            len(system_content),
            len(user_content),
            len(tool_content) if tool_content else 0,
        )

        # 3. Generate
        completion = self._generate(system_content, user_content, tool_content)
        logger.debug("FuseLLM output:\n%s", completion)

        # 4. Parse and return
        output_msg = _parse_output(completion)
        return query, runtime, env, [*messages, output_msg], extra_args


def set_delimiter_ids_in_config(cfg, tokenizer, inst_delm, data_delm,
                                response_delm, num_labels):
    """Encode delimiters and extract first N tokens as anchor."""
    inst_ids = tokenizer.encode(inst_delm, add_special_tokens=False)
    data_ids = tokenizer.encode(data_delm, add_special_tokens=False)
    resp_ids = tokenizer.encode(response_delm, add_special_tokens=False)

    cfg.inst_delm_ids = inst_ids
    cfg.data_delm_ids = data_ids
    cfg.response_delm_ids = resp_ids
    cfg.num_labels = num_labels

# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline builder  —  the main public API
# ═══════════════════════════════════════════════════════════════════════════════

def build_fuse_pipeline(
    model_path:              str,
    customized_model_class:  Optional[str]     = "QwenForCausalLMFuse",
    model_name:              Optional[str]     = None,
    system_message:          str               = "",
    device_map:              str | dict | int | None = "auto",
    torch_dtype:             torch.dtype       = torch.bfloat16,
    max_new_tokens:          int               = 512,
    temperature:             float             = 0.0,
    pass_expert_labels:      bool              = True,
    max_iters:               int               = 15,
) -> AgentPipeline:
    """
    Build a complete AgentDojo pipeline for a 4-role fuse model.

    Slot layout (num_labels=4):
        label 0 (system slot):  agent instructions only (no tool schemas)
        label 1 (user slot):    tool schemas + trusted task
        label 2 (tool slot):    scratchpad — assistant + tool history (untrusted)
        label 3 (response):     model output

    Pipeline structure:
        SystemMessage → InitQuery → FuseLLM → ToolsExecutionLoop(ToolsExecutor, FuseLLM)
    """

    if model_name is None:
        model_name = os.path.basename(model_path.rstrip("/\\"))

    # ── Load model ──────────────────────────────────────────────────────
    model, tokenizer, frontend_delimiters, training_attacks = load_full_model(
        model_path,
        customized_model_class,
        load_as_adapter=True,
        base_model_path=os.path.join(os.getenv("HF_HOME"), os.path.basename(model_path.rstrip("/\\")).split("-TextTextText")[0]),
        load_in_bnb4=True
    )

    delm = DELIMITERS[frontend_delimiters]
    num_labels = 4 if len(delm) == 4 else 3


    if num_labels == 4:
        config = Qwen3DRIPConfig()
        set_delimiter_ids_in_config(config, tokenizer,  delm[1], delm[2], delm[3], num_labels)

    else:
        config = Qwen3DRIPConfig()
        set_delimiter_ids_in_config(config, tokenizer,  delm[0], delm[1], delm[-1], num_labels)


    # ── Build prompt templates (mirrors PROMPT_FORMAT from training) ─────
    prompt_templates = dict(PROMPT_FORMAT[frontend_delimiters])
    if num_labels == 4:
        # prompt_input_tool: 4-slot variant used every step.
        #   {instruction} = system content (agent rules only)
        #   {user}        = tool schemas + trusted task
        #   {input}       = scratchpad / tool results (untrusted, label 2)
        prompt_templates["prompt_input_tool"] = (
            delm[0] + "\n{instruction}\n\n"
            + delm[1] + "\n{user}\n\n"
            + delm[2] + "\n{input}\n\n"
            + delm[3] + "\n"
        )
    else:
        # 3-role: {user} carries schemas+task+scratchpad together.
        prompt_templates["prompt_input_tool"] = (
            delm[0] + "\n{instruction}\n\n\n{user}\n\n"
            + delm[1] + "\n{input}\n\n"
            + delm[2] + "\n"
        )
    logger.info(
        "Prompt templates for '%s': %s",
        frontend_delimiters,
        list(prompt_templates.keys()),
    )

    # ── Build pipeline element ──────────────────────────────────────────
    fuse_llm = FuseLLM(
        config=config,
        prompt_templates=prompt_templates,
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        pass_expert_labels=pass_expert_labels,
    )

    # ── System message ──────────────────────────────────────────────────
    system_message = load_system_message(None)   # AgentDojo default

    system_element = SystemMessage(system_message)
    init_query     = InitQuery()

    # ── Tool execution loop ─────────────────────────────────────────────
    tools_loop = ToolsExecutionLoop(
        elements=[ToolsExecutor(tool_result_to_str), fuse_llm],
        max_iters=max_iters,
    )

    # ── Assemble pipeline ───────────────────────────────────────────────
    pipeline = AgentPipeline(
        elements=[system_element, init_query, fuse_llm, tools_loop]
    )
    pipeline.name = model_name
    return pipeline