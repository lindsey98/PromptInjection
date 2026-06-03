from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Collection, Sequence
from typing import Optional, Dict, List

from data_generation.sft_data_loader import (
    compute_expert_labels_from_input_ids,
)
from transformers import StoppingCriteria, StoppingCriteriaList

import torch
import transformers

# ── AgentDojo internals ──────────────────────────────────────────────────────
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, load_system_message
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor, tool_result_to_str
from testing.test import load_full_model, REGISTRY
from testing.agentdojo.utils import *

logger = logging.getLogger(__name__)

from agentdojo.agent_pipeline.tool_execution import ToolsExecutor
from config import DELIMITERS, PROMPT_FORMAT

_orig_te_query = ToolsExecutor.query


class FuseLLM(BasePipelineElement):

    def __init__(
        self,
        config:          Dict,
        delimiters:       List[str],
        model:              transformers.PreTrainedModel,
        tokenizer:          transformers.PreTrainedTokenizerBase,
        model_name:         str = "fuse-llm",
        max_new_tokens:     int = 1024,
        temperature:        float = 0.0,
        pass_expert_labels: bool = True,
        multiturn:          bool = True,
        repeat_user_instruction: bool = False,
    ) -> None:
        self.model           = model
        self.tokenizer       = tokenizer
        self.name            = model_name
        self.max_new_tokens  = max_new_tokens
        self.temperature     = temperature
        self.pass_expert_labels = pass_expert_labels
        self.config          = config
        self.delimiters      = delimiters     # the 4 (or 3) delimiter strings
        self.multiturn       = multiturn
        self.repeat_user_instruction = repeat_user_instruction

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
    @staticmethod
    def _resolve_eos_ids(tokenizer) -> list[int]:
        """Collect valid stop-token ids: default eos + <|eot_id|> if present."""
        ids: list[int] = []
        if tokenizer.eos_token_id is not None:
            ids.append(tokenizer.eos_token_id)
        try:
            eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot is not None and eot != tokenizer.unk_token_id and eot not in ids:
                ids.append(eot)
        except Exception:
            pass
        return ids

    # ------------------------------------------------------------------
    def _build_multiturn_prompt(self, system_text, user_text, messages):
        delm = self.delimiters
        parts: list[str] = []
        parts.append(delm[0] + "\n" + system_text + "\n\n")
        parts.append(delm[1] + "\n" + user_text + "\n\n")
        parts.append(delm[2] + "\n" + "no history yet" + "\n\n")

        past_first_user = False
        last_tool_body: str | None = None
        last_function_call: str | None = None  # serialised function_name+args

        DEDUP_TOOL_PLACEHOLDER = "[REPEATED_TOOL_RESULT]"
        DEDUP_CALL_PLACEHOLDER = "[REPEATED_FUNCTION_CALL]"

        for msg in messages:
            role = msg["role"]
            if role == "system":
                continue
            if role == "user":
                if not past_first_user:
                    past_first_user = True
                else:
                    txt = content_to_str(msg.get("content")).strip()
                    if txt:
                        parts.append(delm[1] + "\n" + txt + "\n\n")
                        last_tool_body = None
                        last_function_call = None
                continue
            if not past_first_user:
                continue
            if role == "tool":
                body = render_tool_result(msg)
                body_key = body.strip()
                if body_key == last_tool_body:
                    parts.append(delm[2] + "\n" + DEDUP_TOOL_PLACEHOLDER + "\n\n")
                else:
                    last_tool_body = body_key
                    parts.append(delm[2] + "\n" + body + "\n\n")
            elif role == "assistant":
                body = render_assistant(msg)
                if not body.strip():
                    continue

                # Build a key from function calls only (text reasoning ignored)
                call_key = self._serialise_tool_calls(msg)  # see helper below

                if call_key is not None and call_key == last_function_call:
                    parts.append(delm[3] + "\n" + DEDUP_CALL_PLACEHOLDER + "\n\n")
                else:
                    last_function_call = call_key
                    parts.append(delm[3] + "\n" + body + "\n\n")
                    last_tool_body = None  # assistant turn breaks tool sequence

        # Sandwich only useful when there's a chance of injection (i.e., tool history exists)
        if self.repeat_user_instruction and (last_tool_body is not None or last_function_call is not None):
            parts.append(delm[1] + "\n" + user_text + "\n\n")

        parts.append(delm[3])
        return "".join(parts)

    @staticmethod
    def _serialise_tool_calls(msg) -> str | None:
        """
        Return a stable string key for the function call(s) in an assistant message,
        or None if there are no tool calls.

        Handles both dict-shaped tool calls (from JSON logs) and AgentDojo
        Pydantic FunctionCall objects (live runtime).
        """
        import json

        tool_calls = msg.get("tool_calls") if hasattr(msg, "get") else getattr(msg, "tool_calls", None)
        if not tool_calls:
            return None

        parts = []
        for tc in tool_calls:
            # Pydantic FunctionCall: attribute access
            # JSON dict: key access
            if hasattr(tc, "function"):
                fn = tc.function
                args = tc.args if hasattr(tc, "args") else {}
            elif isinstance(tc, dict):
                fn_field = tc.get("function")
                if isinstance(fn_field, dict):
                    # OpenAI-style: {"function": {"name": ..., "arguments": ...}}
                    fn = fn_field.get("name", "")
                    args = fn_field.get("arguments", {})
                else:
                    # AgentDojo-style: {"function": "name", "args": {...}}
                    fn = fn_field or ""
                    args = tc.get("args", {})
            else:
                continue

            # args may be a JSON string (OpenAI) or already a dict (AgentDojo)
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass

            try:
                args_str = json.dumps(args, sort_keys=True, default=str)
            except Exception:
                args_str = str(args)

            parts.append(f"{fn}({args_str})")

        return " | ".join(parts) if parts else None

    # ------------------------------------------------------------------
    def _tokenize(self, system_text: str, user_text: str, messages: Sequence[ChatMessage]) -> torch.Tensor:
        prompt = self._build_multiturn_prompt(system_text, user_text, messages)
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,   # delimiters already contain special tokens
        ).input_ids
        return input_ids.to(self.model.device)

    # ------------------------------------------------------------------
    def _maybe_expert_labels(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        if not (self.pass_expert_labels and self._has_delimiters):
            return None
        try:
            cfg = self.config
            labels = compute_expert_labels_from_input_ids(
                input_ids,
                data_delm_ids=list(cfg.data_delm_ids),
                response_delm_ids=list(cfg.response_delm_ids),
                inst_delm_ids=list(cfg.inst_delm_ids),
                num_labels=getattr(cfg, "num_labels", 4),
            )
            return labels.to(self.model.device)
        except Exception as exc:
            logger.debug("_maybe_expert_labels failed: %s", exc)
            return None

    @torch.inference_mode()
    def _generate(self, system_text: str, user_text: str, messages: Sequence[ChatMessage]) -> str:
        input_ids = self._tokenize(system_text, user_text, messages)

        gen_kwargs: dict = dict(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0.0),
            pad_token_id=(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id),
            use_cache=True,
        )
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [FunctionCallStop(self.tokenizer, input_ids.shape[1])]
        )
        # if self._eos_ids:
        #     gen_kwargs["eos_token_id"] = self._eos_ids
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
        base_system = ""
        for m in messages:
            if m["role"] == "system":
                base_system = content_to_str(m.get("content"))  # type: ignore[arg-type]
                break

        tools = runtime.functions.values()
        system_content = make_system_prompt(base_system, tools)
        user_content   = build_task_content(messages)

        logger.debug(
            "FuseLLM.query | step=%d | multiturn=%s | sys_len=%d | user_len=%d | n_msgs=%d",
            sum(1 for m in messages if m["role"] == "assistant"),
            self.multiturn,
            len(system_content),
            len(user_content),
            len(messages),
        )

        completion = self._generate(system_content, user_content, messages)
        logger.debug("FuseLLM output:\n%s", completion)
        output_msg = parse_output(completion)
        return query, runtime, env, [*messages, output_msg], extra_args


def set_delimiter_ids_in_config(cfg, tokenizer, inst_delm, data_delm,
                                response_delm, num_labels):
    inst_ids = tokenizer.encode(inst_delm, add_special_tokens=False)
    data_ids = tokenizer.encode(data_delm, add_special_tokens=False)
    resp_ids = tokenizer.encode(response_delm, add_special_tokens=False)
    cfg.inst_delm_ids = inst_ids
    cfg.data_delm_ids = data_ids
    cfg.response_delm_ids = resp_ids
    cfg.num_labels = num_labels


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_fuse_pipeline(
    model_path:              str,
    base_model_path: Optional[str],
    customized_model_class:  Optional[str]     = None,
    model_name:              Optional[str]     = None,
    load_as_adapter: bool    = False,
    load_in_bnb4:   bool =False,
    max_new_tokens:          int               = 1024,
    temperature:             float             = 0.0,
    pass_expert_labels:      bool              = True,
    max_iters:               int               = 15,
    multiturn:               bool              = True,
    repeat_user_instruction: bool = False,
) -> AgentPipeline:
    if model_name is None:
        model_name = os.path.basename(model_path.rstrip("/\\"))

    model, tokenizer, frontend_delimiters, training_attacks = load_full_model(
        model_path,
        customized_model_class,
        load_as_adapter=load_as_adapter,
        base_model_path= base_model_path if base_model_path else os.path.join(os.getenv("HF_HOME"),
                                                                              os.path.basename(model_path.rstrip("/\\")).split("-TextTextText")[0]),
        load_in_bnb4=load_in_bnb4,
    )

    # model.config.residual = True  # keep fusion
    # model.config.bit_flip = False

    delm = DELIMITERS[frontend_delimiters]
    num_labels = 4 if len(delm) == 4 else 3

    if customized_model_class and customized_model_class in REGISTRY:
        ConfigCls, _ = REGISTRY[customized_model_class]
    else:
        ConfigCls = type(model.config)

    try:
        config = ConfigCls.from_pretrained(model_path)
    except FileNotFoundError:
        config = ConfigCls()

    if num_labels == 4:
        set_delimiter_ids_in_config(config, tokenizer, delm[1], delm[2], delm[3], num_labels)
    else:
        set_delimiter_ids_in_config(config, tokenizer, delm[0], delm[1], delm[-1], num_labels)

    if num_labels != 4 and multiturn:
        logger.warning("multiturn=True only supported for 4-role; falling back to single-turn.")
        multiturn = False

    fuse_llm = FuseLLM(
        config=config,
        delimiters=delm,
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        pass_expert_labels=pass_expert_labels,
        multiturn=multiturn,
        repeat_user_instruction=repeat_user_instruction,
    )

    system_message = load_system_message(None)
    system_element = SystemMessage(system_message)
    init_query     = InitQuery()

    tools_loop = ToolsExecutionLoop(
        elements=[ToolsExecutor(tool_result_to_str), fuse_llm],
        max_iters=max_iters,
    )

    pipeline = AgentPipeline(
        elements=[system_element, init_query, fuse_llm, tools_loop]
    )
    pipeline.name = model_name
    return pipeline