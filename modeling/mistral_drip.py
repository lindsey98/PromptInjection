import transformers
import torch
from transformers import MistralConfig
from modeling.llama_drip import (
    _get_last_indices,
    compute_expert_labels_from_input_ids,
    CausalLMFuseOutputWithPast,
    set_delimiter_ids_in_config,
    _get_first_indices
)
from transformers.utils import logging
from typing import Union, Optional, List, Dict, Any
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
import inspect

logger = logging.get_logger(__name__)


class MistralDRIPConfig(MistralConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_experts       = kwargs.get('num_experts', 4)
        self.residual          = kwargs.get('residual', True)
        self.bit_flip          = kwargs.get('bit_flip', True)
        self.data_delm_ids     = kwargs.get('data_delm_ids', None)      # List[int]
        self.response_delm_ids = kwargs.get('response_delm_ids', None)  # List[int]
        self.inst_delm_ids     = kwargs.get('inst_delm_ids', None)      # List[int] | None
        self.num_labels        = kwargs.get('num_labels', 4)
        self.instruct_label    = kwargs.get('instruct_label',  0 if self.num_labels == 3 else 1)
        self.data_label        = kwargs.get('data_label',      1 if self.num_labels == 3 else 2)
        self.response_label    = kwargs.get('response_label',  self.num_labels - 1)


class MistralModel(transformers.MistralModel):
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.config = config
        self.deinstruction_shift = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        # Do NOT override self.layers — super().__init__() already builds it correctly
        self.instruct_label  = config.instruct_label
        self.data_label      = config.data_label
        self.residual_weight = nn.Parameter(torch.tensor([-1.0986]))
        self.shift_tap       = torch.nn.Identity()
        self.post_init()

    def _has_delimiter_config(self) -> bool:
        return (
            getattr(self.config, 'data_delm_ids', None) is not None
            and getattr(self.config, 'response_delm_ids', None) is not None
        )

    def _auto_expert_labels(
        self,
        input_ids: Optional[torch.LongTensor],
    ) -> Optional[torch.LongTensor]:
        if input_ids is None or not self._has_delimiter_config():
            return None
        return compute_expert_labels_from_input_ids(
            input_ids=input_ids,
            data_delm_ids=self.config.data_delm_ids,
            response_delm_ids=self.config.response_delm_ids,
            inst_delm_ids=getattr(self.config, 'inst_delm_ids', None),
            num_labels=getattr(self.config, 'num_labels', 3),
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        expert_labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = kwargs.get("output_attentions", False)
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if isinstance(past_key_values, (list, tuple)):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_self_attns = [] if output_attentions else None

        # Auto-compute expert_labels if not provided externally
        if expert_labels is None:
            expert_labels = self._auto_expert_labels(input_ids)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if self.config.bit_flip and layer_idx == 0:
                if expert_labels is not None:
                    inst_mask_2d = (expert_labels == self.data_label)  # B, L
                    inst_mask_3d = inst_mask_2d.unsqueeze(-1).expand_as(hidden_states)
                    shifts = self.deinstruction_shift(hidden_states) # why having different shifts for different samples?
                    hidden_states  = torch.where(
                        inst_mask_3d,
                        shifts + hidden_states,
                        hidden_states
                    )
            hidden_states = self.shift_tap(hidden_states)
            if output_attentions:
                hidden_states, attn_weights = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                all_self_attns.append(attn_weights)
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            attentions=tuple(all_self_attns) if output_attentions else None,
        )


class MistralForCausalLMDRIP(transformers.MistralForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: MistralDRIPConfig):
        super().__init__(config)
        del self.model
        self.model           = MistralModel(config)
        self.vocab_size      = config.vocab_size
        self.hidden_size     = config.hidden_size
        self.residual_weight = nn.Parameter(torch.tensor([0.0]))
        self.response_label  = config.response_label
        self.instruct_label  = config.instruct_label
        self.final_tap       = torch.nn.Identity()
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        expert_labels: torch.LongTensor = None,          # optional: auto-computed if config has delm ids
        past_inst_hidden_states: Optional[Union[Cache, torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMFuseOutputWithPast:

        # Auto-compute only if not explicitly provided (GCG provides explicitly)
        if expert_labels is None and self.model._has_delimiter_config() and input_ids is not None:
            expert_labels = self.model._auto_expert_labels(input_ids)

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            expert_labels=expert_labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        hidden_states = self.final_tap(hidden_states)

        if expert_labels is not None:
            batch_size, length, hidden_size = hidden_states.shape
            is_generation = (length == 1) and (past_inst_hidden_states is not None) # is it in newly-generated-token phase?
            if (not is_generation) and (self.config.residual): # for newly generated token, do not add the instruction semantic
                batch_idx = torch.arange(batch_size, device=hidden_states.device)  # (B,)

                if past_inst_hidden_states is not None:
                    last_inst = past_inst_hidden_states.to(hidden_states.device)
                else:
                    last_inst_indices = _get_last_indices(self.instruct_label, expert_labels)

                    # Prevent having -1
                    safe_indices = last_inst_indices.clamp(min=0)
                    last_inst = hidden_states[batch_idx, safe_indices]
                    invalid_mask = (last_inst_indices == -1).unsqueeze(-1)  # (B, 1)
                    last_inst = last_inst.masked_fill(invalid_mask, 0.0)

                    past_inst_hidden_states = last_inst.clone().detach()

                    # First token of response region
                first_resp_indices = _get_first_indices(self.response_label, expert_labels)
                # Last token of the response delimiter = first + (delimiter_length - 1)
                resp_delm_len = len(self.config.response_delm_ids)  # stored in config
                end_indices = (first_resp_indices + resp_delm_len - 1).clamp(max=expert_labels.shape[1] - 1)
                end_state = hidden_states[batch_idx, end_indices]  # (B, H)

                # Add last instruction token's semantic as a residual connection to the 1st response token
                fuse_factor = torch.sigmoid(self.residual_weight)  # fixme: factor between 0 and 1
                fused_states = torch.lerp(end_state, last_inst, fuse_factor)  # fixme: only last layer? (B, H)

                mask2d = torch.zeros(batch_size, length, dtype=torch.bool, device=hidden_states.device)
                mask2d[batch_idx, end_indices] = True
                mask3d = mask2d.unsqueeze(-1)  # broadcast to (B, L, 1)

                fused_states = fused_states.unsqueeze(1).expand(-1, length,
                                                                -1)  # expand mixed_states (B, H) → (B, L, H)
                hidden_states = torch.where(mask3d, fused_states, hidden_states)

        hidden_states = hidden_states.to(self.lm_head.weight.dtype)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMFuseOutputWithPast(
            loss=loss,
            logits=logits,
            past_inst_hidden_states=past_inst_hidden_states,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs
        )

        # expert_labels: accepted from caller for backwards-compat (GCG inputs_embeds path).
        # When delimiter ids are in config, NOT needed — model auto-computes in forward().
        expert_labels = kwargs.pop("expert_labels", None)
        if expert_labels is not None:
            position_ids_key = "decoder_position_ids" if self.config.is_encoder_decoder else "position_ids"
            if (
                attention_mask is not None
                and kwargs.get(position_ids_key) is None
                and position_ids_key in set(inspect.signature(self.forward).parameters.keys())
            ):
                expert_labels = expert_labels[:, -model_inputs["input_ids"].shape[1]:]
            model_inputs["expert_labels"] = expert_labels

        past_inst_hidden_states = kwargs.get("past_inst_hidden_states", None)
        if past_inst_hidden_states is not None:
            model_inputs["past_inst_hidden_states"] = past_inst_hidden_states
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens,
        )
        if hasattr(outputs, "past_inst_hidden_states"):
            model_kwargs["past_inst_hidden_states"] = outputs.past_inst_hidden_states
        return model_kwargs