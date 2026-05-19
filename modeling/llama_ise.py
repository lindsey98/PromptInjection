import transformers
import torch
from transformers import LlamaConfig
from modeling.llama_drip import (
    compute_expert_labels_from_input_ids,
    set_delimiter_ids_in_config,
)
from transformers.utils import logging
from typing import Union, Optional, List
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_masks_for_generate
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
import inspect

logger = logging.get_logger(__name__)


class LlamaISEConfig(LlamaConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_input_shifts        = kwargs.get('apply_input_shifts', True)
        self.apply_intermediate_shifts = kwargs.get('apply_intermediate_shifts', False)
        self.num_blocks_with_shifts    = kwargs.get('num_blocks_with_shifts', 1)
        self.num_experts               = kwargs.get('num_experts', 3)
        self.d_gap                     = kwargs.get('d_gap', 512)
        self.bit_flip                  = kwargs.get('bit_flip', True)
        # Delimiter token IDs for runtime expert_label computation
        self.data_delm_ids     = kwargs.get('data_delm_ids', None)      # List[int]
        self.response_delm_ids = kwargs.get('response_delm_ids', None)  # List[int]
        self.inst_delm_ids     = kwargs.get('inst_delm_ids', None)      # List[int] | None
        self.num_labels        = kwargs.get('num_labels', 3)
        self.instruct_label    = kwargs.get('instruct_label',  0 if self.num_labels == 3 else 1)
        self.data_label        = kwargs.get('data_label',      1 if self.num_labels == 3 else 2)
        self.response_label    = kwargs.get('response_label',  self.num_labels - 1)
        assert self.num_experts > 0, "num_experts must be > 0"


def _resolve_expert_labels(
    input_ids: Optional[torch.LongTensor],
    expert_labels: Optional[torch.LongTensor],
    config,
    require: bool = True,
) -> torch.LongTensor:
    """
    Return expert_labels, auto-computing from input_ids if not provided.
    Raises ValueError if labels cannot be determined and require=True.
    """
    if expert_labels is not None:
        return expert_labels
    if (
        input_ids is not None
        and getattr(config, 'data_delm_ids', None) is not None
        and getattr(config, 'response_delm_ids', None) is not None
    ):
        return compute_expert_labels_from_input_ids(
            input_ids=input_ids,
            data_delm_ids=config.data_delm_ids,
            response_delm_ids=config.response_delm_ids,
            inst_delm_ids=getattr(config, 'inst_delm_ids', None),
            num_labels=getattr(config, 'num_labels', 3),
        )
    if require:
        raise ValueError(
            "expert_labels must be provided (or delimiter ids must be set in config "
            "so they can be auto-computed from input_ids)."
        )
    return None


class LlamaModel(transformers.LlamaModel):
    def __init__(self, config: LlamaISEConfig):
        super().__init__(config)
        self.apply_input_shifts        = config.apply_input_shifts
        self.input_shifts              = nn.Embedding(config.num_experts, config.hidden_size)
        self.num_blocks_with_shifts    = config.num_blocks_with_shifts
        self.shift_tap                 = torch.nn.Identity()
        self.post_init()
        self.custom_initialize()

    def custom_initialize(self):
        nn.init.normal_(self.input_shifts.weight, mean=0, std=0.001)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        expert_labels: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states       = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if self.config.bit_flip and self.apply_input_shifts:
            shifts       = self.input_shifts(expert_labels)
            hidden_states = hidden_states + shifts
        hidden_states = self.shift_tap(hidden_states)

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=hidden_states,
            past_key_values=past_key_values,
        )


class LlamaForCausalLMISE(transformers.LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: LlamaISEConfig):
        super().__init__(config)
        del self.model
        self.model      = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.final_tap  = torch.nn.Identity()
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        expert_labels: torch.LongTensor = None,    # optional: auto-computed if config has delm ids
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:

        # Auto-compute before passing to model (also needed for inputs_embeds path
        # since model.forward will fail with require=True if labels are absent)
        expert_labels = _resolve_expert_labels(input_ids, expert_labels, self.config, require=True)

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

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
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
        expert_labels = kwargs.pop("expert_labels", None)

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        # Align expert_labels with the (possibly trimmed) input_ids
        actual_input_ids = model_inputs.get("input_ids")
        if actual_input_ids is None:
            return model_inputs
        cur_len = actual_input_ids.shape[1]
        batch_size = actual_input_ids.shape[0]

        is_decode_step = (
                past_key_values is not None
                and past_key_values.get_seq_length() > 0
        )

        if is_decode_step:
            # New tokens are model output -> they belong to the response section.
            new_labels = torch.full(
                (batch_size, cur_len),
                self.config.response_label,
                dtype=torch.long,
                device=actual_input_ids.device,
            )
            if expert_labels is not None:
                # Concatenate prompt labels with response labels for full length;
                # but only the last `cur_len` are forwarded to the model.
                # Since this is decode step, just send the new ones.
                model_inputs["expert_labels"] = new_labels
            else:
                model_inputs["expert_labels"] = new_labels
        else:
            # Prefill (or no cache): align expert_labels to current input_ids
            if expert_labels is not None:
                if expert_labels.shape[1] > cur_len:
                    expert_labels = expert_labels[:, -cur_len:]
                elif expert_labels.shape[1] < cur_len:
                    # Pad missing positions with response_label as a fallback
                    pad = torch.full(
                        (batch_size, cur_len - expert_labels.shape[1]),
                        self.config.response_label,
                        dtype=torch.long,
                        device=actual_input_ids.device,
                    )
                    expert_labels = torch.cat([expert_labels, pad], dim=1)
                model_inputs["expert_labels"] = expert_labels
            # If expert_labels not provided, _resolve_expert_labels in forward()
            # will auto-compute from input_ids using delimiter ids in config.

        return model_inputs


class LlamaModelV2(transformers.LlamaModel):
    """Positional gap variant: separates instruction/data by shifting position IDs."""

    def __init__(self, config: LlamaISEConfig):
        super().__init__(config)
        self.d_gap = config.d_gap
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        expert_labels: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # Auto-compute expert_labels if not provided
        expert_labels = _resolve_expert_labels(input_ids, expert_labels, self.config, require=True)

        # Add gap in positional embedding to separate instruction vs data tokens
        position_ids        = position_ids + self.d_gap * expert_labels.to(hidden_states.dtype)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=hidden_states,
            past_key_values=past_key_values,
        )


class LlamaForCausalLMPFT(LlamaForCausalLMISE):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LlamaISEConfig):
        super().__init__(config)
        del self.model
        self.model      = LlamaModelV2(config)
        self.vocab_size = config.vocab_size
        self.post_init()