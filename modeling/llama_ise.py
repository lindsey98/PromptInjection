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
        self.num_experts               = kwargs.get('num_experts', 4)
        self.d_gap                     = kwargs.get('d_gap', 512)
        self.bit_flip                  = kwargs.get('bit_flip', True)
        # Delimiter token IDs for runtime expert_label computation
        self.data_delm_ids     = kwargs.get('data_delm_ids', None)      # List[int]
        self.response_delm_ids = kwargs.get('response_delm_ids', None)  # List[int]
        self.inst_delm_ids     = kwargs.get('inst_delm_ids', None)      # List[int] | None
        self.num_labels        = kwargs.get('num_labels', 4)
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

        # Auto-compute expert_labels if not provided
        expert_labels = _resolve_expert_labels(input_ids, expert_labels, self.config, require=True)

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
        model_inputs          = {}
        model_inputs["cache_position"] = cache_position
        # expert_labels: still accepted explicitly for GCG / inputs_embeds path.
        # When delimiter ids are in config, auto-computed in forward() so not needed here.
        expert_labels = kwargs.pop("expert_labels", None)

        # 2. Generic cache-dependent input preparation
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            inputs_embeds, input_ids = self._cache_dependant_input_preparation(
                input_ids, inputs_embeds, cache_position
            )

        # 3. Prepare base model inputs
        input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        if not self.config.is_encoder_decoder:
            if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
                model_inputs[input_ids_key] = None
                model_inputs["inputs_embeds"] = inputs_embeds
            else:
                model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
                model_inputs["inputs_embeds"] = None
        else:
            model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)

        # 4. Create missing `position_ids` on the fly
        encoder_attention_mask = attention_mask if self.config.is_encoder_decoder else None
        attention_mask = (
            kwargs.pop("decoder_attention_mask", None) if self.config.is_encoder_decoder else attention_mask
        )
        attention_mask_key = "decoder_attention_mask" if self.config.is_encoder_decoder else "attention_mask"
        position_ids_key   = "decoder_position_ids"   if self.config.is_encoder_decoder else "position_ids"
        if (
            attention_mask is not None
            and kwargs.get(position_ids_key) is None
            and position_ids_key in set(inspect.signature(self.forward).parameters.keys())
        ):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs[position_ids_key] = position_ids
            if expert_labels is not None:
                expert_labels = expert_labels[:, -input_ids.shape[1]:]

        # 5. Slice model inputs if it's an input that should have the same length as `input_ids`
        for model_input_name in ["position_ids", "token_type_ids", "decoder_position_ids"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                if past_key_values is not None:
                    current_input_length = (
                        model_inputs["inputs_embeds"].shape[1]
                        if model_inputs.get("inputs_embeds") is not None
                        else model_inputs[input_ids_key].shape[1]
                    )
                    model_input = model_input[:, -current_input_length:]
                    model_input = model_input.clone(memory_format=torch.contiguous_format)
                model_inputs[model_input_name] = model_input

        # 6. Create 4D attention mask for compilable cache
        if (
            isinstance(past_key_values, Cache)
            and past_key_values.is_compileable
            and attention_mask is not None
            and attention_mask.ndim == 2
        ):
            if not self.config.is_encoder_decoder and model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
            else:
                batch_size, sequence_length = model_inputs[input_ids_key].shape[:2]

            base_model = getattr(self, self.base_model_prefix, self)
            decoder = base_model.get_decoder() if hasattr(base_model, "get_decoder") else None
            causal_mask_creation_function = getattr(
                base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
            )
            if causal_mask_creation_function is None and decoder is not None:
                causal_mask_creation_function = getattr(
                    decoder, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )

            if causal_mask_creation_function is None:
                token_type_ids = model_inputs.get("token_type_ids", None)
                position_ids   = model_inputs.get(position_ids_key, None)
                causal_mask_creation_function = getattr(self, "create_masks_for_generate", create_masks_for_generate)
                attention_mask = causal_mask_creation_function(
                    config=self.config,
                    input_embeds=torch.empty((batch_size, sequence_length), dtype=self.dtype),
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    token_type_ids=token_type_ids,
                )
            else:
                attention_mask = causal_mask_creation_function(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    dtype=self.dtype,
                    cache_position=cache_position,
                    batch_size=batch_size,
                    config=self.config,
                    past_key_values=past_key_values,
                )

        if attention_mask is not None:
            model_inputs[attention_mask_key] = attention_mask
        if encoder_attention_mask is not None:
            model_inputs["attention_mask"] = encoder_attention_mask

        # 7. Forward ALL kwargs that are uninitialized
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        # 8. Remove unexpected `generate` inputs
        model_inputs.pop("labels", None)
        if expert_labels is not None:
            model_inputs["expert_labels"] = expert_labels

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