
import transformers
import torch
from transformers import MistralConfig
from .llama_drip import _get_last_indices, CausalLMFuseOutputWithPast
from transformers.utils import logging
from typing import Union, Optional, Tuple, List, Dict, Any
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, DynamicCache, StaticCache
logger = logging.get_logger(__name__)

class MistralDRIPConfig(MistralConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_experts = kwargs.get('num_experts', 3)
        self.residual = kwargs.get('residual', True)
        self.bit_flip = kwargs.get('bit_flip', True)


class MistralModel(transformers.MistralModel):
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.config = config
        self.deinstruction_shift = nn.Linear(config.hidden_size, config.hidden_size, bias=True) # rotation+scale+shift
        self.instruct_label = 0
        self.data_label = 1
        self.residual_weight = nn.Parameter(torch.tensor([-1.0986]))  # 0.5 weight assigned to the last instruction token
        self.shift_tap = torch.nn.Identity()  # <— dummy tap point
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        expert_labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, use_cache, output_attentions
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            # one-bit flip for First attention's input
            if self.config.bit_flip and layer_idx == 0:
            # if self.config.bit_flip and layer_idx == self.config.num_hidden_layers-1:
                inst_mask_2d = (expert_labels == self.data_label)  # B, L
                inst_mask_3d = inst_mask_2d.unsqueeze(-1).expand_as(hidden_states)
                shifts = self.deinstruction_shift(hidden_states) # why having different shifts for different samples?
                hidden_states  = torch.where(
                    inst_mask_3d,
                    shifts + hidden_states,
                    hidden_states
                )
                hidden_states = self.shift_tap(hidden_states)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MistralForCausalLMDRIP(transformers.MistralForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: MistralDRIPConfig):
        super().__init__(config)
        del self.model
        self.model      = MistralModel(config)
        self.vocab_size = config.vocab_size
        self.vocab_size      = config.vocab_size
        self.hidden_size     = config.hidden_size
        self.residual_weight = nn.Parameter(torch.tensor([0.01])) # 0.5 weight assigned to the last instruction token
        self.response_label  = 2
        self.instruct_label  = 0
        self.final_tap = torch.nn.Identity()  # <— dummy tap point
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        expert_labels: torch.LongTensor = None,
        past_inst_hidden_states: Optional[Union[Cache, torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> CausalLMFuseOutputWithPast:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            expert_labels=expert_labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        hidden_states = self.final_tap(hidden_states)
        if expert_labels is not None:
            batch_size, length, hidden_size = hidden_states.shape
            is_generation = (length == 1) and (past_inst_hidden_states is not None) # is it in newly-generated-token phase?
            if (not is_generation) and (self.config.residual): # for newly generated token, do not add the instruction semantic
                batch_idx = torch.arange(batch_size, device=hidden_states.device)  # (B,)

                if past_inst_hidden_states is not None: # load cached last instruction hidden states
                    last_inst = past_inst_hidden_states.to(hidden_states.device)
                else:
                    last_inst_indices = _get_last_indices(self.instruct_label, expert_labels) # (B,)
                    last_inst = hidden_states[batch_idx, last_inst_indices]  # (B, H)
                    past_inst_hidden_states = last_inst.clone().detach().cpu()

                end_indices = _get_last_indices(self.response_label, expert_labels) # (B,)
                end_state   = hidden_states[batch_idx, end_indices]  # (B, H)

                # Add last instruction token's semantic as a residual connection to the 1st response token
                fuse_factor  = torch.sigmoid(self.residual_weight)  # fixme: factor between 0 and 1
                fused_states = torch.lerp(end_state, last_inst, fuse_factor) # fixme: only last layer? (B, H)

                mask2d = torch.zeros(batch_size, length, dtype=torch.bool, device=hidden_states.device)
                mask2d[batch_idx, end_indices] = True
                mask3d = mask2d.unsqueeze(-1) # broadcast to (B, L, 1)

                fused_states = fused_states.unsqueeze(1).expand(-1, length, -1) # expand mixed_states (B, H) → (B, L, H)
                hidden_states = torch.where(mask3d, fused_states, hidden_states)

        hidden_states = hidden_states.to(self.lm_head.weight.dtype)
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

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
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        expert_labels = kwargs.pop("expert_labels", None)

        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)
                if expert_labels is not None:
                    expert_labels = expert_labels[:,-input_ids.shape[1]:]  # the expert label will inherit the last token's expert label

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}
        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min
            if attention_mask is not None and attention_mask.dim() == 4:
                # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
                pass
            else:
                target_length=past_key_values.get_max_length()
                causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype,
                                         device=device)
                if sequence_length != 1:
                    causal_mask = torch.triu(causal_mask, diagonal=1)
                causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
                causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
                if attention_mask is not None:
                    causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                    mask_length = attention_mask.shape[-1]
                    padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                    padding_mask = padding_mask == 0
                    causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                        padding_mask, min_dtype
                    )
                attention_mask = causal_mask

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
        model_inputs.pop("labels", None)
        if expert_labels is not None:
            model_inputs["expert_labels"] = expert_labels

        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:

        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder,
            num_new_tokens,
        )
        if hasattr(outputs, "past_inst_hidden_states"):
            model_kwargs["past_inst_hidden_states"] = outputs.past_inst_hidden_states # cache the last instruction token hidden state
        return model_kwargs


