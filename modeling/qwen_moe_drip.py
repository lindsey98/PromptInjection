import transformers
import torch
from transformers import Qwen3Config, Qwen3MoeConfig
from modeling.llama_drip import (
    _get_last_indices,
    compute_expert_labels_from_input_ids,
    _get_first_indices,
    _get_last_segment_start,
)
from transformers.utils import logging
from typing import Union, Optional, List, Dict, Any
from torch import nn
from transformers.modeling_outputs import MoeModelOutputWithPast, MoeCausalLMOutputWithPast
from transformers.models.qwen3_moe.modeling_qwen3_moe import load_balancing_loss_func
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
import inspect
from dataclasses import dataclass

logger = logging.get_logger(__name__)

@dataclass
class CausalLMMoEFuseOutputWithPast(MoeCausalLMOutputWithPast):
    past_inst_hidden_states: Optional[torch.FloatTensor] = None

class Qwen3MoeDRIPConfig(Qwen3MoeConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.residual          = kwargs.get('residual', True)
        # MoE backbones use residual fusion WITHOUT the de-instruction shift by
        # default (see README "Limitations"); set bit_flip=True to enable it.
        self.bit_flip          = kwargs.get('bit_flip', False)
        self.data_delm_ids     = kwargs.get('data_delm_ids', None)      # List[int]
        self.response_delm_ids = kwargs.get('response_delm_ids', None)  # List[int]
        self.inst_delm_ids     = kwargs.get('inst_delm_ids', None)      # List[int] | None
        self.num_labels        = kwargs.get('num_labels', 4)
        self.instruct_label    = kwargs.get('instruct_label',  0 if self.num_labels == 3 else 1)
        self.data_label        = kwargs.get('data_label',      1 if self.num_labels == 3 else 2)
        self.response_label    = kwargs.get('response_label',  self.num_labels - 1)


class Qwen3MoeModel(transformers.Qwen3MoeModel):
    def __init__(self, config: Qwen3MoeDRIPConfig):
        super().__init__(config)
        self.config = config
        self.deinstruction_shift = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        nn.init.zeros_(self.deinstruction_shift.weight)
        nn.init.zeros_(self.deinstruction_shift.bias)
        self.instruct_label  = config.instruct_label
        self.data_label      = config.data_label
        self.shift_tap       = torch.nn.Identity()
        self._warned_empty_data = False  # one-time silent-segmentation guard
        self._router_logits_buffer = []
        self._hooks_attached = False
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
            num_labels=getattr(self.config, 'num_labels', 4),
        )

    def _attach_router_hooks(self):
        """Register forward hooks on every MoE sparse block to capture router_logits."""
        if self._hooks_attached:
            return

        def make_hook(layer_idx):
            def hook(module, inputs, output):
                # Qwen3MoeSparseMoeBlock returns (final_hidden_states, router_logits)
                if isinstance(output, tuple) and len(output) >= 2:
                    self._router_logits_buffer.append(output[1])

            return hook

        for i, layer in enumerate(self.layers):
            # In Qwen3-MoE, layer.mlp is Qwen3MoeSparseMoeBlock for MoE layers
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and "Sparse" in type(mlp).__name__:
                mlp.register_forward_hook(make_hook(i))
        self._hooks_attached = True

    def forward(
        self,
        input_ids=None,
        expert_labels=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=None,
        output_router_logits=None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_router_logits = (
            output_router_logits if output_router_logits is not None
            else self.config.output_router_logits
        )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

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

        if expert_labels is None:
            expert_labels = self._auto_expert_labels(input_ids)

        # Setup router_logits collection
        if output_router_logits:
            self._attach_router_hooks()
            self._router_logits_buffer = []   # reset per forward call

        for layer_idx, decoder_layer in enumerate(self.layers[:self.config.num_hidden_layers]):
            # De-instruction shift (disabled by default for MoE; gated by config.bit_flip).
            if self.config.bit_flip and layer_idx == 0 and expert_labels is not None:
                data_mask_2d = (expert_labels == self.data_label)
                if (not self._warned_empty_data) and hidden_states.shape[1] > 1 and not bool(data_mask_2d.any()):
                    logger.warning(
                        "DRIP: no tokens were labeled as data (data_label=%d); the de-instruction "
                        "shift is a no-op for this input. Check that the prompt's data delimiter "
                        "matches config.data_delm_ids.", self.data_label
                    )
                    self._warned_empty_data = True
                data_mask_3d = data_mask_2d.unsqueeze(-1).expand_as(hidden_states)
                shifts = self.deinstruction_shift(hidden_states)
                hidden_states = torch.where(data_mask_3d, shifts + hidden_states, hidden_states)

            hidden_states = self.shift_tap(hidden_states)
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        router_logits_out = tuple(self._router_logits_buffer) if output_router_logits else None

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            router_logits=router_logits_out,
        )


class Qwen3MoeForCausalLMDRIP(transformers.Qwen3MoeForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: Qwen3MoeDRIPConfig):
        super().__init__(config)
        del self.model
        self.model           = Qwen3MoeModel(config)
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
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMMoEFuseOutputWithPast:

        # Auto-compute only if not explicitly provided (GCG provides explicitly)
        if expert_labels is None and self.model._has_delimiter_config() and input_ids is not None:
            expert_labels = self.model._auto_expert_labels(input_ids)

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            expert_labels=expert_labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        hidden_states = self.final_tap(hidden_states)

        if expert_labels is not None:
            batch_size, length, hidden_size = hidden_states.shape
            is_generation = (length == 1) and (past_inst_hidden_states is not None)
            if (not is_generation) and (self.config.residual):
                batch_idx = torch.arange(batch_size, device=hidden_states.device)

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
                # first_resp_indices = _get_first_indices(self.response_label, expert_labels)
                first_resp_indices = _get_last_segment_start(self.response_label, expert_labels)
                # Last token of the response delimiter = first + (delimiter_length - 1)
                resp_delm_len = len(self.config.response_delm_ids)  # stored in config
                end_indices = (first_resp_indices + resp_delm_len - 1).clamp(max=expert_labels.shape[1] - 1)
                end_state   = hidden_states[batch_idx, end_indices]  # (B, H)

                # Add last instruction token's semantic as a residual connection to the 1st response token
                fuse_factor  = torch.sigmoid(self.residual_weight)  # fixme: factor between 0 and 1
                # in qwen_moe_drip.py, around line 238
                target_dtype = end_state.dtype  # whatever activation dtype is
                fused_states = torch.lerp(
                    end_state.to(target_dtype),
                    last_inst.to(target_dtype),
                    fuse_factor.to(target_dtype),
                )

                mask2d = torch.zeros(batch_size, length, dtype=torch.bool, device=hidden_states.device)
                mask2d[batch_idx, end_indices] = True
                mask3d = mask2d.unsqueeze(-1) # broadcast to (B, L, 1)

                fused_states = fused_states.unsqueeze(1).expand(-1, length, -1) # expand mixed_states (B, H) → (B, L, H)
                hidden_states = torch.where(mask3d, fused_states, hidden_states)

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return CausalLMMoEFuseOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_inst_hidden_states=past_inst_hidden_states,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
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
        outputs: CausalLMMoEFuseOutputWithPast,
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