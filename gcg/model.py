# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import logging
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from jaxtyping import Float

from gcg.eval_input import EvalInput
from gcg.types import BatchTokenIds, PrefixCache, TokenIds
from gcg.utils import (
    Message,
    SuffixManager,
    batchify_kv_cache,
    build_prompt,
    get_prefix_cache,
)
from config import DELIMITERS
logger = logging.getLogger(__name__)

Device = int | str | torch.device
Devices = list[Device] | tuple[Device]
BatchLoss = Float[torch.Tensor, "batch_size"]
BatchLogits = Float[torch.Tensor, "batch_size seq_len vocab_size"]


@dataclasses.dataclass
class LossOutput:
    """Loss output from model."""

    losses: BatchLoss
    logits: BatchLogits | None = None
    texts: List[str] | None = None
    num_queries: int | None = None
    num_tokens: int | None = None


class TransformersModel:
    """Model builder for HuggingFace Transformers model.

    `model` should be in the format model_name@checkpoint_path.

    Call with a list of `Message` objects to generate a response.
    """

    supports_system_message = True
    available_peft = ("none", "noembed", "lora")

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        stream: bool = False,
        top_p: float = 1.0,
        max_tokens: int = 512,
        stop=None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        model: transformers.AutoModelForCausalLM | None = None,
        tokenizer: transformers.AutoTokenizer | None = None,
        suffix_manager: SuffixManager | None = None,
        devices: Device | Devices | None = None,
        system_message: str | None = None,
        dtype: str = "float32",
        pass_expert_labels: bool = False,
        delm_ids: Tuple[TokenIds | None, TokenIds | None, TokenIds | None] = (None, None, None),
        num_labels: int = 3
    ):
        model_name, checkpoint_path = model_name.split("@")
        self.model_name = model_name

        # Generation parameters
        self.checkpoint_path = os.path.expanduser(checkpoint_path)
        self.temperature = temperature
        self.stream = stream
        self.top_p = top_p
        self.max_tokens = max_tokens
        self._stop = stop
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.suffix_manager = suffix_manager
        self.system_message = system_message
        self._dtype = dtype
        if self._dtype not in ("float32", "float16", "bfloat16", "int8"):
            raise ValueError(f"Unknown dtype: {self._dtype}!")

        # Parse devices
        if devices is None:
            devices = ["cuda"]
        elif isinstance(devices, Device):
            devices = [devices]
        self.device = model.device if model is not None else devices[0]

        self._use_mixed_precision = False

        logger.info("Model is specified and already initialized.")
        self.model = model
        assert tokenizer is not None, "tokenizer must be provided if model is provided."
        self.tokenizer = tokenizer

        # ==================== Deal with multi-GPU loading =================== #
        if len(devices) > 1:
            logger.info(
                "%d devices (%s) are specified. Using DataParallel...",
                len(devices),
                devices,
            )
            self.model = torch.nn.DataParallel(self.model, device_ids=devices)
            # Should be fine to have generate run on rank 0 only
            self.model.generate = self.model.module.generate
            embed_layer = self.model.module.get_input_embeddings()
            self.embed_layer = torch.nn.DataParallel(embed_layer, device_ids=devices)

            def get_input_embeddings():
                return self.embed_layer

            self.model.get_input_embeddings = get_input_embeddings
            self.embed_weights = self.embed_layer.module.weight.t().detach()
        else:
            self.embed_layer = self.model.get_input_embeddings()
            self.embed_weights = self.embed_layer.weight.t().detach()
        self.embed_layer.requires_grad_(False)

        # Dictionary containing batched prefix cache (key is batch size)
        self._batch_prefix_cache: dict[int, PrefixCache] = {}
        # Original unbatched prefix cache
        self.prefix_cache: PrefixCache | None = None
        self.prefix_cache_inst_states: PrefixCache | None = None
        self.num_fixed_tokens: int = 0
        self.default_eval_input: EvalInput | None = None
        self.pass_expert_labels = pass_expert_labels
        self.model.eval()

        self.delm_ids = delm_ids
        self.num_labels = num_labels

    def __call__(
        self,
        inputs: List[Message] | list[str] | torch.Tensor | None = None,
        api_key: str = None,
    ):
        if isinstance(inputs[0], Message):
            # Turn messages into strings
            inputs = [build_prompt(inputs, self.model_name)]
        if isinstance(inputs[0], str):
            # Turn strings to token ids
            model_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True)
            #index = toks.index(25782) if 25782 in toks else -1; toks = [x for x in toks if x != 25782]
            #if index != -1: toks.insert(index, 758); toks.insert(index+1, 271)
        else:
            # Assume inputs are token ids
            model_inputs = {
                "input_ids": inputs,
                "attention_mask": torch.ones_like(inputs, dtype=torch.long),
            }
        model_inputs["input_ids"] = model_inputs["input_ids"].to(self.device, non_blocking=True)
        model_inputs["attention_mask"] = model_inputs["attention_mask"].to(
            self.device, non_blocking=True
        )
        prompt_len = model_inputs["attention_mask"].sum(dim=1)
        output = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        response = self.tokenizer.decode(
            output[0][prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [response]

    def _get_batch_prefix_cache(self, batch_size: int) -> PrefixCache:
        if self.prefix_cache is None:
            raise RuntimeError("Prefix cache has not been set!")
        if batch_size not in self._batch_prefix_cache:
            self._batch_prefix_cache[batch_size] = batchify_kv_cache(self.prefix_cache, batch_size)
        return self._batch_prefix_cache[batch_size]

    def set_prefix_cache(self, messages: list[Message]) -> None:
        prefix_cache, num_fixed_tokens = get_prefix_cache(
            self.suffix_manager, self.model, self.tokenizer, messages,
            self.pass_expert_labels, self.delm_ids[0], self.delm_ids[1], self.delm_ids[2],
            self.num_labels
        )
        if isinstance(prefix_cache, Tuple) and len(prefix_cache) == 2:
            self.prefix_cache, self.num_fixed_tokens = prefix_cache[0], num_fixed_tokens
            self.prefix_cache_inst_states = prefix_cache[1]
        else:
            self.prefix_cache, self.num_fixed_tokens = prefix_cache, num_fixed_tokens
        # Reset batched prefix cache
        self._batch_prefix_cache = {}

    def filter_suffixes(
        self,
        suffix_ids: BatchTokenIds | None = None,
        suffix: list[str] | None = None,
        skipped_suffixes: set | None = None,
    ) -> torch.Tensor:
        """Filter suffixes using all models.

        Args:
            suffix_ids: Suffix ids to filter. Defaults to None.
            suffix: Suffix strings to filter. Defaults to None.
            skipped_suffixes: Set of suffixes to skip. Defaults to None.

        Returns:
            Boolean filter of suffixes to keep.
        """
        _, orig_len = suffix_ids.shape
        device = suffix_ids.device
        assert (suffix_ids is not None) ^ (
            suffix is not None
        ), "Either suffix_ids OR suffix must be provided but not both!"
        if suffix is None:
            decoded = self.tokenizer.batch_decode(
                suffix_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            encoded = self.tokenizer(
                decoded,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
            ).input_ids.to(device)
            # Filter out suffixes that do not tokenize back to the same ids
            if self.tokenizer.padding_side == "left":
                filter_cond = torch.all(encoded[:, -orig_len:] == suffix_ids, dim=1)
            else:
                filter_cond = torch.all(encoded[:, :orig_len] == suffix_ids, dim=1)
            # Count number of non-pad tokens
            # pad_token_id = self._tokenizer.pad_token_id
            # filter_cond = (encoded != pad_token_id).sum(1) == orig_len
        else:
            encoded = self.tokenizer(
                suffix,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
            ).input_ids.to(device)
            decoded = self.tokenizer.batch_decode(
                encoded,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            filter_cond = [s == d for s, d in zip(suffix, decoded)]
            filter_cond = torch.tensor(filter_cond, device=device, dtype=torch.bool)

        if skipped_suffixes is not None:
            # Skip seen/visited suffixes
            is_kept = [suffix not in skipped_suffixes for suffix in decoded]
            is_kept = torch.tensor(is_kept, device=device, dtype=torch.bool)
        else:
            # No skip
            is_kept = torch.ones(len(decoded), device=device, dtype=torch.bool)

        filter_cond &= is_kept
        return filter_cond

    def compute_suffix_loss(
        self,
        eval_input: EvalInput,
        batch_size: int | None = None,
        temperature: float = 1.0,
        max_target_len: int | None = None,
        **kwargs,
    ) -> LossOutput:
        """Compute loss given multiple suffixes.

        Args:
            eval_input: Input to evaluate. Must be EvalInput.
            batch_size: Optional batch size. Defaults to None (use all samples).

        Returns:
            LossOutput object.
        """
        _ = kwargs  # Unused
        suffix_ids = eval_input.suffix_ids
        dynamic_input_ids = eval_input.dynamic_input_ids
        targets = eval_input.target_ids
        optim_slice = eval_input.optim_slice
        loss_slice = eval_input.loss_slice
        orig_device = suffix_ids.device
        expert_labels = eval_input.expert_labels
        device = self.device

        if max_target_len is not None:
            # Adjust loss_slice, targets, and input_ids according to
            # most max_target_len
            loss_slice = slice(
                loss_slice.start,
                min(loss_slice.stop, loss_slice.start + max_target_len),
            )
            if targets.ndim == 1:
                targets = targets[:max_target_len]
            else:
                targets = targets[:, :max_target_len]
            dynamic_input_ids = dynamic_input_ids[: loss_slice.stop + 1]
            expert_labels     = expert_labels[: loss_slice.stop + 1]

        # Determine batch size and number of batches
        num_samples = len(suffix_ids)
        if batch_size is None:
            batch_size = num_samples
        else:
            batch_size = min(batch_size, num_samples)
        num_batches = int(np.ceil(num_samples / batch_size))
        # Device placement BEFORE batch loop. This should be fine since inputs
        # don't take much memory anyway.
        dynamic_input_ids       = dynamic_input_ids.to(device)
        batch_dynamic_input_ids = dynamic_input_ids.repeat(batch_size, 1)
        expert_labels       = expert_labels.to(device)
        batch_expert_labels = expert_labels.repeat(batch_size, 1)

        # Expand and repeat batch dimension
        if targets.ndim == 1:
            targets = targets.unsqueeze(0)
        if targets.shape[0] == 1:
            targets = targets.repeat(num_samples, 1)
        assert targets.ndim in (2, 3), targets.shape
        assert targets.shape[0] == num_samples, targets.shape

        loss_list = []
        logits_list = []
        for i in range(num_batches):
            # Update suffixes
            batch_suffix_ids = suffix_ids[i * batch_size : (i + 1) * batch_size]
            batch_targets    = targets[i * batch_size : (i + 1) * batch_size]

            batch_suffix_ids = batch_suffix_ids.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            bs = len(batch_targets)

            batch_dynamic_input_ids[:bs, optim_slice] = batch_suffix_ids
            logits, loss, _, loss_slice = self._compute_loss(
                batch_dynamic_input_ids,
                batch_expert_labels,
                batch_targets,
                loss_slice,
                num_samples=bs,
                temperature=temperature,
            )
            loss_list.append(loss)
            logits_list.append(logits)
        loss = torch.cat(loss_list, dim=0).to(orig_device, non_blocking=True)
        logits = torch.cat(logits_list, dim=0).to(orig_device, non_blocking=True)

        assert loss.shape == (num_samples,), loss.shape
        logits_shape = (
            num_samples,
            loss_slice.stop - loss_slice.start,
            len(self.tokenizer),
        )
        assert logits.shape == logits_shape, logits.shape
        return LossOutput(losses=loss, logits=logits, num_queries=num_samples)

    def _compute_loss(
        self,
        batch_input_ids: BatchTokenIds,
        batch_expert_labels: torch.Tensor,
        batch_targets: torch.Tensor,
        loss_slice: slice | torch.Tensor,
        num_samples: int | None = None,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        num_samples = num_samples or len(batch_input_ids)
        input_embeds = self.embed_layer(batch_input_ids)

        # logits: [batch_size, seq_len, vocab_size]
        if self.pass_expert_labels:
            if self.prefix_cache_inst_states is not None:
                logits = self.model(
                    inputs_embeds=input_embeds,
                    expert_labels=batch_expert_labels,
                    past_key_values=self._get_batch_prefix_cache(len(batch_input_ids)),
                    past_inst_hidden_states = self.prefix_cache_inst_states,
                    use_cache=True,
                ).logits[:num_samples]
            else:
                logits = self.model(
                    inputs_embeds=input_embeds,
                    expert_labels=batch_expert_labels,
                    past_key_values=self._get_batch_prefix_cache(len(batch_input_ids)),
                    use_cache=True,
                ).logits[:num_samples]
        else:
            logits = self.model(
                inputs_embeds=input_embeds,
                past_key_values=self._get_batch_prefix_cache(len(batch_input_ids)),
                use_cache=True,
            ).logits[:num_samples]

        logits = logits / temperature

        if isinstance(loss_slice, slice):
            loss_logits = logits[:, loss_slice]
        else:
            loss_logits = logits.gather(1, loss_slice)

        if batch_targets.dtype == torch.long:
            # Hard-label target usually used for computing loss on target
            # loss_logits: [batch_size, vocab_size, loss_len]
            loss = F.cross_entropy(
                loss_logits.permute(0, 2, 1), batch_targets, reduction="none"
            ).mean(dim=1)
        else:
            # Soft-label target usually used for training proxy model
            loss = F.kl_div(
                loss_logits.log_softmax(dim=-1),
                batch_targets / temperature,
                reduction="none",
            )
            loss = loss.sum(dim=-1).mean(dim=1)
        return loss_logits, loss, logits, loss_slice

    @torch.no_grad()
    def compute_grad(
        self,
        eval_input: EvalInput,
        temperature: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """Compute gradients w.r.t. `input_ids` tokens at `optim_slice`."""
        _ = kwargs  # Unused
        input_ids = eval_input.dynamic_input_ids
        expert_labels = eval_input.expert_labels
        target_ids  = eval_input.target_ids
        optim_slice = eval_input.optim_slice
        loss_slice = eval_input.loss_slice

        orig_device = input_ids.device
        input_ids     = input_ids.to(self.device, non_blocking=True)
        expert_labels = expert_labels.to(self.device, non_blocking=True)
        target_ids    = target_ids.to(self.device, non_blocking=True)
        if target_ids.ndim == 2:
            target_ids.squeeze_(0)

        input_embeds = self.embed_layer(input_ids)
        input_embeds.unsqueeze_(0)
        input_embeds.requires_grad_(True)
        expert_labels.unsqueeze_(0)

        with torch.enable_grad():
            # Forward pass
            if self.pass_expert_labels:
                if self.prefix_cache_inst_states is not None:
                    logits = self.model(
                        inputs_embeds=input_embeds,
                        expert_labels=expert_labels,
                        past_key_values=self._get_batch_prefix_cache(len(input_embeds)),
                        past_inst_hidden_states=self.prefix_cache_inst_states,
                        use_cache=True,
                    ).logits
                else:
                    logits = self.model(
                        inputs_embeds=input_embeds,
                        expert_labels=expert_labels,
                        past_key_values=self._get_batch_prefix_cache(len(input_embeds)),
                        use_cache=True,
                    ).logits
            else:
                logits = self.model(
                    inputs_embeds=input_embeds,
                    past_key_values=self._get_batch_prefix_cache(len(input_embeds)),
                    use_cache=True,
                ).logits

            # Compute loss and gradients
            loss_logits = logits[:, loss_slice].squeeze(0)
            loss = F.cross_entropy(loss_logits / temperature, target_ids)
            embed_grads = torch.autograd.grad(outputs=[loss], inputs=[input_embeds])[0]

        embed_grads.detach_()
        embed_grads = embed_grads[0, optim_slice]
        token_grads = embed_grads @ self.embed_weights
        token_grads /= token_grads.norm(dim=-1, keepdim=True)
        token_grads = token_grads.to(orig_device, non_blocking=True)

        assert token_grads.shape == (
            optim_slice.stop - optim_slice.start,
            len(self.tokenizer),
        ), token_grads.shape
        return token_grads
