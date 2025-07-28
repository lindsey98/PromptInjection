# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import transformers
import torch
from train import smart_tokenizer_and_embedding_resize, ModelArguments, DataArguments, TrainingArguments, AttackArguments, find_closest_match
from config import DEFAULT_TOKENS, SPECIAL_DELM_TOKENS, DELIMITERS
from transformers.utils import logging
from modeling.llama_instfuse import LlamaForCausalLMFuse, LlamaForCausalLMFuseV2, LlamaFuseConfig
from transformers import Trainer
import os
from data_generation.struq import jload, make_supervised_data_module
from transformers.trainer import PreTrainedModel, is_sagemaker_mp_enabled, Adafactor, is_bitsandbytes_available
from transformers.training_args import OptimizerNames
from typing import Optional, Tuple, Any, List
from torch import nn
from packaging import version
import importlib.metadata
logger = logging.get_logger(__name__)
os.environ["WANDB_WATCH"]="all" # log all parameters gradients


class MyTrainer(Trainer):
    @staticmethod
    def get_optimizer_cls_and_kwargs(
        args: TrainingArguments, model: Optional[PreTrainedModel] = None
    ) -> Tuple[Any, Any]:
        optim_args = {}
        if args.optim_args:
            for mapping in args.optim_args.replace(" ", "").split(","):
                key, value = mapping.split("=")
                optim_args[key] = value

        optimizer_kwargs = {}

        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }
        if args.optim == OptimizerNames.ADAFACTOR:
            optimizer_cls = Adafactor
            optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        elif args.optim == OptimizerNames.ADAMW_HF:
            from transformers.optimization import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
        elif args.optim in [OptimizerNames.ADAMW_TORCH, OptimizerNames.ADAMW_TORCH_FUSED]:
            from torch.optim import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
            if args.optim == OptimizerNames.ADAMW_TORCH_FUSED:
                optimizer_kwargs.update({"fused": True})
        elif args.optim == OptimizerNames.ADAMW_TORCH_XLA:
            try:
                from torch_xla.amp.syncfree import AdamW

                optimizer_cls = AdamW
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer failed to import syncfree AdamW from torch_xla.")
        elif args.optim == OptimizerNames.ADAMW_TORCH_NPU_FUSED:
            try:
                from torch_npu.optim import NpuFusedAdamW

                optimizer_cls = NpuFusedAdamW
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer failed to import FusedAdamW from torch_npu.")
        elif args.optim == OptimizerNames.ADAMW_APEX_FUSED:
            try:
                from apex.optimizers import FusedAdam

                optimizer_cls = FusedAdam
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate apex FusedAdam but apex is not installed!")
        elif args.optim in [
            OptimizerNames.ADAMW_BNB,
            OptimizerNames.ADAMW_8BIT,
            OptimizerNames.PAGED_ADAMW,
            OptimizerNames.PAGED_ADAMW_8BIT,
            OptimizerNames.LION,
            OptimizerNames.LION_8BIT,
            OptimizerNames.PAGED_LION,
            OptimizerNames.PAGED_LION_8BIT,
            OptimizerNames.RMSPROP_BNB,
            OptimizerNames.RMSPROP_8BIT,
            OptimizerNames.RMSPROP_32BIT,
        ]:
            try:
                from bitsandbytes.optim import AdamW, Lion, RMSprop

                is_paged = False
                optim_bits = 32
                optimizer_cls = None
                additional_optim_kwargs = adam_kwargs
                if "paged" in args.optim:
                    is_paged = True
                if "8bit" in args.optim:
                    optim_bits = 8
                if "adam" in args.optim:
                    optimizer_cls = AdamW
                elif "lion" in args.optim:
                    optimizer_cls = Lion
                    additional_optim_kwargs = {"betas": (args.adam_beta1, args.adam_beta2)}
                elif "rmsprop" in args.optim:
                    optimizer_cls = RMSprop
                    # Above we pass all `adam_kwargs` to the optimizer, here
                    # we only pass `optim_args` which can be passed by the user.
                    additional_optim_kwargs = optim_args

                bnb_kwargs = {"optim_bits": optim_bits}
                if "rmsprop" not in args.optim:
                    bnb_kwargs["is_paged"] = is_paged

                optimizer_kwargs.update(additional_optim_kwargs)
                optimizer_kwargs.update(bnb_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate bnb optimizer but bnb is not installed!")
            if is_bitsandbytes_available() and version.parse(
                importlib.metadata.version("bitsandbytes")
            ) < version.parse("0.41.1"):
                logger.warning(
                    "You are using 8-bit optimizers with a version of `bitsandbytes` < 0.41.1. "
                    "It is recommended to update your version as a major bug has been fixed in 8-bit optimizers."
                )
        elif args.optim == OptimizerNames.SGD:
            optimizer_cls = torch.optim.SGD
        elif args.optim == OptimizerNames.ADAGRAD:
            optimizer_cls = torch.optim.Adagrad
        elif args.optim == OptimizerNames.RMSPROP:
            optimizer_cls = torch.optim.RMSprop
        else:
            raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")
        return optimizer_cls, optimizer_kwargs

    def create_optimizer(self):

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        special_params_list: List[str] = ["down_fuse", "residual_weight", "label_gap", "inst_fuse",
                                          "deinstruction_shift"]

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if any(kw in n for kw in special_params_list) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate * 10,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (not any(kw in n for kw in special_params_list)) and (n in decay_parameters) and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (not any(kw in n for kw in special_params_list)) and (n not in decay_parameters) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
            ]

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters,
                                           **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, AttackArguments))
    model_args, data_args, training_args, attack_args = parser.parse_args_into_dataclasses()
    data_args.attack = attack_args.attack 
    if 'Instruct' in model_args.model_name_or_path:
        assert 'SpclSpclSpcl' not in data_args.attack
    print('\n\n' + training_args.output_dir + '\n\n')

    config = LlamaFuseConfig.from_pretrained(
        model_args.model_name_or_path,
    )

    config.residual = True
    config.bit_flip = True

    model = LlamaForCausalLMFuse.from_pretrained(
        model_args.model_name_or_path,
        ignore_mismatched_sizes=True,
        config=config,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=True,
        batched=True
    )

    # add new tokens into the dictionary => update model embedding layer
    special_tokens_dict = {"pad_token": DEFAULT_TOKENS['pad_token'],
                           "eos_token": DEFAULT_TOKENS['eos_token'],
                           "bos_token": DEFAULT_TOKENS['bos_token'],
                           "unk_token": DEFAULT_TOKENS['unk_token'],
                           "additional_special_tokens": SPECIAL_DELM_TOKENS}

    smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict, tokenizer=tokenizer, model=model)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"✅ {name} is trainable")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Params: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")

    # Construct dataloader
    frontend_delimiters = 'SpclSpclSpcl' if 'SpclSpclSpcl' in data_args.attack else find_closest_match(model_args.model_name_or_path, DELIMITERS)
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              downsample=training_args.downsample,
                                              frontend_delimiters=frontend_delimiters)
    if not training_args.downsample and training_args.lr_scale:
        training_args.learning_rate /= data_module["train_dataset"].data_copy_count

    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    config = trainer.model.config  # Access the config of the model
    config.save_pretrained(training_args.output_dir)
    
if __name__ == "__main__":
    train()
