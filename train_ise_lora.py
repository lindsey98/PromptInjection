import transformers
from train import smart_tokenizer_and_embedding_resize, ModelArguments, DataArguments, TrainingArguments, AttackArguments
from transformers import Trainer
from transformers.utils import logging
from modeling.llama_instsep import LlamaForCausalLMMoE, LlamaMoEConfig, LlamaForCausalLMMoEV2
from data_generation.struq import jload, make_supervised_data_module
from peft import LoraConfig, get_peft_model
import os
from transformers.trainer import PreTrainedModel, is_sagemaker_mp_enabled, Adafactor, is_bitsandbytes_available
from typing import Optional, Tuple, Any, List
from torch import nn
logger = logging.get_logger(__name__)
os.environ["WANDB_WATCH"]="all" # log all parameters gradients

class ISETrainer(Trainer):

    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        special_params_list: List[str] = ["input_shifts"]

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
    print('\n\n' + training_args.output_dir + '\n\n')

    config = LlamaMoEConfig.from_pretrained(
        model_args.model_name_or_path,
    )

    model = LlamaForCausalLMMoE.from_pretrained(
        model_args.model_name_or_path,
        ignore_mismatched_sizes=True,
        config=config,
    )

    if model_args.window_size > 0:
        model.config.window = model_args.window_size

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=True,
        batched=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
        r=16,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        modules_to_save=["lm_head", "embed_tokens", "input_shifts"],
    )
    # Create the PEFT model
    peft_model = get_peft_model(model, lora_config)

    # Construct dataloader
    frontend_delimiters = data_args.attack.split("_")[0]
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              frontend_delimiters=frontend_delimiters,
                                              downsample=training_args.downsample)
    if not training_args.downsample and training_args.lr_scale:
        training_args.learning_rate /= data_module["train_dataset"].data_copy_count

    trainer = ISETrainer(
        model=peft_model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )
    trainer.model.print_trainable_parameters()
    trainer.train()

    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(training_args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(training_args.output_dir)

    config = trainer.model.config  # Access the config of the model
    config.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()
