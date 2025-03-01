# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import transformers
import torch
from train import smart_tokenizer_and_embedding_resize, make_supervised_data_module, ModelArguments, DataArguments, TrainingArguments, AttackArguments, find_closest_match
from config import DEFAULT_TOKENS, SPECIAL_DELM_TOKENS, DELIMITERS
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer
from transformers.utils import logging
from torch import nn
from modeling.llama_instsep import LlamaForCausalLMMoE, LlamaMoEConfig
import yaml
logger = logging.get_logger(__name__)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, AttackArguments))
    model_args, data_args, training_args, attack_args = parser.parse_args_into_dataclasses()
    data_args.attack = attack_args.attack 
    if 'Instruct' in model_args.model_name_or_path:
        assert 'SpclSpclSpcl' not in data_args.attack
    print('\n\n' + training_args.output_dir + '\n\n')

    config = LlamaMoEConfig.from_pretrained(
        model_args.model_name_or_path,
    )
    if len(model_args.extra_config_yaml_path) > 0:
        with open(model_args.extra_config_yaml_path, "r") as file:
            loaded_custom_config = yaml.safe_load(file)
        merged_config_dict = {**config.to_dict(), **loaded_custom_config}
        config = LlamaMoEConfig.from_dict(merged_config_dict)

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
        use_fast=False,
    )

    # add new tokens into the dictionary => update model embedding layer
    special_tokens_dict = {"pad_token": DEFAULT_TOKENS['pad_token'],
                           "eos_token": DEFAULT_TOKENS['eos_token'],
                           "bos_token": DEFAULT_TOKENS['bos_token'],
                           "unk_token": DEFAULT_TOKENS['unk_token'],
                           "additional_special_tokens": SPECIAL_DELM_TOKENS}

    smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict, tokenizer=tokenizer, model=model)

    # Create PEFT config for these modules and wrap the model to PEFT
    lora_config = LoraConfig(
        r=16,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=["q_proj", "v_proj", "o_proj"],
        modules_to_save=["lm_head", "input_shifts", "intermediate_shifts"], # fixme
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )
    # # Get LoRA model
    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if "input_shifts" in name or "lm_head" in name or "intermediate_shifts" in name:
            param.requires_grad = True  # ensure that they are not frozen

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"✅ {name} is trainable")

    # Construct dataloader
    frontend_delimiters = 'SpclSpclSpcl' if 'SpclSpclSpcl' in data_args.attack else find_closest_match(model_args.model_name_or_path, DELIMITERS)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, downsample=training_args.downsample, frontend_delimiters=frontend_delimiters)
    data_module["train_dataset"].dataloader_num_workers = 4  # Adjust based on your CPU cores
    if not training_args.downsample and training_args.lr_scale:
        training_args.learning_rate /= data_module["train_dataset"].data_copy_count

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    # Save the custom configuration after training
    config = trainer.model.config  # Access the config of the model
    config.save_pretrained(training_args.output_dir)

    
if __name__ == "__main__":
    train()
    # model = "meta-llama/Llama-3.2-1B"
    # data_path = "datasets/alpaca_data_cleaned.json"
    # attack = "SpclSpclSpcl_NaiveCompletion"
    # model_max_length = 512
    # lr = 2e-5

    #  CUDA_VISIBLE_DEVICES=1,2,3,0 python -m torch.distributed.run --nproc_per_node=4 train_seg.py --model_name_or_path meta-llama/Llama-3.2-1B --data_path datasets/alpaca_data_cleaned.json  --output_dir meta-llama/Llama-3.2-1B-SpclSpclSpcl_NaiveCompletion-moe  --num_train_epochs 3  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8  --evaluation_strategy "no"  --save_strategy "epoch"  --learning_rate 2e-4  --weight_decay 0.  --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" --tf32 True --attack SpclSpclSpcl_NaiveCompletion --model_max_length 512 --bf16 True --dataloader_num_workers 4