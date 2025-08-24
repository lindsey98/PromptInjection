# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import torch
import transformers
import trl
from data_generation.struq import smart_tokenizer_and_embedding_resize, make_supervised_data_module_orig
from config import DEFAULT_TOKENS, SPECIAL_DELM_TOKENS
from peft import LoraConfig, get_peft_model

@dataclass
class ModelArguments: 
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    window_size: int = field(default=0, metadata={"help": "Window size for the sliding window attention."})
    padding_side: str = field(default="right", metadata={"help": "Padding side for tokenization."})

@dataclass
class DataArguments:
    data_path_list: List[str] = field(
        default_factory=list,
        metadata={"help": "List of file paths to the training data.", "nargs": "+"}
    )

@dataclass
class AttackArguments: 
    attack: str = field(default='TextTextText_None', metadata={"help": "Attack type for SFT/Align"})

@dataclass
class TrainingArguments(trl.ORPOConfig):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    downsample: Optional[bool] = field(default=True)
    lr_scale: Optional[bool] = field(default=True)
    beta: float = field(default=0.1)
    ref_model_init_kwargs: Optional[str] = field(default=None)
    precompute_ref_log_probs: Optional[bool] = field(default=False)
    report_to: Optional[str] = "wandb"
    resume_from_checkpoint: Optional[bool] = field(default=True)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, AttackArguments))
    model_args, data_args, training_args, attack_args = parser.parse_args_into_dataclasses()
    data_args.attack = attack_args.attack
    print('\n\n' + training_args.output_dir + '\n\n')

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    if model_args.window_size > 0:
        model.config.window = model_args.window_size

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=True,
        batched=True
    )

    ## Add special tokens (special delimiters)
    special_tokens_dict = dict()
    special_tokens_dict["pad_token"] = DEFAULT_TOKENS['pad_token'] ###
    special_tokens_dict["eos_token"] = DEFAULT_TOKENS['eos_token']
    special_tokens_dict["bos_token"] = DEFAULT_TOKENS['bos_token']
    special_tokens_dict["unk_token"] = DEFAULT_TOKENS['unk_token']
    special_tokens_dict["additional_special_tokens"] = SPECIAL_DELM_TOKENS ### 
    smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict, tokenizer=tokenizer, model=model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        modules_to_save=["lm_head", "embed_tokens"],
    )
    # Create the PEFT model
    peft_model = get_peft_model(model, lora_config)

    frontend_delimiters = data_args.attack.split("_")[0]
    data_module = make_supervised_data_module_orig(tokenizer=tokenizer,
                                                   data_args=data_args,
                                                   frontend_delimiters=frontend_delimiters,
                                                   downsample=training_args.downsample)
    if not training_args.downsample and training_args.lr_scale:
        training_args.learning_rate /= data_module["train_dataset"].data_copy_count
        
    trainer = transformers.Trainer(
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