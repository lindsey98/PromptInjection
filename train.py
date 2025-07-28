# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import torch
import transformers
import trl
from data_generation.struq import SupervisedDataset, smart_tokenizer_and_embedding_resize, make_supervised_data_module_orig, find_closest_match
from config import IGNORE_INDEX, DEFAULT_TOKENS, SPECIAL_DELM_TOKENS, TEXTUAL_DELM_TOKENS, DELIMITERS
import difflib

@dataclass
class ModelArguments: 
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    base_model_name_or_path: Optional[str] = field(default="")
    extra_config_path: Optional[str] = field(default="")
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
    alignment: str = field(default='none', metadata={"help": "Alignment type."})

@dataclass
class TrainingArguments(trl.ORPOConfig): #transformers.TrainingArguments): #
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
    desirable_weight: Optional[float] = field(default=1)
    undesirable_weight: Optional[float] = field(default=1)
    report_to: Optional[str] = "wandb"
    resume_from_checkpoint: Optional[bool] = field(default=True)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, AttackArguments))
    model_args, data_args, training_args, attack_args = parser.parse_args_into_dataclasses()
    data_args.attack = attack_args.attack 
    if 'Instruct' in model_args.model_name_or_path:
        assert 'SpclSpclSpcl' not in data_args.attack
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

    special_tokens_dict = dict()
    special_tokens_dict["pad_token"] = DEFAULT_TOKENS['pad_token'] ###
    special_tokens_dict["eos_token"] = DEFAULT_TOKENS['eos_token']
    special_tokens_dict["bos_token"] = DEFAULT_TOKENS['bos_token']
    special_tokens_dict["unk_token"] = DEFAULT_TOKENS['unk_token']
    special_tokens_dict["additional_special_tokens"] = SPECIAL_DELM_TOKENS ### 
    smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict, tokenizer=tokenizer, model=model)

    frontend_delimiters = 'SpclSpclSpcl' if 'SpclSpclSpcl' in data_args.attack else find_closest_match(model_args.model_name_or_path, DELIMITERS)
    data_module = make_supervised_data_module_orig(tokenizer=tokenizer,
                                                   data_args=data_args,
                                                   frontend_delimiters=frontend_delimiters,
                                                   downsample=training_args.downsample)
    if not training_args.downsample and training_args.lr_scale:
        training_args.learning_rate /= data_module["train_dataset"].data_copy_count
        
    trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

    
if __name__ == "__main__":
    train()
    # model = "huggyllama/llama-7b"
    # data_path = "datasets/davinci_003_outputs.json"
    # attack = "SpclSpclSpcl_NaiveCompletion"
    # model_max_length = 512
    # lr = 2e-5

    # --model_name_or_path meta-llama/Llama-3.2-1B --data_path datasets/alpaca_data_cleaned.json --bf16 True  --output_dir debug  --num_train_epochs 3  --per_device_train_batch_size 2 --per_device_eval_batch_size 2
    #             --gradient_accumulation_steps 8  --evaluation_strategy "no"  --save_strategy "no"  --save_total_limit 1  --learning_rate 2e-5  --weight_decay 0.  --warmup_ratio 0.03 \
    #             --lr_scheduler_type "cosine" --logging_steps 1 \
    #             --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \
    #             --tf32 True --attack SpclSpclSpcl_NaiveCompletion --model_max_length 512