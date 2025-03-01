# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from trl import DPOTrainer, KTOTrainer, ORPOTrainer, KTOConfig, ORPOConfig
import os, re, time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import numpy as np
from copy import deepcopy
import transformers
from config import IGNORE_ATTACK_SENTENCES, PROMPT_FORMAT, DEFAULT_TOKENS, DELIMITERS, SPECIAL_DELM_TOKENS
from struq import jload, jdump, format_with_other_delimiters
from train import ModelArguments, DataArguments, AttackArguments, smart_tokenizer_and_embedding_resize, TrainingArguments, find_closest_match
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import io
import json


def generate_preference_data(clean_data_path, frontend_delimiters, attack, alignment, tokenizer):

    preference_data_path = clean_data_path.split('/')[0] + '/preference_' + frontend_delimiters + '_' + alignment + '_' + attack + '_' + clean_data_path.split('/')[1]
    naive_proportion = 0.9

    if not os.path.exists(preference_data_path):
        clean_data = jload(clean_data_path)
        prompt_dict = PROMPT_FORMAT[frontend_delimiters]
        preference_data = []

        if 'Completion' in attack:
            ref_inst_resp = {}
            for ref_sample in jload('datasets/alpaca_data.json'):
                ref_inst_resp[ref_sample['instruction'].replace(tokenizer.pad_token, '')] = ref_sample['output']

        for i in range(len(clean_data)):
            if clean_data[i].get("input", "") == "":
                continue
            current_sample = deepcopy(clean_data[i])
            if current_sample['input'][-1] != '.':
                current_sample['input'] += '.'

            injected_sample = np.random.choice(clean_data)
            injected_prompt = ('answer the following question. ' + injected_sample['instruction'] + ' ' + injected_sample['input']) if injected_sample['instruction'][-1] == '?' else (injected_sample['instruction'][0].lower() + injected_sample['instruction'][1:] + ' ' + injected_sample['input'])

            if attack == 'Naive':
                current_sample['input'] += ' ' + injected_prompt[0].upper() + injected_prompt[1:]
            elif attack == 'NaiveCompletion':
                if np.random.rand() < naive_proportion: 
                    current_sample['input'] += ' ' + injected_prompt[0].upper() + injected_prompt[1:]
                else: 
                    current_sample['input'] += '\n\n' + DELIMITERS[frontend_delimiters][2] + '\n' + \
                                               ref_inst_resp.get(current_sample['instruction'], current_sample['output']) + '\n\n' + \
                                                        DELIMITERS[frontend_delimiters][0] + '\n' + injected_prompt.capitalize()
                    if injected_sample['input'] != '':
                        current_sample['input'] += '\n\n' + \
                                                   DELIMITERS[frontend_delimiters][1] + '\n' + \
                                                   injected_sample['input']
                    current_sample['input'] = format_with_other_delimiters(current_sample['input'], test=False)
            else:
                raise NotImplementedError
            
            preference_data.append({
                'prompt': prompt_dict["prompt_input"].format_map(current_sample),
                'chosen': current_sample['output'] + tokenizer.eos_token,
                'rejected': injected_sample['output'] + tokenizer.eos_token,
            })

        jdump(preference_data, preference_data_path)

    time.sleep(10)
    return load_dataset('json', data_files=preference_data_path, split="train")

def align():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, AttackArguments))
    model_args, training_args, data_args, attack_args = parser.parse_args_into_dataclasses()
    if 'Instruct' in model_args.model_name_or_path:
        assert 'SpclSpclSpcl' not in data_args.attack

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.base_model_name_or_path if len(model_args.base_model_name_or_path) else model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        ignore_mismatched_sizes=True
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.base_model_name_or_path if len(model_args.base_model_name_or_path) else model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
    )

    special_tokens_dict = dict()
    special_tokens_dict["pad_token"] = DEFAULT_TOKENS['pad_token']
    special_tokens_dict["eos_token"] = DEFAULT_TOKENS['eos_token']
    special_tokens_dict["bos_token"] = DEFAULT_TOKENS['bos_token']
    special_tokens_dict["unk_token"] = DEFAULT_TOKENS['unk_token']
    special_tokens_dict["additional_special_tokens"] = SPECIAL_DELM_TOKENS

    smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict, tokenizer=tokenizer, model=model)
    tokenizer.model_max_length = 512  ### the default value is too large for model.generation_config.max_new_tokens
    if len(model_args.base_model_name_or_path):
        model = PeftModel.from_pretrained(model, model_args.model_name_or_path, is_trainable=True)
        model = model.merge_and_unload()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules = ["q_proj", "v_proj"]
        )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print(training_args.output_dir, '\n\n\n')
    if model_args.window_size > 0:
        model.config.window = model_args.window_size

    frontend_delimiters = 'SpclSpclSpcl' if 'SpclSpclSpcl' in data_args.attack else find_closest_match(model_args.model_name_or_path, DELIMITERS)
    train_dataset = generate_preference_data(
        data_args.data_path, 
        frontend_delimiters,
        attack_args.attack,
        attack_args.alignment,
        tokenizer
    )

    trainer = {
        'dpo': DPOTrainer,
        'kto': KTOTrainer,
        'orpo': ORPOTrainer,
        }[attack_args.alignment](
            model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    align()


    # python -m torch.distributed.run  --nproc_per_node=4 secalign.py --model_name_or_path meta-llama/Llama-3.2-1B-SpclSpclSpcl_NaiveCompletion-struq --data_path datasets/alpaca_data_cleaned.json --output_dir meta-llama/Llama-3.2-1B-SpclSpclSpcl_NaiveCompletion-secalign  --num_train_epochs 3 --per_device_train_batch_size 4 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --evaluation_strategy "no" --save_strategy "epoch" --learning_rate 2e-4 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" --tf32 True --attack NaiveCompletion --model_max_length 512 --bf16 True --dataloader_num_workers 4 --base_model_name_or_path meta-llama/Llama-3.2-1B
