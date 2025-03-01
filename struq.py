# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import re
from copy import deepcopy
from torch.utils.data import Dataset
import logging
import io, json
from config import PROMPT_FORMAT, IGNORE_ATTACK_SENTENCES, OTHER_DELM_FOR_TEST, OTHER_DELM_TOKENS, IGNORE_INDEX, \
    DELIMITERS
import torch

def jload(f, mode="r"):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    jdict = json.load(f)
    f.close()
    return jdict

def jdump(obj, f, mode="w", indent=4, default=str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()

def format_with_other_delimiters(text, test=False):
    test_idx = - OTHER_DELM_FOR_TEST
    mark = np.random.choice(OTHER_DELM_TOKENS['mark'][test_idx:] if test else OTHER_DELM_TOKENS['mark'][:test_idx]) + ':'
    
    def sample_delm(delm_name):
        role_name = 'user' if (delm_name == 'inst' or delm_name == 'inpt') else 'asst'
        if test: 
            role = np.random.choice(OTHER_DELM_TOKENS[role_name][test_idx:]) 
            delm = np.random.choice(OTHER_DELM_TOKENS[delm_name][test_idx:])
        else:    
            role = np.random.choice(OTHER_DELM_TOKENS[role_name][:test_idx]) 
            delm = np.random.choice(OTHER_DELM_TOKENS[delm_name][:test_idx])
        
        p = np.random.rand()
        if p < 1/3:
            return (role + delm).upper()
        elif p < 2/3:
            return (role + delm).lower()
        else:
            return role + delm
    
    for delm in DELIMITERS.values():
        if '' in delm or ' ' in delm: continue
        text = text.replace(delm[0], mark.format(s=sample_delm('inst')))
        text = text.replace(delm[1], mark.format(s=sample_delm('inpt')))
        text = text.replace(delm[2], mark.format(s=sample_delm('resp')))
    return text

def generate_training_data(data_dicts, prompt_dict_name, attack, frontend_delimiters, tokenizer):

    prompt_dict = PROMPT_FORMAT[prompt_dict_name]

    if attack == 'None':
        return [
            prompt_dict["prompt_input"].format_map(example) if example.get("input", "") != "" else prompt_dict["prompt_no_input"].format_map(example) for example in data_dicts
        ], [f"{example['output']}{tokenizer.eos_token}" for example in data_dicts]

    if attack == 'Completion':
        ref_inst_resp = {}
        for ref_sample in jload('./datasets/alpaca_data.json'):
            ref_inst_resp[ref_sample['instruction']] = ref_sample['output']
    sources = []

    for i in range(len(data_dicts)):
        # no input => no attack
        if data_dicts[i].get("input", "") == "":
            sources.append(prompt_dict["prompt_no_input"].format_map(data_dicts[i]))
        else:
            # randomly choose another instruction from the same dataset
            injected_sample = deepcopy(np.random.choice(data_dicts)) 
            injected_sample['instruction'] = injected_sample['instruction']
            if injected_sample['instruction'][-1] == '?': 
                injected_prompt = 'answer the following question. ' + \
                                  injected_sample['instruction'] + ' ' + \
                                  injected_sample['input']
            else: 
                injected_prompt = injected_sample['instruction'][0].lower() + \
                                  injected_sample['instruction'][1:] + ' ' + \
                                  injected_sample['input']
            
            data_dicts_item = deepcopy(data_dicts[i])
            if data_dicts_item['input'][-1] != '.': data_dicts_item['input'] += '.'
            if attack == 'Naive':
                data_dicts_item['input'] += ' ' + injected_prompt[0].upper() + injected_prompt[1:]
            elif attack == 'Ignore':
                data_dicts_item['input'] += ' ' + np.random.choice(IGNORE_ATTACK_SENTENCES['train']) + ' ' + injected_prompt
            elif attack == 'Completion':
                data_dicts_item['input'] += '\n\n' + \
                                            DELIMITERS[frontend_delimiters][2] \
                                            + '\n' + \
                                            ref_inst_resp.get(data_dicts_item['instruction'], data_dicts_item['output']) + \
                                            '\n\n' + \
                                             DELIMITERS[frontend_delimiters][0] + \
                                            '\n' + \
                                            injected_prompt.capitalize()
                if injected_sample['input'] != '':
                    data_dicts_item['input'] += '\n\n' + \
                                                DELIMITERS[frontend_delimiters][1] + \
                                                '\n' + \
                                                injected_sample['input']
                data_dicts_item['input'] = format_with_other_delimiters(data_dicts_item['input'], test=False)

            else:
                raise NotImplementedError

            sources.append(prompt_dict["prompt_input"].format_map(data_dicts_item))

    return sources, [f"{example['output']}{tokenizer.eos_token}" for example in data_dicts]


def find_first_occurrence(seq, separator):
    for i in range(len(seq) - len(separator) + 1):
        if torch.equal(seq[i:i + len(separator)], separator):  # Check if slice matches the separator
            return i  # Return the starting index of the first match
    return -1  # Return -1 if separator is not found

def _tokenize_fn(strings, tokenizer, frontend_delimiters='SpclSpclSpcl', compute_gate=False):
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]

    expert_labels = None
    if compute_gate:
        data_seperator = tokenizer(DELIMITERS[frontend_delimiters][1],
                                   return_tensors="pt",
                                   padding=False,
                                   truncation=False,
                                   add_special_tokens=False).input_ids[0] # starting delimiter of data input
        response_seperator = tokenizer(DELIMITERS[frontend_delimiters][2],
                                       return_tensors="pt",
                                       padding=False,
                                       truncation=False,
                                       add_special_tokens=False).input_ids[0] # starting delimiter of response

        # Find the starting delimiters of Input
        expert_labels = []
        for seq in input_ids:
            separator_position = find_first_occurrence(seq, data_seperator)
            if separator_position == -1:
                separator_position = find_first_occurrence(seq, response_seperator)

            expert_this = torch.zeros(len(seq), dtype=torch.long)  # Default: instruction (0)
            if separator_position != -1:
                expert_this[separator_position:] = 1  # After the separator: data (1)
            expert_labels.append(expert_this)

    input_ids_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    expert_labels_lens = labels_lens = input_ids_lens

    return dict(
        input_ids=input_ids,
        labels=labels,
        expert_labels=expert_labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
        expert_labels_lens=expert_labels_lens,
    )

def preprocess(sources, targets, frontend_delimiters, tokenizer):

    examples = [s + t for s, t in zip(sources, targets)]

    # input + responses
    examples_tokenized = _tokenize_fn(examples, tokenizer, frontend_delimiters=frontend_delimiters, compute_gate=True)
    input_ids = examples_tokenized["input_ids"]
    expert_labels = examples_tokenized["expert_labels"]

    # input only => find the input length
    sources_lengths = _tokenize_fn(sources, tokenizer, frontend_delimiters=frontend_delimiters, compute_gate=False)["input_ids_lens"]

    labels = deepcopy(input_ids)
    for label, source_len in zip(labels, sources_lengths):
        label[:source_len] = IGNORE_INDEX # ignore the prompt part, only interested in predicting responses

    return dict(input_ids=input_ids,
                expert_labels=expert_labels,
                labels=labels)


class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, attack, frontend_delimiters="SpclSpclSpcl", downsample=True):
        super(SupervisedDataset, self).__init__() 
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)
        prompt_dict_name, attacks = attack.split('_')
        source_clean, targets_clean = generate_training_data(list_data_dict, prompt_dict_name, 'None', frontend_delimiters, tokenizer)
        
        if attacks == 'None': 
            sources, targets = source_clean, targets_clean
            self.data_copy_count = 1
        else:
            attacks = re.findall('[A-Z][^A-Z]*', attacks)
            sources = []
            targets = []
            self.data_copy_count = len(attacks) + len(attacks) * downsample
            
            for a in attacks:
                source, target = generate_training_data(list_data_dict, prompt_dict_name, a, frontend_delimiters, tokenizer)
                sources += source
                targets += target
                if downsample:
                    sources += source_clean
                    targets += targets_clean
                    
            # downsize data to original size with 50% clean data
            if downsample:
                sample_batch_id = np.random.choice(range(self.data_copy_count), len(source_clean))
                sample_id = [(x * len(sample_batch_id) + i) for i, x in enumerate(sample_batch_id)]
                sources = np.array(sources)[sample_id].tolist()
                targets = np.array(targets)[sample_id].tolist()
            else:
                sources = np.array(sources).tolist()
                targets = np.array(targets).tolist()

        logging.warning("Tokenizing inputs...")
        data_dict = preprocess(sources, targets, frontend_delimiters=frontend_delimiters, tokenizer=tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.expert_labels = data_dict["expert_labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i],
                    expert_labels=self.expert_labels[i],
                    labels=self.labels[i])