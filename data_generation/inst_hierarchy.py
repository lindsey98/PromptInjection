# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import re
from copy import deepcopy
from torch.utils.data import Dataset
import logging
import torch
from data_generation.struq import jload, preprocess, DataCollatorForSupervisedDataset, DataCollatorForSupervisedDatasetOrig, smart_tokenizer_and_embedding_resize, find_closest_match
from config import PROMPT_FORMAT, IGNORE_ATTACK_SENTENCES, DELIMITERS, SYS_INPUT
from typing import List, Dict
import transformers

def generate_training_data(data_dicts, tokenizer, frontend_delimiters):

    sources = []
    targets = []

    for dialog in data_dicts:
        roles = {"system": [], "user": [], "data": [], "assistant": []}
        for turn in dialog:
            if turn["role"] in roles:
                roles[turn["role"]].append(turn["content"])

        system = "\n".join(roles["system"])
        if not system:
            system = SYS_INPUT
        user     = "\n".join(roles["user"])
        data     = "\n".join(roles["data"])
        response = "\n".join(roles["assistant"]) + tokenizer.eos_token
        if not data:
            formatted_prompt = system + '\n\n' + DELIMITERS[frontend_delimiters][0] + f"\n{user}\n\n" + DELIMITERS[frontend_delimiters][2] + "\n"
        else:
            formatted_prompt = system + '\n\n' + DELIMITERS[frontend_delimiters][0] + f"\n{user}\n\n" + DELIMITERS[frontend_delimiters][1] + f"\n{data}\n\n" + DELIMITERS[frontend_delimiters][2] + "\n"

        targets.append(response)
        sources.append(formatted_prompt)

    return sources, targets


class InstructionHierarchyDataset(Dataset):
    def __init__(self, data_path_list: List[str], downsample_ratio_list: List[float], attack, tokenizer, frontend_delimiters="SpclSpclSpcl", downsample=True):

        super(InstructionHierarchyDataset, self).__init__()
        assert len(data_path_list) == len(downsample_ratio_list)

        self.input_ids = []
        self.labels = []
        self.expert_labels = []
        self.data_source = []
        self.data_copy_count = 1
        #
        for downsample_ratio, data_path in zip(downsample_ratio_list, data_path_list):
            logging.warning(f"Loading data {data_path}")

            list_data_dict = jload(data_path)

            sources, targets = generate_training_data(list_data_dict, tokenizer, frontend_delimiters)

            # downsize data to original size with 50% clean data
            if downsample:
                sample_id = np.random.choice(range(len(sources)), size=int(downsample_ratio * len(sources)), replace=False)
                sources = np.array(sources)[sample_id].tolist()
                targets = np.array(targets)[sample_id].tolist()
            else:
                sources = np.array(sources).tolist()
                targets = np.array(targets).tolist()

            logging.warning("Tokenizing inputs...")
            data_dict = preprocess(sources, targets, frontend_delimiters=frontend_delimiters, tokenizer=tokenizer)

            self.input_ids.extend(data_dict["input_ids"])
            self.labels.extend(data_dict["labels"])
            self.expert_labels.extend(data_dict["expert_labels"])
            self.data_source.extend([data_path] * len(sources))

        # Mix the dataset
        combined = np.arange(len(self.input_ids))
        np.random.shuffle(combined)

        # Apply permutation using numpy for better performance
        self.input_ids = [self.input_ids[i] for i in combined]
        self.labels = [self.labels[i] for i in combined]
        self.expert_labels = [self.expert_labels[i] for i in combined]
        self.data_source = [self.data_source[i] for i in combined]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids = self.input_ids[i],
                    expert_labels = self.expert_labels[i],
                    labels = self.labels[i],
                    data_source = self.data_source[i])


def make_supervised_data_module_hier_orig(tokenizer: transformers.PreTrainedTokenizer, data_args, frontend_delimiters, downsample=True) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = InstructionHierarchyDataset(tokenizer=tokenizer,
                                                data_path_list=data_args.data_path_list,
                                                downsample_ratio_list=data_args.downsample_ratio_list,
                                                attack=data_args.attack,
                                                frontend_delimiters=frontend_delimiters,
                                                downsample=downsample)
    data_collator = DataCollatorForSupervisedDatasetOrig(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_supervised_data_module_hier(tokenizer: transformers.PreTrainedTokenizer, data_args, frontend_delimiters, downsample=True) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = InstructionHierarchyDataset(tokenizer=tokenizer,
                                                data_path_list=data_args.data_path_list,
                                                downsample_ratio_list=data_args.downsample_ratio_list,
                                                attack=data_args.attack,
                                                frontend_delimiters=frontend_delimiters,
                                                downsample=downsample)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)