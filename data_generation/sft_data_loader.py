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
from config import PROMPT_FORMAT, IGNORE_ATTACK_SENTENCES, OTHER_DELM_FOR_TEST, OTHER_DELM_TOKENS, \
    IGNORE_INDEX, DELIMITERS, SPECIAL_DELM_TOKENS, TEXTUAL_DELM_TOKENS
import torch
import transformers
from typing import Dict, Optional, Sequence, Union, Any
import difflib
from dataclasses import dataclass, field
from typing import List


def jload(f, mode="r"):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    jdict = json.load(f)
    f.close()
    return jdict

def load_jsonl(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

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
    mark = np.random.choice(
        OTHER_DELM_TOKENS['mark'][test_idx:] if test else OTHER_DELM_TOKENS['mark'][:test_idx]) + ':'

    def sample_delm(delm_name):
        role_name = 'user' if (delm_name == 'inst' or delm_name == 'inpt') else 'asst'
        if test:
            role = np.random.choice(OTHER_DELM_TOKENS[role_name][test_idx:])
            delm = np.random.choice(OTHER_DELM_TOKENS[delm_name][test_idx:])
        else:
            role = np.random.choice(OTHER_DELM_TOKENS[role_name][:test_idx])
            delm = np.random.choice(OTHER_DELM_TOKENS[delm_name][:test_idx])

        p = np.random.rand()
        if p < 1 / 3:
            return (role + delm).upper()
        elif p < 2 / 3:
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
        return [prompt_dict["prompt_input"].format_map(example) for example in data_dicts], \
            [f"{example['output']}{tokenizer.eos_token}" for example in data_dicts]

    if attack == 'Completion':
        ref_inst_resp = {}
        for ref_sample in jload('./datasets/alpaca_data.json'):
            ref_inst_resp[ref_sample['instruction']] = ref_sample['output']
    sources = []

    for i in range(len(data_dicts)):
        # no input => no attack
        if data_dicts[i].get("input", "") == "":
            sources.append(prompt_dict["prompt_input"].format_map(data_dicts[i]))
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
                data_dicts_item['input'] += ' ' + np.random.choice(
                    IGNORE_ATTACK_SENTENCES['train']) + ' ' + injected_prompt
            elif attack == 'Completion':
                _delm = DELIMITERS[frontend_delimiters]
                data_dicts_item['input'] += '\n\n' + \
                                            _delm[-1] \
                                            + '\n' + \
                                            ref_inst_resp.get(data_dicts_item['instruction'],
                                                              data_dicts_item['output']) + \
                                            '\n\n' + \
                                            _delm[0] + \
                                            '\n' + \
                                            injected_prompt.capitalize()
                if injected_sample['input'] != '':
                    data_dicts_item['input'] += '\n\n' + \
                                                _delm[-2] + \
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
            return i  # Return the ENDING index of the first match
    return -1  # Return -1 if separator is not found


def find_last_occurrence(seq, separator):
    last_index = -1
    for i in range(len(seq) - len(separator) + 1):
        if torch.equal(seq[i:i + len(separator)], separator):  # Check if slice matches the separator
            last_index = i  # Update to the ENDING index of this match
    return last_index  # Return -1 if the separator is not found, or the last match index if found


# def compute_expert_labels_from_input_ids(
#     input_ids: torch.LongTensor,      # (B, L)
#     data_delm_ids: List[int],
#     response_delm_ids: List[int],
#     inst_delm_ids: Optional[List[int]] = None,
#     num_labels: int = 4,
# ) -> torch.LongTensor:
#     """
#     Compute expert_labels on-the-fly from input_ids, replicating
#     compute_expert_labels() from struq.py — but batched and on-device.
#
#     Label scheme (num_labels=3):
#         0 = system instruction  (default)
#         1 = data input          (after first data delimiter)
#         2 = response            (after last response delimiter)
#
#     Label scheme (num_labels=4):
#         0 = preamble/padding    (default)
#         1 = user instruction    (after first inst delimiter)
#         2 = data input          (after first data delimiter)
#         3 = response            (after last response delimiter)
#
#     For decode steps (L==1), all tokens are labelled as response
#     (label = num_labels - 1), because the model is generating and
#     we already handled the prefill correctly.
#     """
#     B, L = input_ids.shape
#     device = input_ids.device
#     response_label = num_labels - 1
#
#     # Decode step: single new token, always response
#     if L == 1:
#         return torch.full((B, 1), fill_value=response_label, dtype=torch.long, device=device)
#
#     data_sep = torch.tensor(data_delm_ids,     dtype=torch.long, device=device)
#     resp_sep = torch.tensor(response_delm_ids, dtype=torch.long, device=device)
#     inst_sep = (torch.tensor(inst_delm_ids, dtype=torch.long, device=device)
#                 if inst_delm_ids is not None else None)
#
#     expert_labels = torch.zeros(B, L, dtype=torch.long, device=device)
#
#     for b in range(B):
#         seq = input_ids[b]  # (L,)
#
#         if num_labels == 3:
#             data_pos = find_first_occurrence(seq, data_sep)
#             if data_pos != -1:
#                 expert_labels[b, data_pos:] = 1
#
#             resp_pos = find_last_occurrence(seq, resp_sep)
#             if resp_pos != -1:
#                 expert_labels[b, resp_pos:] = 2
#
#         elif num_labels == 4:
#             if inst_sep is not None:
#                 inst_pos = find_first_occurrence(seq, inst_sep)
#                 if inst_pos != -1:
#                     expert_labels[b, inst_pos:] = 1
#
#             data_pos = find_first_occurrence(seq, data_sep)
#             if data_pos != -1:
#                 expert_labels[b, data_pos:] = 2
#
#             resp_pos = find_last_occurrence(seq, resp_sep)
#             if resp_pos != -1:
#                 expert_labels[b, resp_pos:] = 3
#
#         else:
#             raise ValueError(f"num_labels must be 3 or 4, got {num_labels}")
#
#     return expert_labels

def compute_expert_labels_from_input_ids(
    input_ids: torch.LongTensor,      # (B, L)
    data_delm_ids: List[int],
    response_delm_ids: List[int],
    inst_delm_ids: Optional[List[int]] = None,
    num_labels: int = 4,
) -> torch.LongTensor:
    """
    Multi-segment version: scans the whole sequence and switches the running
    label at EVERY delimiter occurrence (not just first/last). Each delimiter
    and the tokens after it (until the next delimiter) get that segment's label.

    Single-turn equivalence: when there is exactly one inst / one data / one
    response delimiter in order, this produces the SAME labels as the original
    first/last-occurrence version. Verify with the provided test before use.

    Label scheme (num_labels=4):
        0 = preamble/padding (before any delimiter)
        1 = instruction  (after inst delimiter)
        2 = data         (after data delimiter)
        3 = response     (after response delimiter)
    """
    B, L = input_ids.shape
    device = input_ids.device
    response_label = num_labels - 1

    if L == 1:
        return torch.full((B, 1), response_label, dtype=torch.long, device=device)

    instruct_label = 0 if num_labels == 3 else 1
    data_label     = 1 if num_labels == 3 else 2

    # (delimiter_ids, label_for_segment). Longest first so a delimiter that is a
    # prefix of another doesn't match prematurely.
    delms = []
    if inst_delm_ids is not None and num_labels == 4:
        delms.append((list(inst_delm_ids), instruct_label))
    delms.append((list(data_delm_ids), data_label))
    delms.append((list(response_delm_ids), response_label))
    delms.sort(key=lambda x: -len(x[0]))

    labels = torch.zeros(B, L, dtype=torch.long, device=device)

    for b in range(B):
        seq = input_ids[b].tolist()
        cur = 0
        i = 0
        while i < L:
            matched = False
            for delm_ids, lab in delms:
                n = len(delm_ids)
                if i + n <= L and seq[i:i + n] == delm_ids:
                    cur = lab
                    for j in range(i, i + n):
                        labels[b, j] = cur
                    i += n
                    matched = True
                    break
            if not matched:
                labels[b, i] = cur
                i += 1

    return labels


def _tokenize_fn(strings, tokenizer,
                 add_special_tokens=True,
                 compute_gate=False,
                 frontend_delimiters=None):
    tokenizer.model_max_length = 131072
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=add_special_tokens
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]

    input_ids_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    labels_lens = input_ids_lens

    result = dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

    if compute_gate and frontend_delimiters is not None:
        delm = DELIMITERS[frontend_delimiters]
        num_labels = 4 if len(delm) == 4 else 3
        inst_sep = torch.tensor(tokenizer(delm[0], add_special_tokens=False).input_ids, dtype=torch.long)
        data_sep = torch.tensor(tokenizer(delm[1], add_special_tokens=False).input_ids, dtype=torch.long)
        resp_sep = torch.tensor(tokenizer(delm[-1], add_special_tokens=False).input_ids, dtype=torch.long)
        # For 4-role: delm[1]=user_delm, delm[2]=tool_delm, delm[3]=resp_delm
        # compute_expert_labels uses user_inst_separator=delm[0 or 1], data_sep=delm[1 or 2]
        if num_labels == 4:
            user_sep = torch.tensor(tokenizer(delm[1], add_special_tokens=False).input_ids, dtype=torch.long)
            data_sep = torch.tensor(tokenizer(delm[2], add_special_tokens=False).input_ids, dtype=torch.long)
            resp_sep = torch.tensor(tokenizer(delm[3], add_special_tokens=False).input_ids, dtype=torch.long)
        else:
            user_sep = inst_sep  # unused in 3-label mode
        # result["expert_labels"] = [
        #     compute_expert_labels(ids, user_sep, data_sep, resp_sep, num_labels=num_labels)
        #     for ids in input_ids
        # ]
        result["expert_labels"] = [
            compute_expert_labels_from_input_ids(
                ids.unsqueeze(0),
                data_delm_ids=data_sep.tolist(),
                response_delm_ids=resp_sep.tolist(),
                inst_delm_ids=user_sep.tolist(),
                num_labels=num_labels
            ).squeeze(0)
            for ids in input_ids
        ]

    return result


def preprocess(sources, targets, frontend_delimiters, tokenizer):
    examples = [s + t for s, t in zip(sources, targets)]

    # input + responses
    examples_tokenized = _tokenize_fn(examples,
                                      tokenizer,
                                      frontend_delimiters=frontend_delimiters)
    input_ids = examples_tokenized["input_ids"]

    # input only => find the input length
    sources_lengths = _tokenize_fn(sources,
                                   tokenizer,
                                   frontend_delimiters=frontend_delimiters)["input_ids_lens"]

    labels = deepcopy(input_ids)
    for label, source_len in zip(labels, sources_lengths):
        label[:source_len] = IGNORE_INDEX  # ignore the prompt part, only interested in predicting responses

    return dict(input_ids=input_ids,
                labels=labels)


class SupervisedDataset(Dataset):
    def __init__(self, data_path_list: List[str], tokenizer, attack, frontend_delimiters, downsample=True):
        super(SupervisedDataset, self).__init__()

        prompt_dict_name, attacks = attack.split('_', 1)

        self.input_ids = []
        self.labels = []
        self.data_source = []

        for data_path in data_path_list:
            logging.warning(f"Loading data {data_path}")

            list_data_dict = jload(data_path)

            source_clean, targets_clean = generate_training_data(list_data_dict, prompt_dict_name, 'None',
                                                                 frontend_delimiters, tokenizer)

            if attacks == 'None':
                sources, targets = source_clean, targets_clean
                self.data_copy_count = 1
            else:
                sources = []
                targets = []
                attacks = re.findall('[A-Z][^A-Z]*', attacks)
                self.data_copy_count = len(attacks) + len(attacks) * downsample

                for a in attacks:
                    source, target = generate_training_data(list_data_dict, prompt_dict_name, a, frontend_delimiters,
                                                            tokenizer)
                    sources += source
                    targets += target
                    if downsample:
                        sources += source_clean
                        targets += targets_clean
                print(f"Constructed length of data = {len(sources)}")

            # downsize data to original size with 50% clean data
            if downsample:
                sample_batch_id = np.random.choice(range(self.data_copy_count), len(source_clean))
                sample_id = [(x * len(sample_batch_id) + i) for i, x in enumerate(sample_batch_id)]
                print(f"Down-sampled length of data = {len(sample_id)}")
                sources = np.array(sources)[sample_id].tolist()
                targets = np.array(targets)[sample_id].tolist()
            else:
                sources = np.array(sources).tolist()
                targets = np.array(targets).tolist()

            logging.warning(f"Data {data_path} Loaded with {len(sources)} instances")
            logging.warning("Tokenizing inputs...")
            data_dict = preprocess(sources, targets, frontend_delimiters=frontend_delimiters, tokenizer=tokenizer)

            self.input_ids.extend(data_dict["input_ids"])
            self.labels.extend(data_dict["labels"])
            self.data_source.extend([data_path] * len(sources))

        # Mix the dataset
        combined = np.arange(len(self.input_ids))
        # np.random.shuffle(combined) # fixme: I didnt shuffle

        # Apply permutation using numpy for better performance
        self.input_ids = [self.input_ids[i] for i in combined]
        self.labels = [self.labels[i] for i in combined]
        self.data_source = [self.data_source[i] for i in combined]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i],
                    labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDatasetOrig(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def get_embedding_indices(tokenizer):
    init_values = [tokenizer.encode(v, add_special_tokens=False)[0]
                   for v in TEXTUAL_DELM_TOKENS]  # get the delimiters tokens IDs
    ignore_values = [i for i in range(len(tokenizer))
                     if tokenizer.decode(i) == "#"]  # get the padding token ID
    return init_values, ignore_values


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    REAL_DELIMITERS_INIT_EMBD_IND, _ = get_embedding_indices(
        tokenizer)  # get embeddings for ['instruction', 'input',  'response', '###',  ':']

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens] = input_embeddings_avg  # pad token's embedding is initialized to be average
        output_embeddings[-num_new_tokens] = output_embeddings_avg

        ## let ['[INST]', '[INPT]', '[RESP]', '[MARK]', '[COLN]'] = ['instruction', 'input',  'response', '###', ':']
        for i in range(len(SPECIAL_DELM_TOKENS)):
            input_embeddings[-num_new_tokens + i + 1] = input_embeddings[REAL_DELIMITERS_INIT_EMBD_IND[i]]
            output_embeddings[-num_new_tokens + i + 1] = output_embeddings[REAL_DELIMITERS_INIT_EMBD_IND[i]]


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, frontend_delimiters,
                                downsample=True
                                ) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer,
                                      data_path_list=data_args.data_path_list,
                                      attack=data_args.attack,
                                      frontend_delimiters=frontend_delimiters,
                                      downsample=downsample)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def make_supervised_data_module_orig(tokenizer: transformers.PreTrainedTokenizer, data_args, frontend_delimiters,
                                     downsample=True
                                     ) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer,
                                      data_path_list=data_args.data_path_list,
                                      attack=data_args.attack,
                                      frontend_delimiters=frontend_delimiters,
                                      downsample=downsample)
    data_collator = DataCollatorForSupervisedDatasetOrig(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)