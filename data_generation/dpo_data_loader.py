from config import PROMPT_FORMAT, DELIMITERS, DEFAULT_SYSTEM_PROMPT
import torch
import transformers
from transformers import PreTrainedTokenizer
from typing import Dict, Optional, Sequence, Union, Any, List, Tuple
from dataclasses import dataclass, field
from datasets import Dataset as HFDataset
from .sft_data_loader import jload, _tokenize_fn

# mm_token_type_ids convention (Qwen3.5 get_rope_index): text=0, image=1, video=2
MM_TEXT, MM_IMAGE, MM_VIDEO = 0, 1, 2


class _SafeFormatDict(dict):
    def __missing__(self, key):
        if key == "system":
            return DEFAULT_SYSTEM_PROMPT
        raise KeyError(key)

def generate_training_data_dpo(data_dicts, prompt_dict_name, tokenizer):
    prompt_dict = PROMPT_FORMAT[prompt_dict_name]
    prompt_inputs = [prompt_dict["prompt_input"].format_map(_SafeFormatDict(ex)) for ex in data_dicts]
    chosen_responses = [f"{ex['chosen']}{tokenizer.eos_token}" for ex in data_dicts]
    rejected_responses = [f"{ex['rejected']}{tokenizer.eos_token}" for ex in data_dicts]
    return prompt_inputs, chosen_responses, rejected_responses


class PreferenceWithExpertDatasetHF:
    def __init__(self, data_path_list: List[str], tokenizer: PreTrainedTokenizer, attack: str):
        prompt_dict_name, attacks = attack.split('_', 1)
        prompts: List[str] = []
        chosens: List[str] = []
        rejecteds: List[str] = []

        for data_path in data_path_list:
            list_data_dict = jload(data_path)
            p, c, r = generate_training_data_dpo(list_data_dict, prompt_dict_name, tokenizer)
            prompts.extend(p)
            chosens.extend(c)
            rejecteds.extend(r)

        self.ds = HFDataset.from_dict({
            "prompt": prompts,
            "chosen": chosens,
            "rejected": rejecteds,
        })

    def hf_dataset(self):
        return self.ds

@dataclass
class DPOCollatorWithExpert(object):

    tokenizer: transformers.PreTrainedTokenizer
    frontend_delimiters: str
    max_length: int = 4096
    max_prompt_length: int = 3072
    max_target_length: int = 1024
    expert_pad_val: int = field(init=False)   # set in __post_init__ from num_labels

    def __post_init__(self):
        delm = DELIMITERS[self.frontend_delimiters]
        num_labels = 4 if len(delm) == 4 else 3
        self.expert_pad_val = num_labels - 1  # response label = highest label

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        prompt, chosen, rejected = tuple([instance[key] for instance in instances] for key in ("prompt", "chosen", "rejected"))

        tokenized_prompt = _tokenize_fn(prompt, self.tokenizer,
                                        frontend_delimiters=self.frontend_delimiters,
                                        compute_gate=True)
        prom: List[torch.Tensor]      = tokenized_prompt["input_ids"]
        prom_expt: List[torch.Tensor] = tokenized_prompt["expert_labels"]
        prom      = [p[:self.max_prompt_length] for p in prom]
        prom_expt = [e[:self.max_prompt_length] for e in prom_expt]
        prom_mask = [torch.ones_like(p) for p in prom]
        p_lens = torch.tensor([p.size(0) for p in prom], dtype=torch.long) # length of the input prompt part

        # 2) resp: chosen / rejected
        def enc(texts):
            # add_special_tokens=False: the prompt (tokenized separately, with BOS)
            # is prepended to this response, so the response must NOT get its own
            # BOS — otherwise a spurious BOS lands mid-sequence at the prompt/
            # response boundary (train/inference mismatch). The explicit eos in the
            # text is still tokenized to the eos id.
            out = self.tokenizer(texts, padding=False, truncation=True, max_length=self.max_target_length,
                                 add_special_tokens=False, return_tensors=None)
            ids = [torch.tensor(x, dtype=torch.long) for x in out["input_ids"]]
            return ids

        chosen_response: List[torch.LongTensor]   = enc(chosen)
        rejected_response: List[torch.LongTensor] = enc(rejected)

        # 3) concat + pad expert for resp
        def concat(p_ids, p_mask, p_expt, r_ids):
            max_resp = self.max_length - p_ids.size(0)
            r_ids = r_ids[:max_resp] if r_ids.size(0) > max_resp else r_ids # truncate
            r_mask = torch.ones_like(r_ids)
            r_expt = torch.full_like(r_ids, self.expert_pad_val) # responses pad with 2
            ids  = torch.cat([p_ids,  r_ids],  dim=0) # concat prompt with response
            mask = torch.cat([p_mask, r_mask], dim=0)
            expt = torch.cat([p_expt, r_expt], dim=0)
            ids, mask, expt = ids[:self.max_length], mask[:self.max_length], expt[:self.max_length]
            return ids, mask, expt

        chosen_trip = [concat(prom[i], prom_mask[i], prom_expt[i], chosen_response[i]) for i in range(len(prom))]
        reject_trip = [concat(prom[i], prom_mask[i], prom_expt[i], rejected_response[i]) for i in range(len(prom))]

        def pad(trips):
            ids  = torch.nn.utils.rnn.pad_sequence([t[0] for t in trips], batch_first=True,
                                                   padding_value=self.tokenizer.pad_token_id or 0)
            mask = torch.nn.utils.rnn.pad_sequence([t[1] for t in trips], batch_first=True,
                                                   padding_value=0)
            expt = torch.nn.utils.rnn.pad_sequence([t[2] for t in trips], batch_first=True,
                                                   padding_value=self.expert_pad_val)
            return ids, mask, expt

        c_ids, c_am, c_ex = pad(chosen_trip)
        r_ids, r_am, r_ex = pad(reject_trip)

        out = dict(
            chosen_input_ids=c_ids,
            chosen_attention_mask=c_am,
            chosen_expert_labels=c_ex,
            rejected_input_ids=r_ids,
            rejected_attention_mask=r_am,
            rejected_expert_labels=r_ex,
            prompt_lens=p_lens,
        )

        return out


def make_dpo_data_module(tokenizer, data_args, frontend_delimiters,
                         max_length=2048):
    train_dataset = PreferenceWithExpertDatasetHF(
        tokenizer=tokenizer, data_path_list=data_args.data_path_list,
        attack=data_args.attack,
    )
    data_collator = DPOCollatorWithExpert(
        tokenizer=tokenizer,
        frontend_delimiters=frontend_delimiters,
        max_length=max_length,
        max_prompt_length=max_length * 3 // 4,   # 1536 if max_length=2048
        max_target_length=max_length // 4,       # 512
    )
    return dict(train_dataset=train_dataset.hf_dataset(),
                eval_dataset=None,
                data_collator=data_collator)