import fastchat
from fastchat.conversation import register_conv_template, Conversation, SeparatorStyle
from ml_collections import config_dict
from config import (
    DELIMITERS,
    PROMPT_FORMAT,
    TEST_INJECTED_PROMPT,
    TEST_INJECTED_WORD,
)
import dataclasses
import torch
from data_generation.struq import jload, _tokenize_fn
from testing.test import load_full_model, test_model_output, recursive_filter
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
from tqdm import tqdm
import csv
import sys
import logging
logging.basicConfig(level=logging.INFO)
from gcg.gcg import GCGAttack
from gcg.log import setup_logger
from gcg.utils import Message, Role, SuffixManager, get_nonascii_toks
from gcg.eval_input import LengthMismatchError
import numpy as np

def setup_gcg_configs(args):
    cfg = config_dict.ConfigDict()
    cfg.name = "gcg"  # Attack name
    cfg.seed = 0  # Random seed
    cfg.log_freq = 100
    cfg.adv_suffix_init = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    # Init suffix length (auto-generated from adv_suffix_init)
    cfg.init_suffix_len = -1
    cfg.num_steps = 500
    cfg.fixed_params = True  # Used fixed scenario params in each iteration
    cfg.allow_non_ascii = False
    cfg.batch_size = 512  # Number of candidates to evaluate in each step
    # NOTE: Reduce mini_batch_size if OOM
    cfg.mini_batch_size = 32 #32 #128 #256  # -1 for full batch (config.batch_size)
    cfg.seq_len = 50  # Max sequence length for computing loss
    cfg.loss_temperature = 1.0  # Temperature for computing loss
    cfg.max_queries = -1  # Max number of queries (default: -1 for no limit)
    cfg.skip_mode = "none"  # "none", "visited", "seen"
    cfg.add_space = False  # Add metaspace in front of target
    cfg.topk = 256
    cfg.num_coords = (1, 1)  # Number of coordinates to change in one step
    cfg.mu = 0.0  # Momentum parameter
    cfg.custom_name = ""
    cfg.log_dir = args.model_name_or_path if os.path.exists(args.model_name_or_path) else (args.model_name_or_path+'-log')
    cfg.sample_id = -1 # to be initialized in every run of the sample
    cfg.pass_expert_labels = args.pass_expert_labels
    cfg.delm_ids = (args.inst_seperator, args.data_seperator, args.response_seperator)
    cfg.num_labels = 3
    return cfg

@dataclasses.dataclass
class CustomConversation(fastchat.conversation.Conversation):
    def get_prompt(self) -> str:
        system_prompt = self.system_template.format(system_message=self.system_message)
        seps = [self.sep, self.sep2]
        ret = system_prompt + self.sep
        for i, (role, message) in enumerate(self.messages):
            if message:
                ret += role + "\n" + message + seps[i % 2]
            else:
                ret += role + "\n"
        return ret

    def copy(self):
        return CustomConversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

def eval_func(adv_suffix, messages):
    inst, data = messages[1].content.split(f'\n\n{data_delm}\n')
    return test_model_output([
        prompt_template.format_map({
            "instruction": inst,
            "input": data + ' ' + adv_suffix
        })
    ],
        model,
        tokenizer,
        frontend_delimiters,
        pass_expert_labels=args.pass_expert_labels
    )

def gcg(d_item, attack, cfg, data_delm):
    cfg.sample_id = d_item["id"]
    attack._setup_log_file(cfg)
    prompt_no_sys = (
        f"{d_item['instruction']}\n\n{data_delm}\n{d_item['input']}"
        f" {TEST_INJECTED_PROMPT.capitalize()}"
    )
    try:
        adv_suffix = attack.run(
            messages=[
                Message(Role.SYSTEM, ""),
                Message(Role.USER, prompt_no_sys)
            ],
            target=TEST_INJECTED_WORD.lower()
        ).best_suffix
    except LengthMismatchError:
        adv_suffix = cfg.adv_suffix_init
    d_item["input"] += ' ' + TEST_INJECTED_PROMPT.capitalize() + ' ' + adv_suffix
    torch.cuda.empty_cache()
    return d_item

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Testing a model with a specific attack')
    parser.add_argument('-m', '--model_name_or_path', type=str, nargs="+")
    parser.add_argument('-a', '--attack', type=str, default='gcg',
                        choices=["gcg"],
                        nargs='+')
    parser.add_argument('-d', '--defense', type=str, default='none',
                        choices=['none', 'sandwich', 'instructional', 'reminder', 'isolation', 'incontext'],
                        help='Baseline test-time zero-shot prompting defense')
    parser.add_argument('--data_path', type=str, default='datasets/davinci_003_outputs.json')
    parser.add_argument('--pass_expert_labels', default=False,
                        help="Whether to past expert labels instruction/data as an input", action='store_true')
    parser.add_argument('--customized_model_class', type=str, help="Customized model class", default='')
    args = parser.parse_args()
    args.model_name_or_path = args.model_name_or_path[0]

    model, tokenizer, frontend_delimiters, training_attacks = load_full_model(args.model_name_or_path, customized_model_class=args.customized_model_class)

    prompt_format = PROMPT_FORMAT[frontend_delimiters]
    model_path = args.model_name_or_path
    log_path = f"{model_path}-log" if not os.path.exists(model_path) else model_path
    attack_log_file = os.path.join(log_path, f"gcg-{args.defense}-{TEST_INJECTED_WORD}.csv")
    summary_path = os.path.join(log_path, "summary.tsv")

    prompt_template = PROMPT_FORMAT[frontend_delimiters]["prompt_input"]
    inst_delm, data_delm, resp_delm = DELIMITERS[frontend_delimiters]
    args.inst_seperator = tokenizer(inst_delm,
                                   return_tensors="pt",
                                   padding=False,
                                   truncation=False,
                                   add_special_tokens=False).input_ids[0]  # starting delimiter of response
    args.data_seperator = tokenizer(data_delm,
                               return_tensors="pt",
                               padding=False,
                               truncation=False,
                               add_special_tokens=False).input_ids[0]  # starting delimiter of data input
    args.response_seperator = tokenizer(resp_delm,
                                   return_tensors="pt",
                                   padding=False,
                                   truncation=False,
                                   add_special_tokens=False).input_ids[0]  # starting delimiter of response

    if "struq" in model_path or "secalign" in model_path:
        fastchat.conversation.register_conv_template(
            CustomConversation(
                name="struq",
                system_message="",
                roles=(inst_delm, resp_delm),
                sep="\n\n",
                sep2="</s>",
            )
        )

        suffix_manager = SuffixManager(
            tokenizer=tokenizer,
            use_system_instructions=False,
            conv_template=fastchat.conversation.get_conv_template("struq"),
        )
    else:
        fastchat.conversation.register_conv_template(
            CustomConversation(
                name="llama3",
                system_message="",
                roles=(inst_delm, resp_delm),
                sep="\n\n",
                sep2="<|eot_id|>",
            )
        )

        suffix_manager = SuffixManager(
            tokenizer=tokenizer,
            use_system_instructions=False,
            conv_template=fastchat.conversation.get_conv_template("llama3"),
        )

    cfg = setup_gcg_configs(args)
    attack = GCGAttack(
        config=cfg,
        model=model,
        tokenizer=tokenizer,
        eval_func=eval_func,
        suffix_manager=suffix_manager,
        not_allowed_tokens=None if cfg.allow_non_ascii else get_nonascii_toks(tokenizer),
    )

    data = [d for d in jload(args.data_path) if d["input"] != ""]

    open(attack_log_file, 'w').close()
    in_response = 0
    begin_with = 0
    total = 0
    for i, d in tqdm(enumerate(data), "Attacking samples..."):
        d_item = deepcopy(d)
        d_item["id"] = i
        # gcg attack
        new_d_item  = gcg(d_item, attack, cfg, data_delm)
        attacked_str = prompt_format['prompt_input'].format_map(d_item)

        sample_in_response, sample_begin_with, exact_equal, outputs = test_model_output(llm_inputs=[attacked_str],
                                                                                        model=model,
                                                                                        tokenizer=tokenizer,
                                                                                        attack_log_file=attack_log_file,
                                                                                        frontend_delimiters=frontend_delimiters,
                                                                                        pass_expert_labels=args.pass_expert_labels,
                                                                                        print_results=True)

        in_response += sample_in_response
        begin_with  += sample_begin_with
        total += 1
        print(
            f"\nGCG success rate {in_response/total} / {begin_with/total} / {exact_equal/total} (in-response / begin_with / exact_equal) on {args.model_name_or_path}, "
            f"delimiters {frontend_delimiters}, "
            f"training-attacks {training_attacks},"
            f"zero-shot defense {args.defense}\n"
        )

    with open(summary_path, "a") as outfile:
        outfile.write(f"GCG\t{in_response}\t{begin_with}\t{args.defense}_{TEST_INJECTED_WORD}\n")







