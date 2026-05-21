import fastchat
from ml_collections import config_dict
from nanogcg.gcg import GCGConfig, run
from config import (
    DELIMITERS,
    PROMPT_FORMAT,
    SYS_INPUT,
    TEST_INJECTED_PROMPT,
    TEST_INJECTED_WORD,
)
import dataclasses
import torch
from data_generation.sft_data_loader import jload, _tokenize_fn
from testing.test import load_full_model
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
from tqdm import tqdm
import csv
import sys
os.environ['TRANSFORMERS_CACHE'] = "/mnt/sda/hf_cache"


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
    parser.add_argument('--customized_model_class', type=str, help="Customized model class", default='')
    args = parser.parse_args()
    args.model_name_or_path = args.model_name_or_path[0]

    model, tokenizer, frontend_delimiters, training_attacks = load_full_model(args.model_name_or_path, customized_model_class=args.customized_model_class)
    data_path = args.data_path
    data = jload(data_path)
    target_dataset_path = './advprompter/data/prompt_injections/dataset/test_%s.csv' % \
                (frontend_delimiters + '_' + data_path.replace('.', '_').replace('/', '_'))
    os.makedirs(os.path.dirname(target_dataset_path), exist_ok=True)
    prompt_format = DELIMITERS[frontend_delimiters][0] + "\n{instruction}\n\n" + DELIMITERS[frontend_delimiters][1] + "\n{input}"
    with open(target_dataset_path, "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['instruct', 'target'])
        for d in data:
            if d['input'] == '':
                continue
            current_sample = deepcopy(d)
            if current_sample['input'][-1] not in ['.', '!', '?']:
                current_sample['input'] += '.'
            current_sample['input'] += ' ' + TEST_INJECTED_PROMPT
            llm_input = prompt_format.format_map(current_sample)
            writer.writerow([llm_input, TEST_INJECTED_WORD + '!'])

    # target_llm_config = 'spcl_delm_llm'
    # cmd = 'python main.py --config-name=test target_llm=' + target_llm_config + \
    #       ' target_llm.llm_params.model_name=' + model_path.split('/')[-1] + \
    #       ' target_llm.llm_params.checkpoint=../' + base_model_path + \
    #       ' train.dataset_pth=' + target_dataset_path.replace('advprompter', '.') + \
    #       ' eval.data.dataset_pth_dct.train=' + target_dataset_path.replace('advprompter', '.')
    #
    # print(cmd)
    # process = subprocess.Popen('cd advprompter\n' + cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    # while True:
    #     output = process.stdout.readline()
    #     if output == b'' and process.poll() is not None: break
    #     if output: print(output.decode().strip())
