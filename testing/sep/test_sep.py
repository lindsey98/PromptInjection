
import json
from typing import Tuple, List, Dict
from testing.test import load_full_model, test_model_output, recursive_filter, apply_testtime_defense
import argparse
import json
from config import PROMPT_FORMAT
import os
from tqdm import tqdm
from difflib import SequenceMatcher
from attacks import *

def extract_overlap(s1, s2):
    match = SequenceMatcher(None, s1, s2).find_longest_match(0, len(s1), 0, len(s2))

    if match.size == 0:
        return ""

    return s1[match.a: match.a + match.size]

def _make_repetitive_fn(times):
    def repetitive_fn(d_item, injected_prompt):
        # Repeat the prompt N times with newlines or spaces
        # "Ignore previous... Ignore previous... Ignore previous..."
        injected_content = (injected_prompt + "\n") * times
        d_item['input'] += '\n\n' + injected_content.strip()
        return d_item

    repetitive_fn.__name__ = f"stress_repeat_{times}"
    return repetitive_fn

# Create variants for 2, 4, 6 ..20 repetitions
# This tests if the model yields to "force of numbers" or fails due to long-context confusion
REPEAT_COUNTS = range(2, 22, 2)
for n in REPEAT_COUNTS:
    globals()[f"stress_repeat_{n}"] = _make_repetitive_fn(n)


def _make_positional_injection_fn(percent):

    def injection_fn(d_item, injected_prompt):
        text = d_item['input']

        # Calculate the insertion index based on percentage
        # 0 means start, 100 means end
        if percent == 0:
            # Inject at the very beginning
            d_item['input'] = injected_prompt.capitalize() + "\n\n" + text
        elif percent == 100:
            # Inject at the very end (similar to naive)
            d_item['input'] = text + "\n\n" + injected_prompt.capitalize()
        else:
            # Inject in the middle
            split_idx = int(len(text) * (percent / 100.0))

            # Find the nearest whitespace to avoid cutting a word in half
            # Search forward from the calculated index
            while split_idx < len(text) and text[split_idx] not in [' ', '\n', '\t']:
                split_idx += 1

            # Construct new input: [Prefix] [Injection] [Suffix]
            prefix = text[:split_idx]
            suffix = text[split_idx:]
            d_item['input'] = f"{prefix}\n\n{injected_prompt.capitalize()}\n\n{suffix}"

        return d_item

    # Naming convention: inject_pos_10, inject_pos_50, etc.
    injection_fn.__name__ = f"inject_pos_{percent}"
    return injection_fn


# Generate functions for 0%, 10%, 20%, ..., 100%
POSITIONS = range(0, 101, 10)  # 0, 10, 20, ... 100
for p in POSITIONS:
    globals()[f"inject_pos_{p}"] = _make_positional_injection_fn(p)

def format_prompt(elem: Dict, template: Dict, mode: str = 'data_with_probe', attack: str="none", defense: str="none") -> Tuple[str, str]:

    if attack != "none":
        d_item = {
            "instruction": elem["system_prompt_clean"],
            "input": elem["prompt_clean"]
        }
        injected_prompt = extract_overlap(elem["prompt_instructed"],
                                          elem["system_prompt_instructed"])
        attack_method = eval(attack)
        d_item = attack_method(d_item, injected_prompt)
    else:
        if mode == 'data_with_probe':
            d_item = {
                "instruction": elem["system_prompt_clean"],
                "input": elem["prompt_instructed"]
            }
        elif mode == 'probe_with_task':
            d_item = {
                "instruction": elem["system_prompt_instructed"],
                "input": elem["prompt_clean"]
            }
        else:
            raise ValueError( f"Invalid mode for prompt formatting: {mode}. Valid modes are 'data_with_probe' or 'probe_with_task'.")

    llm_input = apply_testtime_defense(d_item, template, defense)
    return llm_input


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Testing a model with a specific attack')
    parser.add_argument('-m', '--model_name_or_path', type=str, nargs="+")
    parser.add_argument('--data_path', type=str, default="./datasets/SEP_dataset.json")
    parser.add_argument('-a', '--attack', type=str, default=["none"],
                        choices=["none",
                                 "inject_pos_0", "inject_pos_10", "inject_pos_20", "inject_pos_30", "inject_pos_40",
                                 "inject_pos_50", "inject_pos_60", "inject_pos_70", "inject_pos_80", "inject_pos_90", "inject_pos_100",
                                 "stress_repeat_2", "stress_repeat_4", "stress_repeat_6",
                                 "stress_repeat_8", "stress_repeat_10", "stress_repeat_12",
                                 "stress_repeat_14", "stress_repeat_16", "stress_repeat_18", "stress_repeat_20"],
                        nargs='+')
    parser.add_argument('-d', '--defense', type=str, default='none', # zero-shot defenses, never included in the adversarial training
                        choices=['none', 'sandwich', 'reminder', 'fakecompletion', 'thinkintervene',
                                 'spotlight_delimit', 'spotlight_datamark', 'spotlight_encode'],
                        help='Baseline test-time zero-shot prompting defense')
    parser.add_argument('--pass_expert_labels', default=False,
                        help="Whether to past expert labels instruction/data as an input", action='store_true')
    parser.add_argument('--customized_model_class', type=str, help="Customized model class", default='')
    args = parser.parse_args()
    args.model_name_or_path = args.model_name_or_path[0]

    model, tokenizer, frontend_delimiters, training_attacks = load_full_model(args.model_name_or_path,
                                                                              customized_model_class=args.customized_model_class)

    with open(args.data_path, 'r') as f:
        dataset = json.load(f)

    model_path = args.model_name_or_path
    log_path = f"{model_path}-log" if not os.path.exists(model_path) else model_path
    os.makedirs(log_path, exist_ok=True)

    for a in args.attack:
        if a == 'none':
            if args.defense == 'none':
                benign_response_name = os.path.join(log_path, f"predictions_on_sep.jsonl")
            else:
                benign_response_name = os.path.join(log_path, f"predictions_on_sep_{args.defense}.jsonl")
        else:
            if args.defense == 'none':
                benign_response_name = os.path.join(log_path, f"predictions_on_sep_{a}.jsonl")
            else:
                benign_response_name = os.path.join(log_path, f"predictions_on_sep_{a}_{args.defense}.jsonl")

        # Load previous results if available
        output = []
        start_idx = 0
        if os.path.exists(benign_response_name):
            with open(benign_response_name, "r", encoding="utf-8") as f:
                try:
                    output = json.load(f)
                    start_idx = len(output)
                    print(f"[INFO] Resuming from index {start_idx} (already processed {start_idx} items).")
                except json.JSONDecodeError:
                    print("[WARNING] Log file is corrupted or empty. Starting from scratch.")

        save_step = 50
        for i, data_point in enumerate(tqdm(dataset, desc=f"Processing dataset")):
            if i < start_idx:
                continue  # Skip already processed entries

            # First prompt with probe in data
            data_with_prob        = format_prompt(
                data_point,
                PROMPT_FORMAT[frontend_delimiters],
                mode='data_with_probe',
                attack=a,
                defense=args.defense
            )

            # Second prompt with probe in task
            instruction_with_prob = format_prompt(
                data_point, PROMPT_FORMAT[frontend_delimiters],
                mode='probe_with_task',
                attack="none",
                defense=args.defense
            )

            _, _, _, injected_out = test_model_output([data_with_prob],
                                                        model, tokenizer,
                                                        attack_log_file=None,
                                                        frontend_delimiters=frontend_delimiters,
                                                        pass_expert_labels=args.pass_expert_labels,
                                                        print_results=False
                                                      )
            data_with_prob_response: str = injected_out[0][0]

            _, _, _, clean_out = test_model_output([instruction_with_prob],
                                                     model, tokenizer,
                                                     attack_log_file=None,
                                                     frontend_delimiters=frontend_delimiters,
                                                     pass_expert_labels=args.pass_expert_labels,
                                                     print_results=False
                                                   )
            instruction_with_prob_response: str = clean_out[0][0]

            output.append({
                "output1_probe_in_data": data_with_prob_response,
                "output2_probe_in_task": instruction_with_prob_response,
                "model": args.customized_model_class,
                "instructions": {
                    "input_1": data_with_prob,
                    "input_2": instruction_with_prob
                },
                "data": data_point
            })

            if i % save_step == 0 or i == len(dataset) - 1:
                with open(benign_response_name, "w", encoding='utf-8') as f:
                    json.dump(output, f, ensure_ascii=False)
