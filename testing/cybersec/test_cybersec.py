import json
from typing import Tuple, List, Dict
from testing.test import load_full_model, test_model_output, recursive_filter
import argparse
import json
from config import PROMPT_FORMAT
import os
from tqdm import tqdm


def format_prompt(elem: Dict, template: Dict) -> Tuple[str, str]:
    d_item = {
        "instruction": elem["test_case_prompt"],
        "input": elem["user_input"]
    }
    llm_input = template['prompt_input'].format_map(d_item)
    return llm_input


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Testing a model with a specific attack')
    parser.add_argument('-m', '--model_name_or_path', type=str, nargs="+")
    parser.add_argument('--data_path', type=str, default="./datasets/cyberseceval/prompt_injection.json")
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
    benign_response_name = os.path.join(log_path, f"predictions_on_cyberseceval.json")

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
        if data_point["risk_category"] == "security-violating": # only focus on logic-violating
            continue

        data_with_prob        = format_prompt(data_point, PROMPT_FORMAT[frontend_delimiters])
        _, _, _, injected_out = test_model_output([data_with_prob],
                                                  model,
                                                  tokenizer,
                                                  attack_log_file=None,
                                                  frontend_delimiters=frontend_delimiters,
                                                  pass_expert_labels=args.pass_expert_labels,
                                                  print_results=False)
        response: str = injected_out[0][0]
        data_point["response"] = response

        output.append(data_point)
        if i % save_step == 0 or i == len(dataset) - 1:
            with open(benign_response_name, "w", encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False)

