
import json
from typing import Tuple, List, Dict
from testing.test import load_full_model, test_model_output, recursive_filter
import argparse
import json
from config import PROMPT_FORMAT, DELIMITERS, DEFAULT_SYSTEM_PROMPT
import os
from tqdm import tqdm


def format_prompt(elem: Dict, template: Dict) -> Tuple[str, str]:
    d_item = {"instruction": elem,
              "input": ""}
    llm_input = template['prompt_input'].format_map(d_item)
    return llm_input


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Testing a model with a specific attack')
    parser.add_argument('-m', '--model_name_or_path', type=str, nargs="+")
    parser.add_argument('--data_path', type=str, default="./datasets/ifeval/input_data.jsonl")
    parser.add_argument('--customized_model_class', type=str, help="Customized model class", default='')
    parser.add_argument("--base_model_path", type=str, default=None,
                   help="Explicit base model path; required when adapter path "
                        "does not encode the base path via the usual suffix convention.")
    args = parser.parse_args()
    args.model_name_or_path = args.model_name_or_path[0]

    model, tokenizer, frontend_delimiters, training_attacks = load_full_model(args.model_name_or_path,
                                                                              customized_model_class=args.customized_model_class,
                                                                              base_model_path=args.base_model_path)
    delm = DELIMITERS[frontend_delimiters]
    fmt = dict(PROMPT_FORMAT[frontend_delimiters])
    if len(delm) == 4:
        fmt['prompt_input_tool'] = (
                delm[0] + DEFAULT_SYSTEM_PROMPT + "\n\n"
                + delm[1] + "\n{instruction}\n\n"
                + delm[2] + "\n{input}\n\n"
                + delm[3] + "\n"
        )

    model_path = args.model_name_or_path
    log_path = f"{model_path}-log" if not os.path.exists(model_path) else model_path
    os.makedirs(log_path, exist_ok=True)
    benign_response_name = os.path.join(log_path, f"predictions_on_ifeval.jsonl")

    # Load previous results if available
    output = []
    start_idx = 0
    if os.path.exists(benign_response_name):
        with open(benign_response_name, 'r', encoding='utf-8') as f_out:
            try:
                output = [json.loads(line) for line in f_out if line.strip()]
                start_idx = len(output)
                print(f"[INFO] Resuming from index {start_idx} (already processed {start_idx} items).")
            except json.JSONDecodeError:
                print("[WARNING] Log file is corrupted or empty. Starting from scratch.")

    save_step = 50

    with open(args.data_path, 'r', encoding='utf-8') as f_in:
        dataset = [json.loads(line) for line in f_in if line.strip()]

    for i, data_point in enumerate(tqdm(dataset, desc=f"Processing dataset")):
        if i < start_idx:
            continue  # Skip already processed entries

        # First prompt with probe in data
        prompt = data_point["prompt"]
        clean_prompt        = format_prompt(prompt, fmt)
        print(clean_prompt)
        _, _, _, clean_out = test_model_output([clean_prompt],
                                                  model,
                                                  tokenizer,
                                                  attack_log_file=None,
                                                  print_results=True,
                                               frontend_delimiters=frontend_delimiters,)
        response: str = clean_out[0][0]
        output.append({
            "prompt": prompt,
            "response": response,
        })

        if i % save_step == 0 or i == len(dataset) - 1:
            with open(benign_response_name, 'w', encoding='utf-8') as outfile:
                for entry in output:
                    json.dump(entry, outfile)
                    outfile.write('\n')
