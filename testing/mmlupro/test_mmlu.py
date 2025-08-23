from datasets import load_dataset
import os
import torch
import random
from tqdm import tqdm
from typing import Tuple, List, Dict
from testing.test import load_full_model, test_model_output, recursive_filter
from config import PROMPT_FORMAT
import argparse
import json
import re

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
max_model_length = 4096
max_new_tokens = 2048
random.seed(12345)

def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)

def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)

def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None

def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    return res_df

def select_by_category(df, subject):
    res = []
    for each in df:
        if each["category"] == subject:
            res.append(each)
    return res

def compute_res(res):
    accu, corr, wrong = 0.0, 0.0, 0.0

    for each in res:
        if not each["pred"]:
            x = random.randint(0, len(each["options"]) - 1)
            if x == each["answer_index"]:
                corr += 1
            else:
                wrong += 1
        elif each["pred"] == each["answer"]:
            corr += 1
        else:
            wrong += 1

    if corr + wrong == 0:
        return 0.0, 0.0, 0.0
    accu = corr / (corr + wrong)
    return accu, corr, wrong


def generate_cot_prompt(val_df, curr, k, template):
    prompt = ""
    with open(f"./datasets/mmlu/initial_prompt.txt", "r") as fi:
        for line in fi.readlines():
            prompt += line
    subject = curr["category"]
    val_df = select_by_category(val_df, subject)
    val_df = val_df[: k]
    prompt = prompt.replace("{$}", subject) + "\n"
    for example in val_df:
        prompt += format_cot_example(example, including_answer=True) # few-shot examples
    prompt += format_cot_example(curr, including_answer=False)
    d_item = {"instruction": prompt,
              "input": ""}
    llm_input = template['prompt_input'].format_map(d_item)
    return llm_input

def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace("A: Let's think step by step.",
                                                     "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt

@torch.inference_mode()
def eval_cot(subject, model, tokenizer, val_df, test_df, output_path, template):
    global choices
    print("evaluating " + subject)
    inference_batches = []

    if os.path.exists(output_path):
        with open(output_path, "r") as fo:
            res = json.load(fo)
        if res:
            accu, corr, wrong = compute_res(res)
            print("this batch accu is: {}, corr: {}, wrong: {}\n".format(str(accu), str(corr), str(wrong)))
            return accu, corr, wrong

    for i in tqdm(range(len(test_df))):
        k = args.ntrain
        curr = test_df[i]
        prompt_length_ok = False
        prompt = None
        while not prompt_length_ok:
            prompt = generate_cot_prompt(val_df, curr, k, template)
            inputs = tokenizer(prompt, return_tensors="pt")
            length = len(inputs["input_ids"][0])
            if length < max_model_length - max_new_tokens:
                prompt_length_ok = True
            k -= 1 # reduce no. few-shot if exceeds prompt allowed length
        inference_batches.append(prompt)

    if len(inference_batches):
        _, _, _, response_batch = test_model_output(inference_batches,
                                               model,
                                               tokenizer,
                                               attack_log_file=None,
                                               frontend_delimiters=frontend_delimiters,
                                               pass_expert_labels=args.pass_expert_labels,
                                               print_results=False)
        response_batch: List[str] = [x[0] for x in response_batch]
        pred_batch = [extract_answer(x) for x in response_batch]

        res = []
        for j, curr in enumerate(test_df):
            curr["pred"] = pred_batch[j]
            curr["model_outputs"] = response_batch[j]
            res.append(curr)
        with open(output_path, "w") as fo:
            fo.write(json.dumps(res))

        accu, corr, wrong = compute_res(res)
        print("this batch accu is: {}, corr: {}, wrong: {}\n".format(str(accu), str(corr), str(wrong)))

    return accu, corr, wrong

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='Testing a model with a specific attack')
    parser.add_argument('-m', '--model_name_or_path', type=str, nargs="+")
    parser.add_argument("--ntrain", "-k", type=int, default=0) ## MMLU-0
    parser.add_argument('--pass_expert_labels', default=False,
                        help="Whether to past expert labels instruction/data as an input", action='store_true')
    parser.add_argument('--customized_model_class', type=str, help="Customized model class", default='')
    args = parser.parse_args()
    args.model_name_or_path = args.model_name_or_path[0]

    model, tokenizer, frontend_delimiters, training_attacks = load_full_model(args.model_name_or_path,
                                                                              customized_model_class=args.customized_model_class)

    model_path = args.model_name_or_path
    log_path = f"{model_path}-log" if not os.path.exists(model_path) else model_path
    summary_path = os.path.join(log_path, "mmlu_summary.txt")

    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = preprocess(dataset["test"]), preprocess(dataset["validation"])

    all_subjects = []
    for each in test_df:
        if each["category"] not in all_subjects:
            all_subjects.append(each["category"])

    selected_subjects = all_subjects
    print("selected subjects:\n" + "\n".join(selected_subjects))
    sta_dict = {}
    selected_subjects = sorted(selected_subjects)

    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------category level sta------\n")

    for subject in selected_subjects:
        if subject not in sta_dict:
            sta_dict[subject] = {
                "corr": 0.0,
                "wrong": 0.0,
                "accu": 0.0
            }
        selected_test_df = select_by_category(test_df, subject)
        selected_val_df  = select_by_category(val_df, subject)
        output_path = os.path.join(log_path, "mmlu_{}.json".format(subject))

        acc, corr_count, wrong_count = eval_cot(subject, model, tokenizer,
                                                selected_val_df, selected_test_df,
                                                output_path, PROMPT_FORMAT[frontend_delimiters])
        sta_dict[subject]["corr"] = corr_count
        sta_dict[subject]["wrong"] = wrong_count
        sta_dict[subject]["accu"] = acc
        print(sta_dict)

        with open(os.path.join(summary_path), 'a') as f:
            f.write("Average accuracy {:.4f} - {}\n".format(sta_dict[subject]["accu"], subject))

    total_corr, total_wrong = 0.0, 0.0
    for k, v in sta_dict.items():
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    total_accu = total_corr / (total_corr + total_wrong + 0.000001)
    sta_dict["total"] = {"corr": total_corr, "wrong": total_wrong, "accu": total_accu}
    print(sta_dict)