import json
import os
import argparse
import sys
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm

import torch
from datasets import load_dataset

from testing.test import load_full_model, test_model_output, apply_testtime_defense
from config import PROMPT_FORMAT, DEFAULT_SYSTEM_PROMPT


# All 57 MMLU subjects
ALL_SUBJECTS = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
    'clinical_knowledge', 'college_biology', 'college_chemistry',
    'college_computer_science', 'college_mathematics', 'college_medicine',
    'college_physics', 'computer_security', 'conceptual_physics',
    'econometrics', 'electrical_engineering', 'elementary_mathematics',
    'formal_logic', 'global_facts', 'high_school_biology',
    'high_school_chemistry', 'high_school_computer_science',
    'high_school_european_history', 'high_school_geography',
    'high_school_government_and_politics', 'high_school_macroeconomics',
    'high_school_mathematics', 'high_school_microeconomics',
    'high_school_physics', 'high_school_psychology', 'high_school_statistics',
    'high_school_us_history', 'high_school_world_history', 'human_aging',
    'human_sexuality', 'international_law', 'jurisprudence',
    'logical_fallacies', 'machine_learning', 'management', 'marketing',
    'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios',
    'nutrition', 'philosophy', 'prehistory', 'professional_accounting',
    'professional_law', 'professional_medicine', 'professional_psychology',
    'public_relations', 'security_studies', 'sociology', 'us_foreign_policy',
    'virology', 'world_religions',
]

ANSWER_LABELS = ['A', 'B', 'C', 'D']

SUBJECT_TO_CATEGORY = {
    # Humanities
    'formal_logic': 'humanities', 'high_school_european_history': 'humanities',
    'high_school_us_history': 'humanities', 'high_school_world_history': 'humanities',
    'international_law': 'humanities', 'jurisprudence': 'humanities',
    'logical_fallacies': 'humanities', 'moral_disputes': 'humanities',
    'moral_scenarios': 'humanities', 'philosophy': 'humanities',
    'prehistory': 'humanities', 'professional_law': 'humanities',
    'world_religions': 'humanities',
    # Social sciences
    'econometrics': 'social_sciences', 'high_school_geography': 'social_sciences',
    'high_school_government_and_politics': 'social_sciences',
    'high_school_macroeconomics': 'social_sciences',
    'high_school_microeconomics': 'social_sciences',
    'high_school_psychology': 'social_sciences', 'human_sexuality': 'social_sciences',
    'professional_psychology': 'social_sciences', 'public_relations': 'social_sciences',
    'security_studies': 'social_sciences', 'sociology': 'social_sciences',
    'us_foreign_policy': 'social_sciences',
    # STEM
    'abstract_algebra': 'stem', 'anatomy': 'stem', 'astronomy': 'stem',
    'college_biology': 'stem', 'college_chemistry': 'stem',
    'college_computer_science': 'stem', 'college_mathematics': 'stem',
    'college_physics': 'stem', 'computer_security': 'stem',
    'conceptual_physics': 'stem', 'electrical_engineering': 'stem',
    'elementary_mathematics': 'stem', 'high_school_biology': 'stem',
    'high_school_chemistry': 'stem', 'high_school_computer_science': 'stem',
    'high_school_mathematics': 'stem', 'high_school_physics': 'stem',
    'high_school_statistics': 'stem', 'machine_learning': 'stem',
    # Other
    'business_ethics': 'other', 'clinical_knowledge': 'other',
    'college_medicine': 'other', 'global_facts': 'other',
    'human_aging': 'other', 'management': 'other', 'marketing': 'other',
    'medical_genetics': 'other', 'miscellaneous': 'other', 'nutrition': 'other',
    'professional_accounting': 'other', 'professional_medicine': 'other',
    'virology': 'other',
}


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_mmlu_prompt(item: Dict, template: Dict, defense: str,
                       few_shot_examples: List[Dict] | None = None) -> str:
    """
    Format a single MMLU question into the flat string expected by
    _tokenize_fn, following the same apply_testtime_defense pattern
    as test_sep.py.

    The instruction is a system-level task description.
    The input carries the question + choices (+ optional few-shot examples).
    """
    choices_str = "\n".join(
        f"{label}. {choice}"
        for label, choice in zip(ANSWER_LABELS, item["choices"])
    )

    # Build few-shot prefix if provided
    few_shot_str = ""
    if few_shot_examples:
        lines = []
        for ex in few_shot_examples:
            ex_choices = "\n".join(
                f"{l}. {c}" for l, c in zip(ANSWER_LABELS, ex["choices"])
            )
            lines.append(
                f"Question: {ex['question']}\n{ex_choices}\nAnswer: {ANSWER_LABELS[ex['answer']]}"
            )
        few_shot_str = "\n\n".join(lines) + "\n\n"

    # For MMLU there is no untrusted "input" data — the full question goes in
    # the instruction slot. We pass input="" so apply_testtime_defense falls
    # through to prompt_no_input (both 3-role and 4-role).
    d_item = {
        "instruction": (
            "The following are multiple choice questions (with answers) about "
            f"{item['subject'].replace('_', ' ')}. "
            "Answer with only the letter (A, B, C, or D) of the correct option.\n\n"
            f"{few_shot_str}"
            f"Question: {item['question']}\n"
            f"{choices_str}\n"
            f"Your Answer:"
        ),
        "input": "",   # was: () — empty tuple is truthy, caused KeyError in format_map
    }
    return apply_testtime_defense(d_item, template, defense)


def extract_answer(response: str) -> str | None:
    """
    Parse the model's response to extract the answer label (A/B/C/D).
    Handles various formats: bare letter, 'Answer: A', '(A)', etc.
    """
    response = response.strip()
    # Bare letter at start
    if response and response[0].upper() in ANSWER_LABELS:
        return response[0].upper()
    # 'Answer: X' pattern
    import re
    m = re.search(r'\b([ABCD])\b', response.upper())
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_scores(results: List[Dict]) -> Dict:
    """
    Compute per-subject, per-category, and overall accuracy.
    Mirrors the output structure used in test_injecagent.py get_score().
    """
    by_subject  = defaultdict(lambda: {"correct": 0, "total": 0})
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})
    overall     = {"correct": 0, "total": 0}

    for r in results:
        subject  = r["subject"]
        category = SUBJECT_TO_CATEGORY.get(subject, "other")
        correct  = r["correct"]

        by_subject[subject]["total"]   += 1
        by_category[category]["total"] += 1
        overall["total"]               += 1

        if correct:
            by_subject[subject]["correct"]   += 1
            by_category[category]["correct"] += 1
            overall["correct"]               += 1

    def acc(d):
        return round(d["correct"] / d["total"] * 100, 2) if d["total"] else 0.0

    return {
        "overall_accuracy": acc(overall),
        "overall_correct":  overall["correct"],
        "overall_total":    overall["total"],
        "by_category": {k: {"accuracy": acc(v), **v}
                        for k, v in sorted(by_category.items())},
        "by_subject":  {k: {"accuracy": acc(v), **v}
                        for k, v in sorted(by_subject.items())},
    }


def summary_results(summary_path: str, scores: Dict,
                    model_path: str, defense: str) -> None:
    """Append one-line TSV summary, same style as test.py / test_injecagent.py."""
    print()
    print(f"  MMLU Results")
    print(f"  Model    : {model_path}")
    print(f"  Defense  : {defense}")
    print(f"  Overall  : {scores['overall_accuracy']}%  "
          f"({scores['overall_correct']} / {scores['overall_total']})")
    for cat, v in scores["by_category"].items():
        print(f"  {cat:20s}: {v['accuracy']}%")
    print()

    header_needed = not os.path.exists(summary_path)
    with open(summary_path, "a") as f:
        if header_needed:
            f.write("attack\taccuracy\tcorrect\ttotal\tdefense\tmodel\n")
        f.write(
            f"mmlu\t"
            f"{scores['overall_accuracy']}\t"
            f"{scores['overall_correct']}\t"
            f"{scores['overall_total']}\t"
            f"{defense}\t"
            f"{model_path}\n"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Testing a model on MMLU')
    parser.add_argument('-m', '--model_name_or_path', type=str, nargs="+")
    parser.add_argument("--base_model_path", type=str, default=None,
                   help="Explicit base model path; required when adapter path "
                        "does not encode the base path via the usual suffix convention.")
    parser.add_argument('-d', '--defense', type=str, default='none',
                        choices=['none', 'sandwich', 'reminder', 'fakecompletion',
                                 'thinkintervene', 'spotlight_delimit',
                                 'spotlight_datamark', 'spotlight_encode'],
                        help='Baseline test-time zero-shot prompting defense')
    parser.add_argument('--subjects', type=str, nargs='+', default=None,
                        help='Subset of MMLU subjects to evaluate (default: all 57)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['test', 'validation', 'dev'],
                        help='Dataset split to evaluate on (default: test)')
    parser.add_argument('--num_few_shot', type=int, default=0,
                        help='Number of few-shot examples from dev split (default: 0)')
    parser.add_argument('--num_samples', type=int, default=-1,
                        help='Max samples per subject (-1 = all)')
    parser.add_argument('--customized_model_class', type=str, default='',
                        help='Custom model class name')
    parser.add_argument('--load_as_adapter', action='store_true')

    args = parser.parse_args()
    args.model_name_or_path = args.model_name_or_path[0]

    model, tokenizer, frontend_delimiters, _ = load_full_model(
        args.model_name_or_path,
        customized_model_class=args.customized_model_class,
        load_as_adapter=args.load_as_adapter,
        base_model_path=args.base_model_path
    )
    from config import DELIMITERS
    _delm = DELIMITERS[frontend_delimiters]
    template = dict(PROMPT_FORMAT[frontend_delimiters])
    if len(_delm) == 4:
        # 4-role: question goes in user slot (trusted), input slot is empty
        template['prompt_input_tool'] = (
            _delm[0] + DEFAULT_SYSTEM_PROMPT + "\n\n"
            + _delm[1] + "\n{instruction}\n\n"
            + _delm[2] + "\n{input}\n\n"
            + _delm[3] + "\n"
        )

    model_path = args.model_name_or_path
    log_path   = f"{model_path}-log" if not os.path.exists(model_path) else model_path
    os.makedirs(log_path, exist_ok=True)

    subjects = args.subjects if args.subjects else ALL_SUBJECTS

    # Output file names (same naming convention as test_sep.py)
    if args.defense == 'none':
        response_name = os.path.join(log_path, f"predictions_on_mmlu_{args.split}.json")
    else:
        response_name = os.path.join(log_path, f"predictions_on_mmlu_{args.split}_{args.defense}.json")
    summary_path = os.path.join(log_path, "summary.tsv")

    # Resume from existing results
    all_results: List[Dict] = []
    done_keys = set()   # (subject, idx) already evaluated
    if os.path.exists(response_name):
        with open(response_name, "r", encoding="utf-8") as f:
            try:
                all_results = json.load(f)
                done_keys   = {(r["subject"], r["idx"]) for r in all_results}
                print(f"[INFO] Resuming: found {len(all_results)} existing results.")
            except json.JSONDecodeError:
                print("[WARNING] Results file is corrupted. Starting from scratch.")

    # Evaluate subject by subject
    for subject in subjects:
        print(f"\n→ Subject: {subject}")

        dataset = load_dataset("cais/mmlu", subject, trust_remote_code=True)
        split_data = list(dataset[args.split])

        if args.num_samples > 0:
            split_data = split_data[:args.num_samples]

        # Load few-shot examples from dev split if requested
        few_shot_examples = []
        if args.num_few_shot > 0:
            dev_data = list(dataset["dev"])
            few_shot_examples = dev_data[:args.num_few_shot]
            for ex in few_shot_examples:
                ex["subject"] = subject

        # Build inputs only for items not yet evaluated
        pending_idxs   = []
        pending_inputs = []
        for idx, item in enumerate(split_data):
            if (subject, idx) in done_keys:
                continue
            item["subject"] = subject
            pending_idxs.append(idx)
            pending_inputs.append(
                format_mmlu_prompt(item, template, args.defense, few_shot_examples)
            )

        if not pending_inputs:
            print(f"  All {len(split_data)} items already done, skipping.")
            continue

        # Run inference via test_model_output (same as test_sep.py)
        _, _, _, raw_outputs = test_model_output(
            pending_inputs,
            model,
            tokenizer,
            attack_log_file=None,
            print_results=False,
            frontend_delimiters=frontend_delimiters,
        )

        # Score and collect results
        subject_correct = 0
        for idx, item_idx in enumerate(pending_idxs):
            item     = split_data[item_idx]
            response = raw_outputs[idx][0]
            pred     = extract_answer(response)
            gold     = ANSWER_LABELS[item["answer"]]
            correct  = (pred == gold)
            if correct:
                subject_correct += 1

            all_results.append({
                "subject":  subject,
                "idx":      item_idx,
                "question": item["question"],
                "choices":  item["choices"],
                "gold":     gold,
                "pred":     pred,
                "response": response,
                "correct":  correct,
            })
            done_keys.add((subject, item_idx))

        subject_total = len(pending_idxs)
        print(f"  {subject_correct}/{subject_total} = "
              f"{subject_correct/subject_total*100:.1f}% (this run)")

        # Save after each subject (same save pattern as test_sep.py)
        with open(response_name, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Final scoring and summary
    scores = compute_scores(all_results)
    summary_results(summary_path, scores, model_path, args.defense)

    # Also save the stat JSON (same pattern as test_cyberseceval.py)
    stat_name = response_name.replace("predictions_on_", "stat_")
    with open(stat_name, "w", encoding="utf-8") as f:
        json.dump({model_path: scores}, f, ensure_ascii=False, indent=2)
    print(f"Stat saved to {stat_name}")