
import os
import sys
import json
import argparse
from collections import defaultdict
from typing import Dict, List

# Reuse category mapping from test_mmlu.py (keep in sync)
SUBJECT_TO_CATEGORY = {
    'formal_logic': 'humanities', 'high_school_european_history': 'humanities',
    'high_school_us_history': 'humanities', 'high_school_world_history': 'humanities',
    'international_law': 'humanities', 'jurisprudence': 'humanities',
    'logical_fallacies': 'humanities', 'moral_disputes': 'humanities',
    'moral_scenarios': 'humanities', 'philosophy': 'humanities',
    'prehistory': 'humanities', 'professional_law': 'humanities',
    'world_religions': 'humanities',
    'econometrics': 'social_sciences', 'high_school_geography': 'social_sciences',
    'high_school_government_and_politics': 'social_sciences',
    'high_school_macroeconomics': 'social_sciences',
    'high_school_microeconomics': 'social_sciences',
    'high_school_psychology': 'social_sciences', 'human_sexuality': 'social_sciences',
    'professional_psychology': 'social_sciences', 'public_relations': 'social_sciences',
    'security_studies': 'social_sciences', 'sociology': 'social_sciences',
    'us_foreign_policy': 'social_sciences',
    'abstract_algebra': 'stem', 'anatomy': 'stem', 'astronomy': 'stem',
    'college_biology': 'stem', 'college_chemistry': 'stem',
    'college_computer_science': 'stem', 'college_mathematics': 'stem',
    'college_physics': 'stem', 'computer_security': 'stem',
    'conceptual_physics': 'stem', 'electrical_engineering': 'stem',
    'elementary_mathematics': 'stem', 'high_school_biology': 'stem',
    'high_school_chemistry': 'stem', 'high_school_computer_science': 'stem',
    'high_school_mathematics': 'stem', 'high_school_physics': 'stem',
    'high_school_statistics': 'stem', 'machine_learning': 'stem',
    'business_ethics': 'other', 'clinical_knowledge': 'other',
    'college_medicine': 'other', 'global_facts': 'other',
    'human_aging': 'other', 'management': 'other', 'marketing': 'other',
    'medical_genetics': 'other', 'miscellaneous': 'other', 'nutrition': 'other',
    'professional_accounting': 'other', 'professional_medicine': 'other',
    'virology': 'other',
}


def compute_scores(results: List[Dict]) -> Dict:
    by_subject  = defaultdict(lambda: {"correct": 0, "total": 0})
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})
    overall     = {"correct": 0, "total": 0}

    for r in results:
        subject  = r["subject"]
        category = SUBJECT_TO_CATEGORY.get(subject, "other")
        c = bool(r.get("correct"))
        by_subject[subject]["total"]   += 1
        by_category[category]["total"] += 1
        overall["total"]               += 1
        if c:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file",
                    default="meta-llama/Meta-Llama-3-8B-Instruct-SpclSpclSpcl-struq-sep-none/predictions_on_mmlu_test.json")
    ap.add_argument("--by_subject", action="store_true",
                    help="Also print per-subject breakdown")
    ap.add_argument("--unparsed", action="store_true",
                    help="Show count of items where pred could not be parsed (None)")
    args = ap.parse_args()

    path = args.pred_file
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if not results:
        print("No results in file.")
        return

    scores = compute_scores(results)

    print(f"\nFile     : {path}")
    print(f"Items    : {scores['overall_total']}")
    print(f"Overall  : {scores['overall_accuracy']}%  "
          f"({scores['overall_correct']} / {scores['overall_total']})")

    print("\nBy category:")
    for cat, v in scores["by_category"].items():
        print(f"  {cat:20s}: {v['accuracy']:6.2f}%  ({v['correct']:>5d} / {v['total']:>5d})")

    # Subject coverage info
    subjects_seen = sorted({r["subject"] for r in results})
    print(f"\nSubjects covered: {len(subjects_seen)} / 57")

    if args.by_subject:
        print("\nBy subject:")
        for sub, v in scores["by_subject"].items():
            print(f"  {sub:45s}: {v['accuracy']:6.2f}%  ({v['correct']:>4d} / {v['total']:>4d})")

    if args.unparsed:
        n_none = sum(1 for r in results if r.get("pred") is None)
        print(f"\nUnparsed predictions (pred=None): {n_none} / {len(results)}")


if __name__ == "__main__":
    main()