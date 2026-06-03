"""Check that the SEP (or Alpaca) train and eval sets are disjoint at the
content level — guards against train/test leakage that would inflate results.

The DRIP training set (datasets/sep/train_dataset.json) and the SEP eval set
(datasets/SEP_dataset.json) use different field names but the same underlying
(system prompt, clean data) content. This script normalizes that content and
reports any overlap. It is read-only and makes no changes.

Usage:
    python data_generation/check_sep_leakage.py \
        --train datasets/sep/train_dataset.json \
        --eval  datasets/SEP_dataset.json

    # Alpaca injection split (override the field names):
    python data_generation/check_sep_leakage.py \
        --train datasets/alpaca_data.json \
        --eval  datasets/davinci_003_outputs.json \
        --sys-fields instruction --data-fields input

Exit code is non-zero if any (system, clean-data) pair appears in both sets,
so it can also be used as a CI assertion.
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Candidate field names (first match wins) for the two halves of the key.
SYS_FIELDS = ["system_prompt_clean", "system_prompt", "instruction"]
DATA_FIELDS = ["data_prompt_clean", "prompt_clean", "input", "data"]


def _load(path: Path):
    text = path.read_text()
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(text)
    # JSONL fallback
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _pick(item: dict, fields):
    for f in fields:
        if f in item and item[f] not in (None, ""):
            return item[f]
    return ""


def _keys(items, sys_fields, data_fields):
    sys_set, data_set, both_set = set(), set(), set()
    for it in items:
        s = _norm(_pick(it, sys_fields))
        d = _norm(_pick(it, data_fields))
        if s:
            sys_set.add(s)
        if d:
            data_set.add(d)
        if s or d:
            both_set.add((s, d))
    return sys_set, data_set, both_set


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--eval", required=True, type=Path)
    ap.add_argument("--sys-fields", nargs="+", default=SYS_FIELDS,
                    help="Candidate field names for the system/instruction text.")
    ap.add_argument("--data-fields", nargs="+", default=DATA_FIELDS,
                    help="Candidate field names for the clean data/input text.")
    ap.add_argument("--show", type=int, default=3, help="How many overlapping examples to print.")
    args = ap.parse_args()

    train, ev = _load(args.train), _load(args.eval)
    print(f"train: {len(train)} items  |  eval: {len(ev)} items")

    t_sys, t_data, t_both = _keys(train, args.sys_fields, args.data_fields)
    e_sys, e_data, e_both = _keys(ev, args.sys_fields, args.data_fields)

    both = t_both & e_both
    data_only = t_data & e_data
    sys_only = t_sys & e_sys

    def pct(n, d):
        return f"{100 * n / d:.1f}%" if d else "n/a"

    print(f"\n(system, clean-data) pairs in BOTH : {len(both):5d}  "
          f"({pct(len(both), len(e_both))} of eval)")
    print(f"clean-data strings in BOTH          : {len(data_only):5d}  "
          f"({pct(len(data_only), len(e_data))} of eval)")
    print(f"system strings in BOTH              : {len(sys_only):5d}  "
          f"({pct(len(sys_only), len(e_sys))} of eval)")

    if both and args.show:
        print(f"\n--- {min(args.show, len(both))} overlapping (system, data) examples ---")
        for s, d in list(both)[:args.show]:
            print(f"  SYS : {s[:100]}")
            print(f"  DATA: {d[:100]}\n")

    if both:
        print("LEAKAGE DETECTED: eval items also appear in train (see above).")
        sys.exit(1)
    print("OK: no (system, clean-data) overlap between train and eval.")


if __name__ == "__main__":
    main()
