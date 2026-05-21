# monitor_results.py
import json
import sys
import time
from pathlib import Path
from collections import defaultdict


def monitor_logdir(logdir: Path):
    # suite -> attack -> {utility: [...], security: [...]}
    results = defaultdict(lambda: defaultdict(lambda: {"utility": [], "security": []}))

    for json_file in logdir.rglob("*.json"):
        parts = json_file.relative_to(logdir).parts
        if len(parts) < 4:
            continue
        suite, _user_task, attack, _inj_task_file = parts[0], parts[1], parts[2], parts[3]

        try:
            with open(json_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if "utility" in data:
            results[suite][attack]["utility"].append(bool(data["utility"]))
        if "security" in data:
            results[suite][attack]["security"].append(bool(data["security"]))

    if not results:
        print("  (no results yet)")
        return

    for suite in sorted(results):
        print(f"\n{suite}:")
        for attack in sorted(results[suite]):
            util = results[suite][attack]["utility"]
            sec = results[suite][attack]["security"]
            if attack == "none":
                # benign: only utility is meaningful
                if util:
                    print(f"  [benign]                    utility {sum(util)}/{len(util)} = {sum(util)/len(util)*100:5.1f}%")
            else:
                # under attack: utility = utility-under-attack, security = defense rate (1 - ASR)
                line = f"  [{attack:20s}]"
                if util:
                    line += f"  util-under-attack {sum(util)}/{len(util)} = {sum(util)/len(util)*100:5.1f}%"
                if sec:
                    asr = 1 - sum(sec)/len(sec)
                    line += f"  ASR {len(sec)-sum(sec)}/{len(sec)} = {asr*100:5.1f}%"
                print(line)

if __name__ == "__main__":
    default = "./agentdojo_runs/local_Qwen_Qwen3-30B-A3B-Instruct-2507-TextTextTextQwen-instfuse-alpaca-none_"
    # default = "./agentdojo_runs/local_Qwen_Qwen3-30B-A3B-Instruct-2507-TextTextTextQwen-instfuse-alpaca-none"

    logdir = Path(sys.argv[1] if len(sys.argv) > 1 else default)

    try:
        while True:
            print("\033[2J\033[H", end="")
            print(f"Monitor: {logdir}")
            print(f"Refresh: every 5s   (Ctrl+C to quit)")
            monitor_logdir(logdir)
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nstopped.")