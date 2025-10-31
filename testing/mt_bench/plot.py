import argparse
import json
import os
import re
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import numpy as np

def _regex_find_score(text):
    patterns = [
        r"[Rr]ating\s*[:=]\s*(\d+(\.\d+)?)",
        r"[Ss]core\s*[:=]\s*(\d+(\.\d+)?)",
        r"final[_\s-]?score\s*[:=]\s*(\d+(\.\d+)?)",
        r"overall\s*[:=]\s*(\d+(\.\d+)?)",
        r"\b(\d+(\.\d+)?)[ ]*/[ ]*10\b",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                return float(m.group(1))
            except:
                pass
    return None

def try_extract_score(rec):
    for k in ["judge_score", "score", "rating"]:
        if k in rec and isinstance(rec[k], (int, float)):
            return float(rec[k])
    for k in ["openai_output", "llm_output", "judge"]:
        if k in rec and isinstance(rec[k], dict):
            for kk in ["judge_score", "score", "rating"]:
                v = rec[k].get(kk)
                if isinstance(v, (int, float)):
                    return float(v)
            for kk in ["text", "message", "content", "raw"]:
                tv = rec[k].get(kk)
                if isinstance(tv, str):
                    s = _regex_find_score(tv)
                    if s is not None:
                        return s
    for k in ["text", "message", "content", "raw"]:
        tv = rec.get(k)
        if isinstance(tv, str):
            s = _regex_find_score(tv)
            if s is not None:
                return s
    return None

def load_qid2cat(qfile):
    qid2cat = {}
    with open(qfile, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = rec.get("question_id")
            cat = rec.get("category")
            if qid is not None and cat:
                qid2cat[qid] = cat
    return qid2cat

def load_cat_avg(jsonl_path, qid2cat=None, single_only=True):
    recs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    cat_scores = defaultdict(list)
    cat_counts = Counter()
    for r in recs:
        # single-turn only
        if single_only:
            if "turn" in r and r.get("turn") not in (None, 1):
                continue
            j = r.get("judge")
            if isinstance(j, (list, tuple)):
                j_lower = " ".join(map(str, j)).lower()
                if "multi-turn" in j_lower or "multi_turn" in j_lower or "multi" in j_lower:
                    continue
            elif isinstance(j, str):
                if "multi-turn" in j.lower() or "multi_turn" in j.lower() or "multi" in j.lower():
                    continue

        qid = r.get("question_id")
        cat = None
        if qid2cat is not None and qid in qid2cat:
            cat = qid2cat[qid]
        if cat is None:
            cat = r.get("category") or (r.get("question") or {}).get("category") or "overall"

        s = try_extract_score(r)
        if s is not None:
            cat_scores[cat].append(float(s))
            cat_counts[cat] += 1

    if not cat_scores:
        raise ValueError(f"[{jsonl_path}]。")
    cat_avg = {c: (sum(v) / len(v)) for c, v in cat_scores.items()}
    return cat_avg, cat_counts

def choose_k_axes(all_cat_counts, k=None, prefer=None):
    if prefer:
        wanted = [c for c in prefer if c]
        return wanted if (k is None or k <= 0 or len(wanted) == k) else wanted[:k]

    merged = Counter()
    for cc in all_cat_counts:
        merged.update(cc)

    cats = list(merged.keys())
    if not cats:
        return []

    if k is None or k <= 0:
        return cats
    if len(cats) <= k:
        return cats
    # top-k by frequency
    return [c for c, _ in merged.most_common(k)]

# ---------------------------
# plotting utilities
# ---------------------------
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
PASTEL    = ["#2a6fdb", "#e07a5f", "#3da27a", "#a270b3", "#e3b341", "#6fb1f4"]

def sentence_case(s: str) -> str:
    t = s.replace("_", " ").replace("-", " ").strip()
    return t[:1].upper() + t[1:].lower() if t else t

def radar_setup(ax, labels, vmin=0, vmax=10):
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_xticks(angles[:-1])

    pretty_labels = [sentence_case(l) for l in labels]
    ax.set_xticklabels(pretty_labels, fontsize= 20 , fontweight="bold")

    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([f"{int(x*(vmax-vmin)+vmin)}" for x in [0.25, 0.5, 0.75, 1.0]], fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle=(0, (3, 3)), color="#999999", alpha=0.6)
    return angles


def plot_polygon(ax, angles, values, color, label, vmin=0, vmax=10, alpha=0.16, lw=2.5, marker=None):
    vals = [(v - vmin) / (vmax - vmin) if vmax > vmin else 0.0 for v in values]
    vals += vals[:1]
    ax.plot(angles, vals, color=color, linewidth=lw, label=label, marker=marker, markersize=5)
    ax.fill(angles, vals, color=color, alpha=alpha)

def build_values(inputs, qid2cat, axes_wanted, single_only, vmax):
    cat_avgs_list, cat_counts_list = [], []
    for p in inputs:
        cat_avg, cat_counts = load_cat_avg(p, qid2cat=qid2cat, single_only=single_only)
        cat_avgs_list.append(cat_avg)
        cat_counts_list.append(cat_counts)
    values_list = []
    for cat_avg in cat_avgs_list:
        vals = []
        for c in axes_wanted:
            if c not in cat_avg:
                overall = cat_avg.get("overall", None)
                if overall is None:
                    overall = sum(cat_avg.values()) / len(cat_avg)
                vals.append(overall)
            else:
                vals.append(cat_avg[c])
        values_list.append(vals)
    return values_list, cat_counts_list

def main():
    '''
    E.g. --title-a "LLaMA-8B" --title-b "Mistral-7B" \
    --inputs-a meta-llama/Meta-Llama-3-8B-Instruct-SpclSpclSpcl-secalign-sep-none/gpt-4_judgement_on_mtbench.jsonl meta-llama/Meta-Llama-3-8B-Instruct-SpclSpclSpcl-struq-sep-none/gpt-4_judgement_on_mtbench.jsonl meta-llama/Meta-Llama-3-8B-Instruct-TextTextText-instfuse-sep-none-newdata-dpo/gpt-4_judgement_on_mtbench.jsonl meta-llama/Meta-Llama-3-8B-Instruct-TextTextText-ise-sep-none/gpt-4_judgement_on_mtbench.jsonl meta-llama/Meta-Llama-3-8B-Instruct-TextTextText-possep-sep-none/gpt-4_judgement_on_mtbench.jsonl meta-llama/Meta-Llama-3-8B-Instruct-log/gpt-4_judgement_on_mtbench.jsonl \
     --labels-a "SecAlign" "StruQ" "Ours" "ISE" "PFT" "Undefended" \
    --inputs-b mistralai/Mistral-7B-Instruct-v0.3-SpclSpclSpcl-secalign-sep-none/gpt-4_judgement_on_mtbench.jsonl mistralai/Mistral-7B-Instruct-v0.3-SpclSpclSpcl-struq-sep-none/gpt-4_judgement_on_mtbench.jsonl mistralai/Mistral-7B-Instruct-v0.3-TextTextTextMistral-instfuse-sep-none-newdata-dpo/gpt-4_judgement_on_mtbench.jsonl mistralai/Mistral-7B-Instruct-v0.3-TextTextTextMistral-ise-sep-none/gpt-4_judgement_on_mtbench.jsonl mistralai/Mistral-7B-Instruct-v0.3-TextTextTextMistral-possep-sep-none/gpt-4_judgement_on_mtbench.jsonl mistralai/Mistral-7B-Instruct-v0.3-log/gpt-4_judgement_on_mtbench.jsonl \
    --labels-b "SecAlign" "StruQ" "Ours" "ISE" "PFT" "Undefended" --out debug.png
    :return:
    '''
    parser = argparse.ArgumentParser()
    # Group A (left subplot: LLaMA-8B)
    parser.add_argument("--inputs-a", type=str, nargs="+", required=True, help="LHS model list")
    parser.add_argument("--labels-a", type=str, nargs="+", help="LHS model IDs, same length as --input-a")
    # Group B (right subplot: Mistral-7B)
    parser.add_argument("--inputs-b", type=str, nargs="+", required=True, help="RHS model list")
    parser.add_argument("--labels-b", type=str, nargs="+", help="RHS model IDs, same length as --input-b")

    parser.add_argument("--axes", type=str, default="", help="5 reasoning types")
    parser.add_argument("--vmax", type=float, default=10.0, help="max score")
    parser.add_argument("--out", type=str, default="pentagon_radar_1x2.png", help="out file")
    parser.add_argument("--title-a", type=str, default="LLaMA-8B", help="LHS title")
    parser.add_argument("--title-b", type=str, default="Mistral-7B", help="RHS title")
    parser.add_argument("--palette", type=str, default="okabe", choices=["okabe", "pastel"], help="color palette")
    parser.add_argument("--qfile", type=str, default="datasets/mtbench.jsonl", help="MT-Bench file path")
    parser.add_argument("--single-only", action="store_true", help="single-turn evaluation only")
    parser.add_argument("--k-axes", type=int, default=-1,
                        help="number of axes；5=full；<=0 all")

    parser.set_defaults(single_only=True)
    args = parser.parse_args()

    if args.labels_a and len(args.labels_a) != len(args.inputs_a):
        raise ValueError("--labels-a length shall be equal to --inputs-a")
    if args.labels_b and len(args.labels_b) != len(args.inputs_b):
        raise ValueError("--labels-b length shall be equal to --inputs-b")

    qid2cat = load_qid2cat(args.qfile)
    palette = OKABE_ITO if args.palette == "okabe" else PASTEL
    markers = ["o", "s", "^", "D", "P", "X"]

    # Build values and collect category counts for both groups
    # Use temporary wanted (we'll recompute after counts merged if not provided)
    tmp_wanted = ["math","coding","reasoning","writing","roleplay"]  # placeholder
    vals_a, counts_a = build_values(args.inputs_a, qid2cat, tmp_wanted, args.single_only, args.vmax)
    vals_b, counts_b = build_values(args.inputs_b, qid2cat, tmp_wanted, args.single_only, args.vmax)

    if args.axes.strip():
        wanted = [x.strip() for x in args.axes.split(",") if x.strip()]
    else:
        wanted = choose_k_axes(counts_a + counts_b, k=args.k_axes, prefer=None)
    # Rebuild values with the actual wanted axes
    vals_a, _ = build_values(args.inputs_a, qid2cat, wanted, args.single_only, args.vmax)
    vals_b, _ = build_values(args.inputs_b, qid2cat, wanted, args.single_only, args.vmax)

    # Plot 1x2
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 12, "axes.labelweight": "bold", "axes.titleweight": "bold"
    })
    fig, axs = plt.subplots(1, 2, figsize=(14, 7),
                            subplot_kw={"projection": "polar"})

    # Left (A): LLaMA-8B
    angles_a = radar_setup(axs[0], wanted, vmin=0, vmax=args.vmax)
    for i, vals in enumerate(vals_a):
        label = (args.labels_a[i] if args.labels_a else os.path.basename(args.inputs_a[i]).split(".")[0])
        color = palette[i % len(palette)]
        marker = markers[i % len(markers)]
        plot_polygon(axs[0], angles_a, vals, color, label, vmin=0, vmax=args.vmax, alpha=0.16, lw=2.6, marker=marker)
    axs[0].set_title(args.title_a, pad=12, fontsize=20, fontstyle="italic", fontweight="bold")
    leg0 = axs[0].legend(loc="upper right", bbox_to_anchor=(1.25, 1.10),
                         frameon=True, ncol=1, fontsize=15)
    for lh in leg0.legend_handles:
        lh.set_alpha(0.9)

    # Right (B): Mistral-7B
    angles_b = radar_setup(axs[1], wanted, vmin=0, vmax=args.vmax)
    for i, vals in enumerate(vals_b):
        label = (args.labels_b[i] if args.labels_b else os.path.basename(args.inputs_b[i]).split(".")[0])
        color = palette[i % len(palette)]
        marker = markers[i % len(markers)]
        plot_polygon(axs[1], angles_b, vals, color, label, vmin=0, vmax=args.vmax, alpha=0.16, lw=2.6, marker=marker)
    axs[1].set_title(args.title_b, pad=12, fontsize=20, fontstyle="italic", fontweight="bold")
    leg1 = axs[1].legend(loc="upper right", bbox_to_anchor=(1.25, 1.10),
                         frameon=True, ncol=1, fontsize=15)
    for lh in leg1.legend_handles:
        lh.set_alpha(0.9)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved 1x2 radar to {args.out}")
    print("Axes:", wanted)

if __name__ == "__main__":
    main()
