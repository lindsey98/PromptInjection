import os

import re
import gc
import json
import copy
import difflib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from tqdm import tqdm

import transformers
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader

# ---- Project modules ----
from testing.test import load_delimiters
from data_generation.sft_data_loader import SupervisedDataset, _tokenize_fn
from modeling import *  # noqa: F401,F403
from modeling.llama_drip import compute_expert_labels_from_input_ids
from config import DELIMITERS, IGNORE_INDEX


# =============================================================================
# Helpers: probe span localization
# =============================================================================

def _norm(s: str) -> str:
    return "".join(s.lower().split())

def _fuzzy(a: str, b: str, thr: float = 0.8) -> bool:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio() >= thr

def find_probe_span(tokenizer, input_ids, probe_string, max_extra=3, thr=0.8):
    ids = input_ids[0].tolist() if getattr(input_ids, "ndim", 1) == 2 else list(input_ids)
    probe_ids = tokenizer.encode(probe_string, add_special_tokens=False)
    n = max(1, len(probe_ids))
    for i in range(len(ids)):
        for w in range(max(1, n - max_extra), min(n + max_extra, len(ids) - i) + 1):
            if _fuzzy(tokenizer.decode(ids[i:i + w], skip_special_tokens=True), probe_string, thr):
                return i, w
    return -1, 0


# =============================================================================
# Forward-pass hook utilities
# =============================================================================

class TapCache:
    def __init__(self, tap_modules):
        self.tap_modules = tap_modules
        self.hooks = []
        self.buffers = {k: [] for k in tap_modules}

    def __enter__(self):
        for k, m in self.tap_modules.items():
            self.hooks.append(
                m.register_forward_hook(
                    lambda mod, inp, out, key=k:
                        self.buffers[key].append(out.detach().to(torch.float32).cpu())
                )
            )
        return self

    def __exit__(self, *exc):
        for h in self.hooks:
            h.remove()


class BitFlipToggle:
    def __init__(self, model, flag):
        self.model = model
        self.flag = bool(flag)
        self.prev = None

    def __enter__(self):
        self.prev = bool(self.model.config.bit_flip)
        self.model.config.bit_flip = self.flag
        return self

    def __exit__(self, *exc):
        self.model.config.bit_flip = self.prev


def register_baseline_shift_tap(model):
    """For vanilla AutoModelForCausalLM: tap embed_tokens output as shift_tap
    (equivalent to pre-layer-0, no shift applied)."""
    model.model.shift_tap = torch.nn.Identity()

    def embed_hook(module, input, output):
        return model.model.shift_tap(output)

    model.model.embed_tokens.register_forward_hook(embed_hook)
    return model


def forward_pass_with_hooks(model, batch, hooks=("model.shift_tap",), bit_flip=False):
    def _resolve(root, dotted):
        obj = root
        for name in dotted.split('.'):
            obj = getattr(obj, name)
        return obj

    taps = {name: _resolve(model, name) for name in hooks}

    fwd_kwargs = dict(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    # Pass expert_labels only to models whose forward signature accepts it
    # (e.g., ISE, Fuse). Vanilla AutoModelForCausalLM does not.
    import inspect
    sig = inspect.signature(model.forward)
    if "expert_labels" in sig.parameters and "expert_labels" in batch:
        fwd_kwargs["expert_labels"] = batch["expert_labels"]

    with TapCache(taps) as tc, BitFlipToggle(model, bit_flip):
        _ = model(**fwd_kwargs)
    return {k.replace("model.", ""): tc.buffers[k][0].numpy() for k in tc.buffers}


# =============================================================================
# Representation collection
# =============================================================================

def collect_reps_for_story(
    model, loader, cases, tokenizer, device,
    max_cases=200, hook_name="model.shift_tap",
    data_label=1, inst_label=0, bit_flip=False,
):
    probe_list, normal_data_list, instruction_list = [], [], []

    for idx, batch in tqdm(enumerate(loader)):
        if idx >= max_cases:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        roles = batch["expert_labels"][0].cpu().numpy()

        probe_start, w = find_probe_span(
            tokenizer, batch["input_ids"], cases[idx]["probe"],
            max_extra=0, thr=0.99,
        )
        if probe_start < 0 or w <= 0:
            continue
        probe_idx = np.arange(probe_start, probe_start + w)

        out = forward_pass_with_hooks(model, batch, hooks=(hook_name,), bit_flip=bit_flip)
        key = hook_name.split(".")[-1]
        X = out[key][0] if out[key].ndim == 3 else out[key]

        probe_idx = probe_idx[probe_idx < X.shape[0]]
        if probe_idx.size == 0:
            continue

        data_pos = np.where(roles == data_label)[0]
        inst_pos = np.where(roles == inst_label)[0]
        probe_set = set(probe_idx.tolist())
        normal_pos = np.array([p for p in data_pos if p not in probe_set and p < X.shape[0]])
        inst_pos = inst_pos[inst_pos < X.shape[0]]

        if normal_pos.size == 0 or inst_pos.size == 0:
            continue

        probe_list.append(X[probe_idx])
        normal_data_list.append(X[normal_pos])
        instruction_list.append(X[inst_pos])

    if not probe_list:
        return None, None, None

    probe_arr = np.concatenate(probe_list, axis=0)
    normal_arr = np.concatenate(normal_data_list, axis=0)
    instr_arr = np.concatenate(instruction_list, axis=0)
    print(f"[collect_reps] raw counts -> probe={len(probe_arr)}, "
          f"normal_data={len(normal_arr)}, instruction={len(instr_arr)} "
          f"(from {len(probe_list)} cases)")
    return probe_arr, normal_arr, instr_arr


# =============================================================================
# Silhouette + t-SNE
# =============================================================================

def compute_probe_vs_instruction_silhouette(
    probe, instructions, metric="cosine", max_per_class=600, seed=0,
):
    """2-cluster silhouette on raw H-dim hidden states: {probe} vs {instruction}."""
    rng = np.random.default_rng(seed)

    def _sub(X, n):
        if len(X) <= n:
            return X
        return X[rng.choice(len(X), n, replace=False)]

    P = _sub(probe, max_per_class)
    I = _sub(instructions, max_per_class)
    if len(P) < 2 or len(I) < 2:
        return None

    X = np.concatenate([P, I], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(len(P), dtype=int),
                        np.ones(len(I), dtype=int)])

    overall = float(silhouette_score(X, y, metric=metric))
    samples = silhouette_samples(X, y, metric=metric)
    return {
        "overall": overall,
        "probe_mean": float(samples[y == 0].mean()),
        "instruction_mean": float(samples[y == 1].mean()),
        "n_probe": int(len(P)),
        "n_instruction": int(len(I)),
        "metric": metric,
    }


def plot_story_tsne(probe, normal_data, instructions, title="",
                    n_jobs=-1):

    rng = np.random.default_rng(0)
    n_probe = len(probe)
    n_normal = min(len(normal_data), n_probe * 2)
    n_instr = min(len(instructions), n_probe * 2)

    probe_sub  = probe[rng.choice(len(probe),        n_probe,  replace=False)]
    normal_sub = normal_data[rng.choice(len(normal_data), n_normal, replace=False)]
    instr_sub  = instructions[rng.choice(len(instructions), n_instr, replace=False)]

    X = np.concatenate([probe_sub, normal_sub, instr_sub], axis=0).astype(np.float32)
    n1, n2 = len(probe_sub), len(normal_sub)

    # ---- denser clusters ----
    perp = min(30, X.shape[0] - 1)        # ↑ from 30 (larger neighborhood = tighter clusters)
    tsne = TSNE(
        n_components=2,
        init="pca",
        random_state=0,
        perplexity=perp,
        learning_rate="auto",
        n_iter=300,
        n_jobs=n_jobs,
    )
    X_2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(X_2d[:n1, 0], X_2d[:n1, 1],
               alpha=0.9, s=40, marker="*", color="red",
               label="Probe tokens (instruction-like in data section)", zorder=5)
    ax.scatter(X_2d[n1:n1 + n2, 0], X_2d[n1:n1 + n2, 1],
               alpha=0.4, s=18, marker="^", color="tab:green",   # alpha+size ↑
               label="Normal data tokens")
    ax.scatter(X_2d[n1 + n2:, 0], X_2d[n1 + n2:, 1],
               alpha=0.4, s=18, marker="o", color="tab:blue",
               label="Instruction tokens")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1", fontsize=16)
    ax.set_ylabel("t-SNE dim 2", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=15)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

# =============================================================================
# Case building
# =============================================================================

def build_cases_from_ours_only(ours_json_path: Path):
    with open(ours_json_path, 'r') as f:
        ours = json.load(f)

    cases = []
    for ours_elem in ours:
        witness = ours_elem["data"]["witness"]
        try:
            from testing.sep.evaluation_main import diff_sentences
            diff = diff_sentences(ours_elem['data']["prompt_instructed"],
                                  ours_elem["data"]["prompt_clean"])
            probe = diff["removed"][0] if diff["removed"] else diff["added"][0]
        except Exception:
            probe = ours_elem["data"].get("probe_text", "")
        cases.append({
            "input": ours_elem["instructions"]["input_1"],
            "input_2": ours_elem["instructions"]["input_2"],
            "ours_output": ours_elem["output1_probe_in_data"],
            "witness": witness.lower(),
            "probe": probe,
            "type": ours_elem["data"]["info"]["type"],
        })
    return cases


# =============================================================================
# Dataset / DataLoader
# =============================================================================

class DebugDataset(SupervisedDataset):
    def __init__(self, data_dict):
        self.input_ids = data_dict["input_ids"]
        self.expert_labels = data_dict["expert_labels"]
        self.labels = data_dict["labels"]

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            expert_labels=self.expert_labels[i],
            labels=self.labels[i],
        )


@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, expert_labels, labels = tuple(
            [inst[k] for inst in instances]
            for k in ("input_ids", "expert_labels", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        expert_labels = torch.nn.utils.rnn.pad_sequence(
            expert_labels, batch_first=True, padding_value=2)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            expert_labels=expert_labels,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def preprocess(sources, targets, tokenizer, frontend_delimiters):
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized = _tokenize_fn(examples, tokenizer, frontend_delimiters=frontend_delimiters)
    input_ids = examples_tokenized["input_ids"]
    sources_lens = _tokenize_fn(sources, tokenizer,
                                frontend_delimiters=frontend_delimiters)["input_ids_lens"]

    labels = copy.deepcopy(input_ids)
    for label, src_len in zip(labels, sources_lens):
        label[:src_len] = IGNORE_INDEX

    encode = lambda s: tokenizer(s, add_special_tokens=False).input_ids
    delm = DELIMITERS[frontend_delimiters]
    num_labels = 4 if len(delm) == 4 else 3

    if num_labels == 3:
        inst_ids, data_ids, resp_ids = encode(delm[0]), encode(delm[1]), encode(delm[2])
    else:
        inst_ids, data_ids, resp_ids = encode(delm[1]), encode(delm[2]), encode(delm[3])

    expert_labels = [
        compute_expert_labels_from_input_ids(
            input_ids=x.unsqueeze(0),
            data_delm_ids=data_ids,
            response_delm_ids=resp_ids,
            inst_delm_ids=inst_ids,
            num_labels=num_labels,
        ).squeeze(0)
        for x in input_ids
    ]
    return dict(input_ids=input_ids, expert_labels=expert_labels, labels=labels)


def make_loader(tokenizer, frontend_delimiters,
                texts_A: List[str], texts_B: List[str], batch_size=1):
    data_dict = preprocess(texts_A, texts_B, tokenizer=tokenizer,
                           frontend_delimiters=frontend_delimiters)
    dataset = DebugDataset(data_dict)
    collator = DataCollatorForSupervisedDataset(tokenizer)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collator, num_workers=0, pin_memory=True,
    )


# =============================================================================
# Model loaders
# =============================================================================

def load_baseline_model(path: str):
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, attn_implementation="eager",
    )
    model = register_baseline_shift_tap(model)
    model.config.bit_flip = False
    model.eval()
    return model


def load_ours_model(path: str):
    config = LlamaDRIPConfig.from_pretrained(path)
    model = LlamaForCausalLMDRIP.from_pretrained(
        path, config=config, torch_dtype=torch.bfloat16,
        device_map="auto", low_cpu_mem_usage=True, attn_implementation="eager",
    )
    model.config.attn_implementation = "eager"
    model.eval()
    return model


def load_ise_model(path: str):
    config = LlamaISEConfig.from_pretrained(path)
    model = LlamaForCausalLMISE.from_pretrained(
        path, config=config, torch_dtype=torch.bfloat16,
        device_map="auto", low_cpu_mem_usage=True, attn_implementation="eager",
    )
    model.config.attn_implementation = "eager"
    model.eval()
    return model


def free_model(model):
    del model
    torch.cuda.empty_cache()
    gc.collect()


def report_silhouette(tag: str, probe, instructions, results: dict):
    res = compute_probe_vs_instruction_silhouette(probe, instructions)
    if res is None:
        print(f"[{tag}] silhouette: insufficient samples")
        return
    results[tag] = res
    print(
        f"[{tag}] silhouette(probe vs instruction, {res['metric']}): "
        f"overall={res['overall']:+.4f}  "
        f"probe_mean={res['probe_mean']:+.4f}  "
        f"instr_mean={res['instruction_mean']:+.4f}  "
        f"(n_probe={res['n_probe']}, n_instr={res['n_instruction']})"
    )


# =============================================================================
# Main
# =============================================================================

def main():
    ours_model_path       = "meta-llama/Llama-3.1-8B-Instruct-log-TextTextText-instfuse-alpaca-dpo-4roles"
    # ours_model_path = "meta-llama/Meta-Llama-3-8B-Instruct-TextTextText-3roles-air-sep-none"
    undefended_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    secalign_model_path   = "meta-llama/Meta-SecAlign-8B-merged"
    ise_model_path        = "meta-llama/Meta-Llama-3-8B-Instruct-TextTextText-ise-sep-none"

    ours_json = f"{ours_model_path}/predictions_on_sep.jsonl"
    out_dir = Path("viz_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cases = build_cases_from_ours_only(Path(ours_json))

    # ---- 3-role loader (undefended / SecAlign / ISE) ----
    tok3 = transformers.AutoTokenizer.from_pretrained(
        undefended_model_path, use_fast=True, use_auth_token=True)
    tok3.model_max_length = 512
    tok3.pad_token = tok3.eos_token
    delim3 = load_delimiters(os.path.basename(undefended_model_path), undefended_model_path)
    loader_3role = make_loader(
        tok3, delim3,
        [x["input"] for x in all_cases],
        [x["ours_output"] for x in all_cases],
        batch_size=1,
    )
    INST_3, DATA_3 = 0, 1   # 3-role: inst=0, data=1, response=2

    # ---- 4-role loader (ours) ----
    # tok4 = transformers.AutoTokenizer.from_pretrained(
    #     ours_model_path, use_fast=True, use_auth_token=True)
    # tok4.model_max_length = 512
    # tok4.pad_token = tok4.eos_token
    # delim4 = load_delimiters(os.path.basename(ours_model_path), ours_model_path)
    # loader_4role = make_loader(
    #     tok4, delim4,
    #     [x["input"] for x in all_cases],
    #     [x["ours_output"] for x in all_cases],
    #     batch_size=1,
    # )
    # INST_4, DATA_4 = 1, 2   # 4-role: system=0, inst=1, data=2, response=3

    sil_results = {}
    MAX_CASES = 200

    def run_one(tag, loader_fn, model_path, loader, tok, inst_lab, data_lab, bit_flip):
        print(f"=== Loading {tag} ===")
        m = loader_fn(model_path)
        device = next(m.parameters()).device
        probe, normal, instrs = collect_reps_for_story(
            model=m, loader=loader, cases=all_cases,
            tokenizer=tok, device=device,
            max_cases=MAX_CASES, hook_name="model.shift_tap",
            bit_flip=bit_flip, data_label=data_lab, inst_label=inst_lab,
        )
        free_model(m); print(f"{tag} freed from GPU.")
        report_silhouette(tag, probe, instrs, sil_results)
        plot_story_tsne(probe, normal, instrs, title="")

    run_one("Undefended",    load_baseline_model, undefended_model_path,
            loader_3role, tok3, INST_3, DATA_3, bit_flip=False)

    # run_one("Meta-SecAlign", load_baseline_model, secalign_model_path,
    #         loader_3role, tok3, INST_3, DATA_3, bit_flip=False)

    # run_one("Ours",          load_ours_model,     ours_model_path,
    #         loader_3role, tok3, INST_3, DATA_3, bit_flip=True)

    sil_path = out_dir / "silhouette.json"
    with open(sil_path, "w") as f:
        json.dump(sil_results, f, indent=2)
    print(f"Silhouette scores saved to: {sil_path}")
    print(f"All figures under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

    # [Undefended] silhouette(probe vs instruction, cosine): overall=+0.0491  probe_mean=+0.0445  instr_mean=+0.0537  (n_probe=2000, n_instr=2000)
    # [Meta-SecAlign] silhouette(probe vs instruction, cosine): overall=+0.0462  probe_mean=+0.0420  instr_mean=+0.0504  (n_probe=2000, n_instr=2000)
    #
    # [Ours]