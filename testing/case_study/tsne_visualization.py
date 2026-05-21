from pathlib import Path
import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import torch
import matplotlib.pyplot as plt
import textwrap
# ---- Import your project modules ----
import transformers
from testing.test import load_delimiters
from data_generation.sft_data_loader import preprocess, DataCollatorForSupervisedDataset, SupervisedDataset
from torch.utils.data import DataLoader
from modeling import LlamaForCausalLMISE, LlamaISEConfig, LlamaForCausalLMDRIP, LlamaDRIPConfig, MistralForCausalLMDRIP, MistralDRIPConfig
import numpy as np, difflib
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
plt.rcParams.update({
    "font.family": "arial",
    "font.size": 8,
    "axes.labelsize": 24,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "legend.title_fontsize": 16,  # 👈 legend 标题字号
    "axes.linewidth": 0.8,
    "lines.linewidth": 2.5,
})

# Number of tokens to sample for visualization (to keep it readable)
SAMPLE_SIZE = 5000
# Metric sample size (can be larger)
METRIC_SIZE = 20000


def compute_density_metrics(embeddings, name):
    # Randomly sample to speed up NN search
    indices = np.random.choice(embeddings.shape[0], min(METRIC_SIZE, embeddings.shape[0]), replace=False)
    subset = embeddings[indices]

    print(f"[{name}] Computing Nearest Neighbor distances for {len(subset)} tokens...")
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(subset)
    distances, _ = nbrs.kneighbors(subset)

    return distances[:, 1]



if __name__ == '__main__':

    model_path = "meta-llama/Meta-Llama-3-8B-Instruct-TextTextText-instfuse-sep-none-newdata-dpo"  # run

    config = LlamaDRIPConfig.from_pretrained(model_path)
    llama_model = LlamaForCausalLMDRIP.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # required for attention tensors
    )

    llama_model.eval()
    llama_embeddings = llama_model.get_input_embeddings().weight.detach().cpu().float().numpy()
    del llama_model


    model_path = "mistralai/Mistral-7B-Instruct-v0.3-TextTextTextMistral-instfuse-sep-none-newdata-dpo"  # run

    config = MistralDRIPConfig.from_pretrained(model_path)
    mistral_model = MistralForCausalLMDRIP.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # required for attention tensors
    )

    mistral_model.eval()
    mistral_embeddings = mistral_model.get_input_embeddings().weight.detach().cpu().float().numpy()
    del mistral_model


    results = {}
    indices    = np.random.choice(llama_embeddings.shape[0], min(SAMPLE_SIZE, llama_embeddings.shape[0]), replace=False)
    subset_vis = llama_embeddings[indices]
    results["Llama-8B"] = subset_vis

    indices    = np.random.choice(mistral_embeddings.shape[0], min(SAMPLE_SIZE, mistral_embeddings.shape[0]), replace=False)
    subset_vis = mistral_embeddings[indices]
    results["Mistral-7B"] = subset_vis

    # Using PCA instead of t-SNE here because t-SNE normalizes density, making sparse/dense clusters look the same size.
    # PCA preserves global variance better for this specific comparison.
    for name, subset in results.items():
        plt.figure(figsize=(12, 5), dpi=150)
        print(f"[{name}] Running PCA...")
        pca  = PCA(n_components=2)
        proj = pca.fit_transform(subset)

        plt.scatter(proj[:, 0], proj[:, 1], s=1, alpha=0.5, label=name)

        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"vocab_density_comparison_{name}.pdf")
        plt.show()

    # Llama (红色): 曲线应该更靠左，且峰值更高。这意味着 Llama 的大部分 Token 距离邻居非常近（Distance < 0.5）。这证明了它的空间极其拥挤（Dense）。
    # Mistral (蓝色): 曲线应该更靠右，分布更平缓。这意味着 Mistral 的 Token 之间保留了较大的“安全距离”。