"""Lightweight model-class registry helpers.

Kept free of heavy ML imports (no torch/transformers) so the path-to-class
mapping can be reused and unit tested on its own. ``testing/test.py`` builds the
concrete REGISTRY of config/model classes and asserts that its keys stay in sync
with :data:`KNOWN_MODEL_CLASSES` here.
"""

# The set of valid ``--customized_model_class`` keys, mirrored by the REGISTRY
# in testing/test.py.
KNOWN_MODEL_CLASSES = frozenset(
    {
        "LlamaForCausalLMDRIP",
        "LlamaForCausalLMISE",
        "LlamaForCausalLMAIR",
        "LlamaForCausalLMPFT",
        "LlamaForCausalLMEmbeddingShift",
        "LlamaForCausalLMNoFuse",
        "LlamaForCausalLMConcatFuse",
        "MistralForCausalLMDRIP",
        "MistralForCausalLMISE",
        "MistralForCausalLMAIR",
        "MistralForCausalLMPFT",
        "Qwen3MoeForCausalLMDRIP",
        "Qwen3ForCausalLMDRIP",
    }
)


def detect_model_class_from_path(path: str) -> str:
    """Best-effort mapping from a checkpoint path to a model-class key.

    Mirrors the substring logic in ``scripts/evaluation/**/*.sh`` so the Python
    entry points can pick the right custom class when ``--customized_model_class``
    is not supplied. The candidate key is always validated against
    :data:`KNOWN_MODEL_CLASSES`, so unknown architecture/method combinations
    return ``""`` and the caller falls back to the stock HF class.
    """
    p = (path or "").lower()

    # Architecture prefix, matching the registry key naming.
    if "mistral" in p:
        prefix = "MistralForCausalLM"
    elif "qwen" in p:
        prefix = "Qwen3MoeForCausalLM" if "moe" in p else "Qwen3ForCausalLM"
    else:
        prefix = "LlamaForCausalLM"

    # Method keyword -> class suffix, most specific first.
    if "instfuse" in p and "nofusion" in p:
        suffix = "NoFuse"
    elif "instfuse" in p and "concatfusion" in p:
        suffix = "ConcatFuse"
    elif "instfuse" in p and "embeddingshift" in p:
        suffix = "EmbeddingShift"
    elif "instfuse" in p or "drip" in p:
        suffix = "DRIP"
    elif "ise" in p:
        suffix = "ISE"
    elif "air" in p:
        suffix = "AIR"
    elif "possep" in p:
        suffix = "PFT"
    else:
        return ""

    key = prefix + suffix
    return key if key in KNOWN_MODEL_CLASSES else ""
