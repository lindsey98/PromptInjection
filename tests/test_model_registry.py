"""Unit tests for the path -> model-class detection logic."""

import pytest

from testing.model_registry import KNOWN_MODEL_CLASSES, detect_model_class_from_path


@pytest.mark.parametrize(
    "path,expected",
    [
        ("meta-llama/Meta-Llama-3-8B-Instruct-TextTextText-sep-drip", "LlamaForCausalLMDRIP"),
        ("out/llama-instfuse-nofusion", "LlamaForCausalLMNoFuse"),
        ("out/llama-instfuse-concatfusion", "LlamaForCausalLMConcatFuse"),
        ("out/llama-instfuse-embeddingshift", "LlamaForCausalLMEmbeddingShift"),
        ("out/llama-instfuse", "LlamaForCausalLMDRIP"),
        ("out/llama-ise", "LlamaForCausalLMISE"),
        ("out/llama-air-run", "LlamaForCausalLMAIR"),
        ("out/llama-possep", "LlamaForCausalLMPFT"),
        ("mistralai/Mistral-7B-Instruct-v0.3-drip", "MistralForCausalLMDRIP"),
        ("out/mistral-ise", "MistralForCausalLMISE"),
    ],
)
def test_detect_known_classes(path, expected):
    assert detect_model_class_from_path(path) == expected


@pytest.mark.parametrize(
    "path",
    [
        "meta-llama/Meta-Llama-3-8B-Instruct",  # base model, no method keyword
        "out/secalign-llama",  # baseline, no method keyword
        "out/mistral-instfuse-nofusion",  # unsupported combo -> guarded to ""
        "",
        None,
    ],
)
def test_detect_unknown_returns_empty(path):
    assert detect_model_class_from_path(path) == ""


def test_detector_only_returns_known_classes():
    samples = [
        "llama-drip", "mistral-ise", "llama-air", "llama-possep",
        "llama-instfuse-nofusion", "llama-instfuse-concatfusion",
        "llama-instfuse-embeddingshift", "random-base-model",
    ]
    for p in samples:
        key = detect_model_class_from_path(p)
        assert key == "" or key in KNOWN_MODEL_CLASSES
