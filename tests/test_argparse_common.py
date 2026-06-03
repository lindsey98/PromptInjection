"""Unit tests for the shared model-argument helper."""

import argparse

import pytest

from testing.argparse_common import add_model_args


def test_optional_defaults():
    p = argparse.ArgumentParser()
    add_model_args(p, required=False)
    ns = p.parse_args([])
    assert ns.model_name_or_path is None
    assert ns.base_model_path is None
    assert ns.customized_model_class == ""


def test_model_path_is_nargs_plus():
    p = argparse.ArgumentParser()
    add_model_args(p, required=False)
    ns = p.parse_args(["-m", "a", "b", "--customized_model_class", "LlamaForCausalLMDRIP"])
    assert ns.model_name_or_path == ["a", "b"]
    assert ns.customized_model_class == "LlamaForCausalLMDRIP"


def test_required_flag_is_enforced():
    p = argparse.ArgumentParser()
    add_model_args(p, required=True)
    with pytest.raises(SystemExit):
        p.parse_args([])
