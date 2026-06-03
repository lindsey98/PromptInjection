"""Shared command-line arguments for the evaluation entry points.

Every test_*.py script needs the same trio of model-identification arguments.
Centralizing them here keeps the flags (names, types, defaults) consistent
across scripts while letting each script decide whether the model path is
required.
"""

import argparse


def add_model_args(parser: argparse.ArgumentParser, *, required: bool = False) -> argparse.ArgumentParser:
    """Add the model-identification arguments shared by all evaluation scripts.

    Args:
        parser: the parser to extend.
        required: whether ``-m/--model_name_or_path`` must be supplied.

    Returns:
        The same parser, for chaining.
    """
    parser.add_argument(
        "-m", "--model_name_or_path", type=str, nargs="+", required=required,
        help="Model path(s) or HF id(s) to evaluate.",
    )
    parser.add_argument(
        "--base_model_path", type=str, default=None,
        help="Explicit base model path; required when adapter path "
             "does not encode the base path via the usual suffix convention.",
    )
    parser.add_argument(
        "--customized_model_class", type=str, default="",
        help="Custom model-class key from the REGISTRY (empty = stock "
             "AutoModelForCausalLM, or auto-detected from the path).",
    )
    return parser
