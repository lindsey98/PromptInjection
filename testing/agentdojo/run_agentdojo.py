
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv

# ---- Project imports -------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---- agentdojo imports (mirror official benchmark.py) ----------------------
from agentdojo.agent_pipeline.agent_pipeline import (
    DEFENSES,
    AgentPipeline,
    PipelineConfig,
)
from agentdojo.attacks.attack_registry import ATTACKS, load_attack
from agentdojo.benchmark import (
    SuiteResults,
    benchmark_suite_with_injections,
    benchmark_suite_without_injections,
)
from agentdojo.logging import OutputLogger
from agentdojo.models import ModelsEnum
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.task_suite.task_suite import TaskSuite
from fuse_llm import build_fuse_pipeline
from agentdojo.benchmark import TaskResults
from agentdojo.functions_runtime import FunctionCall
import agentdojo.benchmark as _bench_module

_bench_module.FunctionCall = FunctionCall

TaskResults.model_rebuild()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ALL_SUITES = ["banking", "slack", "travel", "workspace"]
import os
os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"

# ============================================================================
# benchmark_suite — direct migration of upstream benchmark.py::benchmark_suite.
# Identical to upstream except `pipeline` is supplied directly (already built).
# ============================================================================
def benchmark_suite(
    suite: TaskSuite,
    pipeline: AgentPipeline,
    logdir: Path,
    force_rerun: bool,
    benchmark_version: str,
    user_tasks: tuple[str, ...] = (),
    injection_tasks: tuple[str, ...] = (),
    attack: Optional[str] = None,
    defense: Optional[str] = None,
) -> SuiteResults:
    if not load_dotenv(".env"):
        warnings.warn("No .env file found")

    print(f"Running benchmark for suite: '{suite.name}'")
    print(f"Using pipeline: '{pipeline.name}'")
    if attack is not None:
        print(f"Using attack: '{attack}'")
    if defense is not None:
        print(f"Using defense: '{defense}'")
    if len(user_tasks) > 0:
        print(f"Using user tasks: {', '.join(user_tasks)}")

    with OutputLogger(str(logdir)):
        if attack is None:
            results = benchmark_suite_without_injections(
                pipeline,
                suite,
                user_tasks=user_tasks if len(user_tasks) != 0 else None,
                logdir=logdir,
                force_rerun=force_rerun,
                benchmark_version=benchmark_version,
            )
        else:
            attacker_ = load_attack(attack, suite, pipeline)
            results = benchmark_suite_with_injections(
                pipeline,
                suite,
                attacker_,
                user_tasks=user_tasks if len(user_tasks) != 0 else None,
                injection_tasks=injection_tasks if len(injection_tasks) != 0 else None,
                logdir=logdir,
                force_rerun=force_rerun,
                benchmark_version=benchmark_version,
            )

    print(f"Finished benchmark for suite: '{suite.name}'")
    return results


# ============================================================================
# show_results — direct migration of upstream benchmark.py::show_results.
# ============================================================================
def show_results(
    suite_name: str, results: SuiteResults, show_security_results: bool
) -> None:
    utility_results = list(results["utility_results"].values())
    avg_utility = (
        sum(utility_results) / len(utility_results) if utility_results else 0.0
    )
    print(f"Results for suite {suite_name}")
    print(f"Average utility: {avg_utility * 100:.2f}%")

    if show_security_results:
        passed = sum(results["injection_tasks_utility_results"].values())
        total = len(results["injection_tasks_utility_results"])
        print(f"\nPassed injection tasks as user tasks: {passed}/{total}")

        security_results = list(results["security_results"].values())
        avg_security = (
            sum(security_results) / len(security_results) if security_results else 0.0
        )
        print(f"Average security: {avg_security * 100:.2f}%")


def save_summary(
    all_results: dict[str, SuiteResults],
    pipeline_name: str,
    mode: str,
    attack: Optional[str],
    defense: Optional[str],
    output_path: Path,
) -> None:
    summary: dict[str, dict] = {}
    for suite_name, results in all_results.items():
        util_vals = list(results["utility_results"].values())
        sec_vals = list(results["security_results"].values())
        summary[suite_name] = {
            "utility": sum(util_vals) / len(util_vals) if util_vals else None,
            "security": sum(sec_vals) / len(sec_vals) if sec_vals else None,
            "n_pairs": len(util_vals),
        }
    out = {
        "pipeline": pipeline_name,
        "mode": mode,
        "attack": attack,
        "defense": defense,
        "results": summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    logger.info("Summary saved to %s", output_path)


# ============================================================================
# Pipeline construction
# ============================================================================
def build_official_pipeline(args: argparse.Namespace) -> AgentPipeline:
    try:
        llm = ModelsEnum(args.model_name_or_path)
    except ValueError as e:
        valid = [m.value for m in ModelsEnum]
        raise SystemExit(
            f"--mode official requires --model_name_or_path to be a value in "
            f"agentdojo.models.ModelsEnum.\n"
            f"Got: {args.model_name_or_path!r}\n"
            f"Valid values: {valid}"
        ) from e

    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm=llm,
            model_id=args.model_name or args.model_name_or_path,  # ADD THIS
            defense=args.defense,
            system_message_name=args.system_message_name,
            system_message=args.system_message,
            tool_output_format=args.tool_output_format,
        )
    )
    pipeline.name = args.model_name or args.model_name_or_path
    return pipeline


def build_fuse_pipeline_from_args(args: argparse.Namespace) -> AgentPipeline:
    """Build our custom 4-role / expert-labels pipeline."""

    repeat = (args.defense == "repeat_user_prompt")
    if args.defense is not None and not repeat:
        warnings.warn(f"--defense={args.defense} not supported in mode=fuse; ignored")

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    if args.defense is not None:
        warnings.warn(
            "--defense is ignored in mode=fuse: the fuse pipeline does not "
            "implement agentdojo defenses. Drop --defense or switch to mode=official."
        )
    if args.tool_output_format is not None:
        warnings.warn(
            "--tool-output-format is ignored in mode=fuse."
        )
    if args.system_message_name is not None:
        warnings.warn(
            "--system-message-name is ignored in mode=fuse "
            "(only --system-message is honoured)."
        )

    display_name = args.model_name or f"local/{args.model_name_or_path}"
    pipeline = build_fuse_pipeline(
        model_path=args.model_name_or_path,
        customized_model_class=args.customized_model_class or None,
        model_name=display_name,
        system_message=args.system_message or "",
        torch_dtype=torch_dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        pass_expert_labels=not args.no_expert_labels,
        max_iters=args.max_iters,
    )
    pipeline.name = display_name
    return pipeline


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark on AgentDojo. Two modes: fuse (custom 4-role "
        "pipeline for our trained models) or official (upstream "
        "AgentPipeline.from_config, for GPT/Claude/Gemini or vLLM-served local).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- mode -------------------------------------------------------------
    p.add_argument(
        "--mode",
        choices=["fuse", "official"],
        required=True,
        help="fuse: use build_fuse_pipeline. official: mirror upstream "
        "agentdojo.scripts.benchmark exactly.",
    )

    # --- model identifier -------------------------------------------------
    p.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        nargs="+",
        required=True,
        help="In mode=fuse: HF path or local dir. "
        "In mode=official: a ModelsEnum value "
        "(e.g. 'gpt-4o-2024-05-13' or a vLLM-served local name).",
    )
    p.add_argument(
        "--model-name", default=None,
        help="Display name; defaults to model_name_or_path.",
    )

    # --- mode=fuse specific ----------------------------------------------
    p.add_argument(
        "--customized_model_class",
        type=str,
        default="",
        help="(mode=fuse only) Registry key, e.g. QwenForCausalLMFuse.",
    )
    p.add_argument("--no-expert-labels", action="store_true",
                   help="(mode=fuse only) Disable expert_labels passing.")
    p.add_argument(
        "--dtype", default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="(mode=fuse only) HF model dtype.",
    )
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="(mode=fuse only) Generation token budget per step.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="(mode=fuse only) Sampling temperature.")
    p.add_argument("--max-iters", type=int, default=15,
                   help="(mode=fuse only) Max tool-call rounds.")

    # --- benchmark mirror of upstream CLI --------------------------------
    p.add_argument("--benchmark-version", default="v1.2.1")
    p.add_argument(
        "--suites", "-s", nargs="+", default=ALL_SUITES,
        choices=ALL_SUITES + ["all"],
    )
    p.add_argument(
        "--attack", default=None,
        choices=list(ATTACKS) + [None],  # type: ignore[arg-type]
    )
    p.add_argument(
        "--defense", default=None,
        choices=list(DEFENSES) + [None],  # type: ignore[arg-type]
        help="(mode=official only) Applied via PipelineConfig.",
    )
    p.add_argument(
        "--user-task", "-ut",
        dest="user_tasks", action="append", default=[],
        help="Repeatable. Empty -> all.",
    )
    p.add_argument(
        "--injection-task", "-it",
        dest="injection_tasks", action="append", default=[],
        help="Repeatable. Empty -> all.",
    )
    p.add_argument("--system-message-name", default=None,
                   help="(mode=official only)")
    p.add_argument("--system-message", default=None)
    p.add_argument(
        "--tool-output-format", default=None,
        choices=["yaml", "json", None],
        help="(mode=official only)",
    )
    p.add_argument("--logdir", type=Path, default=Path("./agentdojo_runs"))
    p.add_argument("--force-rerun", "-f", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.model_name_or_path = args.model_name_or_path[0]

    if args.user_tasks and len(args.suites) != 1:
        raise ValueError(
            "A user task can be specified only when one suite is being executed"
        )

    suites = ALL_SUITES if "all" in args.suites else args.suites

    # --- Build pipeline ---------------------------------------------------
    if args.mode == "official":
        logger.info("Mode=official | building AgentPipeline.from_config")
        pipeline = build_official_pipeline(args)
    else:
        logger.info("Mode=fuse | building custom fuse pipeline")
        pipeline = build_fuse_pipeline_from_args(args)
    logger.info("Pipeline ready: %s", pipeline.name)

    # --- Run --------------------------------------------------------------
    all_results: dict[str, SuiteResults] = {}
    for suite_name in suites:
        suite = get_suite(args.benchmark_version, suite_name)
        all_results[suite_name] = benchmark_suite(
            suite=suite,
            pipeline=pipeline,
            logdir=args.logdir,
            force_rerun=args.force_rerun,
            benchmark_version=args.benchmark_version,
            user_tasks=tuple(args.user_tasks),
            injection_tasks=tuple(args.injection_tasks),
            attack=args.attack,
            defense=args.defense,
        )

    # --- Display ---------------------------------------------------------
    for suite_name, result in all_results.items():
        show_results(
            suite_name, result, show_security_results=(args.attack is not None)
        )

    if len(suites) > 1:
        all_util = [
            v for r in all_results.values() for v in r["utility_results"].values()
        ]
        all_sec = [
            v for r in all_results.values() for v in r["security_results"].values()
        ]
        print(f"\n{'=' * 55}")
        print(f"OVERALL  ({len(suites)} suites)")
        print(f"  Utility  : {sum(all_util) / len(all_util) * 100:.2f}%")
        if args.attack:
            print(f"  Security : {sum(all_sec) / len(all_sec) * 100:.2f}%")
        print(f"{'=' * 55}")

    # --- Save summary ----------------------------------------------------
    summary_path = (
        args.logdir
        / f"{pipeline.name.replace('/', '_')}"
        f"_{args.mode}"
        f"_{args.attack or 'no-attack'}"
        f"_{args.defense or 'no-defense'}_summary.json"
    )
    save_summary(
        all_results,
        pipeline.name,
        args.mode,
        args.attack,
        args.defense,
        summary_path,
    )


if __name__ == "__main__":
    main()