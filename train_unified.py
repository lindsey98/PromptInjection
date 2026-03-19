import os
import sys
import argparse
from typing import Optional, List, Dict, Any
import transformers
from transformers.utils import logging
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import DPOTrainer

from training.trainer import DPOTrainerOurs, SFTTrainerOurs, SFTTrainerISE, DPOArgsInstFuse, \
    ModelArguments, DataArguments, TrainingArguments, AttackArguments, DPOArgsSecAlign
from data_generation.dpo import make_dpo_data_module
from data_generation.struq import make_supervised_data_module, \
    make_supervised_data_module_orig, \
    smart_tokenizer_and_embedding_resize

from modeling.llama_instfuse import (
    LlamaFuseConfig,
    LlamaForCausalLMFuse,
    LlamaForCausalLMNoFuse,
    LlamaForCausalLMConcatFuse,
    LlamaForCausalLMEmbeddingShift,
)

from modeling import (
    MistralFuseConfig,
    MistralForCausalLMFuse,
    MistralMoEConfig,
    MistralForCausalLMMoE,
    MistralForCausalLMMoEV2,
)

from modeling.llama_instsep import (
    LlamaMoEConfig,
    LlamaForCausalLMMoE,
    LlamaForCausalLMMoEV2,
)

from modeling.qwen_instfuse import (
    Qwen3FuseConfig,
    Qwen3ForCausalLMFuse,
    Qwen3ForCausalLMNoFuse,
    Qwen3ForCausalLMConcatFuse,
    Qwen3ForCausalLMEmbeddingShift,
)

from modeling.qwen_instsep import (
    Qwen3MoEConfig,
    Qwen3ForCausalLMMoE,
    Qwen3ForCausalLMMoEV2,
)

from config import DEFAULT_TOKENS, SPECIAL_DELM_TOKENS

logger = logging.get_logger(__name__)
os.environ.setdefault("WANDB_WATCH", "all")


def pick_llama_model(arch: str):
    if arch == "fuse":
        return LlamaFuseConfig, LlamaForCausalLMFuse
    if arch == "nofuse":
        return LlamaFuseConfig, LlamaForCausalLMNoFuse
    if arch == "concatfuse":
        return LlamaFuseConfig, LlamaForCausalLMConcatFuse
    if arch == "embeddingshift":
        return LlamaFuseConfig, LlamaForCausalLMEmbeddingShift
    if arch == "ise":
        return LlamaMoEConfig, LlamaForCausalLMMoE
    if arch == "possep":
        return LlamaMoEConfig, LlamaForCausalLMMoEV2
    raise ValueError(f"Unsupported LLaMA arch: {arch}")


def pick_mistral_model(arch: str):
    if arch == "fuse":
        return MistralFuseConfig, MistralForCausalLMFuse
    if arch == "ise":
        return MistralMoEConfig, MistralForCausalLMMoE
    if arch == "possep":
        return MistralMoEConfig, MistralForCausalLMMoEV2
    raise ValueError(f"Unsupported Mistral arch: {arch}")


def pick_qwen_model(arch: str):
    if arch == "fuse":
        return Qwen3FuseConfig, Qwen3ForCausalLMFuse
    if arch == "nofuse":
        return Qwen3FuseConfig, Qwen3ForCausalLMNoFuse
    if arch == "concatfuse":
        return Qwen3FuseConfig, Qwen3ForCausalLMConcatFuse
    if arch == "embeddingshift":
        return Qwen3FuseConfig, Qwen3ForCausalLMEmbeddingShift
    if arch == "ise":
        return Qwen3MoEConfig, Qwen3ForCausalLMMoE
    if arch == "possep":
        return Qwen3MoEConfig, Qwen3ForCausalLMMoEV2
    raise ValueError(f"Unsupported Qwen3 arch: {arch}")


def build_lora_config(objective: str, arch: str) -> LoraConfig:
    if objective in ("dpo", "sft") and arch in ("fuse", "nofuse", "concatfuse", "embeddingshift"):
        modules = ["embed_tokens", "lm_head", "deinstruction_shift"]
    elif arch in ("ise",):
        modules = ["lm_head", "embed_tokens", "input_shifts"]
    else:
        modules = ["lm_head", "embed_tokens"]

    return LoraConfig(
        r=16, lora_alpha=8, lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        exclude_modules=modules,
        modules_to_save=modules
    )


def main(argv: Optional[List[str]] = None):
    # Stage 1: route by objective/arch
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--objective", choices=["dpo", "sft", "secalign_dpo", "struq_sft"], required=True)
    parser.add_argument("--model-family", choices=["llama", "mistral", "qwen"], required=False, default="base")
    parser.add_argument("--arch", required=True,
                        choices=["base", "fuse", "nofuse", "concatfuse", "embeddingshift", "ise", "possep"])
    known, remaining = parser.parse_known_args(argv)

    # Stage 2: parse HF dataclasses per flow
    if known.objective == "dpo":
        hf_parser = transformers.HfArgumentParser((ModelArguments, DataArguments, DPOArgsInstFuse, AttackArguments))
        model_args, data_args, training_args, attack_args = hf_parser.parse_args_into_dataclasses(remaining)
    elif known.objective == "secalign_dpo":
        hf_parser = transformers.HfArgumentParser((ModelArguments, DPOArgsSecAlign, DataArguments, AttackArguments))
        model_args, training_args, data_args, attack_args = hf_parser.parse_args_into_dataclasses(remaining)
    else:  # SFT
        hf_parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, AttackArguments))
        model_args, data_args, training_args, attack_args = hf_parser.parse_args_into_dataclasses(remaining)

    data_args.attack = getattr(attack_args, "attack", None)
    logger.info(f"Objective={known.objective} | Family={getattr(known, 'model_family', 'base')} | Arch={known.arch}")
    print("\n\n" + training_args.output_dir + "\n\n")

    # ------------------
    # Create tokenizer
    # ------------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=getattr(training_args, "cache_dir", None),
        model_max_length=getattr(training_args, "model_max_length", 512),
        padding_side=getattr(model_args, "padding_side", "right"),
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # ------------------
    # Create model
    # ------------------
    if known.arch == "base" or known.objective in ("secalign_dpo", "struq_sft"):
        config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=getattr(training_args, "cache_dir", None),
            config=config,
        )
        special_tokens_dict = dict()
        special_tokens_dict["pad_token"] = DEFAULT_TOKENS.get('pad_token', tokenizer.pad_token)
        special_tokens_dict["eos_token"] = DEFAULT_TOKENS.get('eos_token', tokenizer.eos_token)
        special_tokens_dict["bos_token"] = DEFAULT_TOKENS.get('bos_token', tokenizer.bos_token)
        special_tokens_dict["unk_token"] = DEFAULT_TOKENS.get('unk_token', tokenizer.unk_token)
        special_tokens_dict["additional_special_tokens"] = SPECIAL_DELM_TOKENS
        smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict,
                                             tokenizer=tokenizer,
                                             model=model)
    else:
        if known.model_family == "llama":
            Cfg, Model = pick_llama_model(known.arch)
        elif known.model_family == "mistral":
            Cfg, Model = pick_mistral_model(known.arch)
        elif known.model_family == "qwen":
            Cfg, Model = pick_qwen_model(known.arch)
        else:
            raise ValueError(f"Unsupported model-family: {known.model_family}")

        config = Cfg.from_pretrained(model_args.model_name_or_path)
        model = Model.from_pretrained(
            model_args.model_name_or_path,
            ignore_mismatched_sizes=True,
            config=config,
        )

    # Optional window attribute
    if hasattr(model_args, "window_size") and getattr(model_args, "window_size", 0) > 0 and hasattr(model.config, "window"):
        model.config.window = model_args.window_size

    # ------------------
    # LoRA
    # ------------------
    lora_config = build_lora_config(known.objective, known.arch)
    model = get_peft_model(model, lora_config)

    # ------------------
    # Data module
    # ------------------
    frontend_delimiters = (data_args.attack or "").split("_")[0] if getattr(data_args, "attack", None) else ""
    if known.objective == "dpo":
        data_module = make_dpo_data_module(tokenizer=tokenizer,
                                           data_args=data_args,
                                           frontend_delimiters=frontend_delimiters)
    elif known.objective == "secalign_dpo":
        train_dataset = load_dataset('json', data_files=data_args.data_path_list, split="train")
        data_module = {"train_dataset": train_dataset, "eval_dataset": None}
    elif known.objective == "sft":
        data_module = make_supervised_data_module(tokenizer=tokenizer,
                                                  data_args=data_args,
                                                  frontend_delimiters=frontend_delimiters,
                                                  downsample=getattr(training_args, "downsample", False))
        if not getattr(training_args, "downsample", True) and getattr(training_args, "lr_scale", True):
            training_args.learning_rate /= data_module["train_dataset"].data_copy_count
    else:  # struq_sft
        data_module = make_supervised_data_module_orig(tokenizer=tokenizer,
                                                       data_args=data_args,
                                                       frontend_delimiters=frontend_delimiters,
                                                       downsample=getattr(training_args, "downsample", False))
        if not getattr(training_args, "downsample", True) and getattr(training_args, "lr_scale", True):
            training_args.learning_rate /= data_module["train_dataset"].data_copy_count

    # ------------------
    # Trainer selection
    # ------------------
    if known.objective == "dpo":
        trainer = DPOTrainerOurs(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            **data_module
        )
    elif known.objective == "secalign_dpo":
        trainer = DPOTrainer(
            model,
            args=training_args,
            processing_class=tokenizer,
            **data_module,
        )
    else:  # SFT objective
        if known.arch in ("ise",):
            trainer_class = SFTTrainerISE
        elif known.arch in ("fuse", "nofuse", "concatfuse", "embeddingshift"):
            trainer_class = SFTTrainerOurs
        else:
            trainer_class = transformers.Trainer
        trainer = trainer_class(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            **data_module
        )

    # ------------------
    # Train & save
    # ------------------
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()
    trainer.train()

    # ------------------------------------------------------------------ #
    # Save
    # merge_and_unload() folds LoRA deltas back into the base weights so
    # the saved checkpoint is a standalone model — no adapter needed at
    # inference time. This also correctly handles modules_to_save weights.
    # ------------------------------------------------------------------ #
    print("Merging LoRA weights into base model...")
    merged_model = trainer.model.merge_and_unload()

    merged_model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    merged_model.config.save_pretrained(training_args.output_dir)
    print(f"Saved merged model to {training_args.output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/sda/hf_cache")
    main()