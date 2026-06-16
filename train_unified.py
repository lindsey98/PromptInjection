import os
import argparse
from typing import Optional, List

import torch
import transformers
from transformers import BitsAndBytesConfig
from transformers.utils import logging
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import DPOTrainer

from config import DELIMITERS, DEFAULT_TOKENS, SPECIAL_DELM_TOKENS
from training.trainer import (
    DPOTrainerOurs, SFTTrainerOurs, SFTTrainerISE,
    DPOTrainerAIR,
    DPOArgsDRIP, DPOArgsSecAlign,
    ModelArguments, DataArguments, TrainingArguments, AttackArguments,
)
from data_generation.dpo_data_loader import make_dpo_data_module
from data_generation.sft_data_loader import (
    make_supervised_data_module,
    make_supervised_data_module_orig,
    smart_tokenizer_and_embedding_resize,
)
from modeling import (
    MistralDRIPConfig, MistralForCausalLMDRIP,
    MistralISEConfig, MistralForCausalLMISE,
    MistralAIRConfig, MistralForCausalLMAIR,
    MistralForCausalLMPFT,
    LlamaISEConfig, LlamaForCausalLMISE,
    LlamaAIRConfig, LlamaForCausalLMAIR,
    LlamaForCausalLMPFT,
    LlamaDRIPConfig, LlamaForCausalLMDRIP,
    LlamaForCausalLMNoFuse, LlamaForCausalLMConcatFuse,
    LlamaForCausalLMEmbeddingShift,
    set_delimiter_ids_in_config,
)


logger = logging.get_logger(__name__)


# =============================================================================
# Model registries
# =============================================================================

LLAMA_REGISTRY = {
    "fuse":           (LlamaDRIPConfig, LlamaForCausalLMDRIP),
    "nofuse":         (LlamaDRIPConfig, LlamaForCausalLMNoFuse),
    "concatfuse":     (LlamaDRIPConfig, LlamaForCausalLMConcatFuse),
    "embeddingshift": (LlamaDRIPConfig, LlamaForCausalLMEmbeddingShift),
    "ise":            (LlamaISEConfig,  LlamaForCausalLMISE),
    "air":            (LlamaAIRConfig,  LlamaForCausalLMAIR),
    "possep":         (LlamaISEConfig,  LlamaForCausalLMPFT),
}

MISTRAL_REGISTRY = {
    "fuse":   (MistralDRIPConfig, MistralForCausalLMDRIP),
    "ise":    (MistralISEConfig,  MistralForCausalLMISE),
    "air":    (MistralAIRConfig,  MistralForCausalLMAIR),
    "possep": (MistralISEConfig,  MistralForCausalLMPFT),
}


FAMILY_REGISTRY = {
    "llama":      LLAMA_REGISTRY,
    "mistral":    MISTRAL_REGISTRY,
}


def pick_model(family: str, arch: str):
    reg = FAMILY_REGISTRY.get(family)
    if reg is None:
        raise ValueError(f"Unsupported model-family: {family}")
    if arch not in reg:
        raise ValueError(f"Unsupported {family} arch: {arch}")
    return reg[arch]


# =============================================================================
# LoRA config
# =============================================================================

_FUSE_LIKE = {"fuse", "nofuse", "concatfuse", "embeddingshift"}
_ATTN_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]


def build_lora_config(
        objective: str,
        arch: str,
        model_family: str,
) -> LoraConfig:

    if arch == "air":
        modules_to_save = ["intermediate_shifts"]
        if objective == "dpo":
            modules_to_save += ["lm_head", "embed_tokens"]
        return LoraConfig(
            r=32,
            lora_alpha=8,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules= ["q_proj", "v_proj"],
            modules_to_save=modules_to_save,
        )

    if arch == "ise":
        modules_to_save = ["input_shifts", "embed_tokens"]
        target_modules = "all-linear"
        return LoraConfig(
            r=32,
            lora_alpha=8,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            modules_to_save=modules_to_save,
        )


    # ── Fuse-like (DRIP and variants) ───────────────────────────────────
    if objective in ("dpo", "sft") and arch in _FUSE_LIKE:
        if is_moe:
            modules_to_save = ["deinstruction_shift"]
            target_modules = _ATTN_MODULES
        else:
            # Dense: full LoRA + save embed/lm_head/deinstruction_shift
            modules_to_save = ["embed_tokens", "lm_head", "deinstruction_shift"]
            target_modules = "all-linear"

        return LoraConfig(
            r=16,
            lora_alpha=8,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            modules_to_save=modules_to_save,
        )

    # ── Default (vanilla SFT/DPO without custom architecture) ───────────
    modules_to_save = ["lm_head", "embed_tokens"] if not is_moe else None
    target_modules = _ATTN_MODULES if is_moe else "all-linear"

    return LoraConfig(
        r=32,
        lora_alpha=8,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )


# =============================================================================
# AIR: shifts-only training (no LoRA)
# =============================================================================

def freeze_base_train_shifts(model, target_param_name: str = "intermediate_shifts"):
    """Freeze entire base model except the target shift layer.
    Used for AIR: only nn.Embedding `intermediate_shifts` is trained."""
    trainable, frozen = 0, 0
    for name, p in model.named_parameters():
        if target_param_name in name:
            p.requires_grad = True
            trainable += p.numel()
        else:
            p.requires_grad = False
            frozen += p.numel()

    total = trainable + frozen
    print(f"=== Trainable params ({target_param_name} only) ===")
    print(f"trainable: {trainable:,} ({100 * trainable / total:.4f}%)")
    print(f"frozen:    {frozen:,}")
    print(f"total:     {total:,}")

    if trainable == 0:
        raise RuntimeError(
            f"No parameter contains '{target_param_name}'. "
            f"Check param names with `for n, _ in model.named_parameters(): print(n)`."
        )
    return model


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args(argv):
    """Two-stage parse: route by objective/arch, then parse HF dataclasses."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--objective",
                   choices=["dpo", "sft", "secalign_dpo", "struq_sft"],
                   required=True)
    p.add_argument("--model-family",
                   choices=["llama", "mistral", ],
                   default="base")
    p.add_argument("--arch", required=True,
                   choices=["base", "fuse", "nofuse", "concatfuse",
                            "embeddingshift", "ise", "possep", "air"])
    p.add_argument("--use_qlora", action="store_true",
                   help="Use 4-bit QLoRA quantization (saves ~70%% VRAM, "
                        "incompatible with FSDP, use DDP instead)")
    p.add_argument("--qlora_bits", type=int, default=4, choices=[4, 8],
                   help="Quantization bits (4 or 8). Default 4.")
    known, remaining = p.parse_known_args(argv)

    if known.objective in ("dpo"):
        cls = (ModelArguments, DataArguments, DPOArgsDRIP, AttackArguments)
        order = ("model", "data", "training", "attack")
    elif known.objective == "secalign_dpo":
        cls = (ModelArguments, DataArguments, DPOArgsSecAlign, AttackArguments)
        order = ("model", "training", "data", "attack")  # SecAlign uses (model, training, data) order
    else:  # sft, struq_sft
        cls = (ModelArguments, DataArguments, TrainingArguments, AttackArguments)
        order = ("model", "data", "training", "attack")

    parsed = transformers.HfArgumentParser(cls).parse_args_into_dataclasses(remaining)
    # SecAlign branch returns dataclasses in (model, training, data, attack) order
    # whereas other branches use (model, data, training, attack). Normalize.
    if known.objective == "secalign_dpo":
        model_args, training_args, data_args, attack_args = parsed
    else:
        model_args, data_args, training_args, attack_args = parsed
    return known, model_args, data_args, training_args, attack_args


# =============================================================================
# Model construction
# =============================================================================

def build_bnb_config(bits: int = 4) -> BitsAndBytesConfig:
    """NF4 / int8 quantization config for QLoRA."""
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
            llm_int8_skip_modules=[
                "lm_head",
                "embed_tokens",
                "deinstruction_shift",
                "intermediate_shifts",
                "input_shifts",
            ],
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def _common_load_kwargs(training_args, bnb_config=None):
    kwargs = dict(
        cache_dir=getattr(training_args, "cache_dir", None),
        low_cpu_mem_usage=True,        # init on meta device, save ~16GB CPU RAM/rank
        torch_dtype=torch.bfloat16,    # avoid fp32 intermediate copy
    )
    if bnb_config is not None:
        kwargs["quantization_config"] = bnb_config
        # QLoRA requires explicit device_map; FSDP is not supported on this path
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        kwargs["device_map"] = {"": local_rank}
    return kwargs


def build_base_model(model_args, training_args, tokenizer, bnb_config=None):
    """Vanilla AutoModelForCausalLM + special-token resize.
    Used for arch='base' or objectives 'secalign_dpo' / 'struq_sft'."""
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, config=config,
        trust_remote_code=True,
        **_common_load_kwargs(training_args, bnb_config=bnb_config),
    )
    special_tokens = {
        "pad_token": DEFAULT_TOKENS.get("pad_token", tokenizer.pad_token),
        "eos_token": DEFAULT_TOKENS.get("eos_token", tokenizer.eos_token),
        "bos_token": DEFAULT_TOKENS.get("bos_token", tokenizer.bos_token),
        "unk_token": DEFAULT_TOKENS.get("unk_token", tokenizer.unk_token),
        "additional_special_tokens": SPECIAL_DELM_TOKENS,
    }
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens, tokenizer=tokenizer, model=model)
    return model, config


def build_custom_model(known, model_args, training_args, tokenizer,
                       frontend_delimiters, bnb_config=None):
    """Family/arch-specific config + model class with delimiter ids set."""
    Cfg, Model = pick_model(known.model_family, known.arch)

    config = Cfg.from_pretrained(model_args.model_name_or_path)
    if known.arch == "air":
        config.apply_intermediate_shifts = True

    delms = DELIMITERS[frontend_delimiters]
    num_labels = len(delms)
    if num_labels == 4:
        inst, data, resp = delms[1], delms[2], delms[3]
    else:
        inst, data, resp = delms[0], delms[1], delms[-1]

    # VL: DRIP fields live on config.text_config, not the top-level config.
    cfg_target = getattr(config, "text_config", config)
    set_delimiter_ids_in_config(
        cfg_target, tokenizer,
        inst_delm=inst, data_delm=data, response_delm=resp,
        num_labels=num_labels,
    )

    model = Model.from_pretrained(
        model_args.model_name_or_path,
        ignore_mismatched_sizes=True,
        config=config,
        trust_remote_code=True,
        **_common_load_kwargs(training_args, bnb_config=bnb_config),
    )
    return model, config, Model


# =============================================================================
# Data module
# =============================================================================

def build_data_module(known, tokenizer, data_args, training_args, frontend_delimiters):
    if known.objective in ("dpo"):
        return make_dpo_data_module(
            tokenizer=tokenizer, data_args=data_args,
            frontend_delimiters=frontend_delimiters,
            max_length=getattr(training_args, "model_max_length", 2048),
        )

    if known.objective == "secalign_dpo":
        train_dataset = load_dataset(
            "json", data_files=data_args.data_path_list, split="train")
        return {"train_dataset": train_dataset, "eval_dataset": None}

    # sft / struq_sft share the same lr-scaling logic
    builder = (make_supervised_data_module if known.objective == "sft"
               else make_supervised_data_module_orig)
    data_module = builder(
        tokenizer=tokenizer, data_args=data_args,
        frontend_delimiters=frontend_delimiters,
        downsample=getattr(training_args, "downsample", False),
    )
    if (not getattr(training_args, "downsample", True)
            and getattr(training_args, "lr_scale", True)):
        training_args.learning_rate /= data_module["train_dataset"].data_copy_count
    return data_module


# =============================================================================
# Reference model (for DPO variants)
# =============================================================================

def build_ref_model(Model, model_args, config, bnb_config=None):
    if bnb_config is not None:
        return None

    import copy
    ref_config = copy.deepcopy(config)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ref = Model.from_pretrained(
        model_args.model_name_or_path,
        config=ref_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map={"": local_rank},
    )

    with torch.no_grad():
        for n, p in ref.named_parameters():
            if "deinstruction_shift" in n:
                p.zero_()

    ref.config.use_cache = False
    for n, p in ref.named_parameters():
        if "deinstruction_shift" in n:
            print(n, p.norm().item())  # should be 0
            break
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    return ref


# =============================================================================
# Trainer dispatch
# =============================================================================

def build_trainer(known, model, ref_model, tokenizer, training_args, data_module):

    if known.objective == "dpo":
        cls = DPOTrainerOurs
        return cls(model=model,
                   ref_model=ref_model,
                   processing_class=tokenizer,
                   args=training_args,
                   **data_module)
    elif known.objective == "secalign_dpo":
        return DPOTrainer(model, args=training_args,
                          processing_class=tokenizer,
                          **data_module)

    # SFT objectives
    if known.arch == "ise":
        cls = SFTTrainerISE
    elif known.arch in _FUSE_LIKE:
        cls = SFTTrainerOurs
    else:
        cls = transformers.Trainer
    return cls(model=model, tokenizer=tokenizer, args=training_args, **data_module)


# =============================================================================
# Resume logic
# =============================================================================

_FALSY = {"false", "0", "no", ""}
_TRUTHY = {"true", "1", "yes"}


def resolve_resume(training_args, trainer):
    """Map --resume_from_checkpoint into a value the Trainer accepts.

    Returns:
        None  -> start fresh
        True  -> auto-resume from latest checkpoint-* in output_dir
        path  -> resume from explicit path
    """
    arg = getattr(training_args, "resume_from_checkpoint", None)

    # Falsy
    if arg is None or arg is False or (isinstance(arg, str) and arg.strip().lower() in _FALSY):
        return None

    # Explicit path (anything that isn't bool True or a truthy string)
    if not (arg is True or (isinstance(arg, str) and arg.strip().lower() in _TRUTHY)):
        if trainer.is_world_process_zero():
            logger.info(f"[resume] Resuming from explicit checkpoint path: {arg}")
        return arg

    # Auto-detect latest under output_dir
    out = training_args.output_dir
    ckpts = []
    if os.path.isdir(out):
        for d in os.listdir(out):
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(out, d)):
                try:
                    ckpts.append((int(d.split("-")[1]), d))
                except ValueError:
                    pass
    if not ckpts:
        if trainer.is_world_process_zero():
            logger.info(f"[resume] No checkpoint-* found in {out}; starting fresh.")
        return None

    if trainer.is_world_process_zero():
        logger.info(f"[resume] Auto-resuming from latest checkpoint: {max(ckpts)[1]}")
    return True


# =============================================================================
# Main
# =============================================================================

def main(argv: Optional[List[str]] = None):
    known, model_args, data_args, training_args, attack_args = parse_args(argv)

    data_args.attack = getattr(attack_args, "attack", None)
    logger.info(f"Objective={known.objective} | Family={known.model_family} | Arch={known.arch}")
    if known.use_qlora:
        logger.info(f"[QLoRA] enabled with {known.qlora_bits}-bit quantization")
    print("\n\n" + training_args.output_dir + "\n\n")

    # ---- QLoRA config (None if disabled) ----
    bnb_config = build_bnb_config(known.qlora_bits) if known.use_qlora else None

    # ---- Tokenizer ----
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=getattr(training_args, "cache_dir", None),
        model_max_length=getattr(training_args, "model_max_length", 512),
        padding_side=getattr(model_args, "padding_side", "right"),
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    frontend_delimiters = (data_args.attack or "").split("_")[0] if data_args.attack else ""

    # ---- Model ----
    use_base_path = (known.arch == "base"
                     or known.objective in ("secalign_dpo", "struq_sft"))
    if use_base_path:
        model, config = build_base_model(
            model_args, training_args, tokenizer, bnb_config=bnb_config)
        Model = None  # ref model construction skipped on this path
    else:
        model, config, Model = build_custom_model(
            known, model_args, training_args, tokenizer, frontend_delimiters,
            bnb_config=bnb_config)

    # ---- Trainable params ----
    if known.use_qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        model = get_peft_model(model, build_lora_config(known.objective, known.arch, known.model_family))

    elif known.arch == "air":
        model = freeze_base_train_shifts(model, "intermediate_shifts")
    else:
        model = get_peft_model(model, build_lora_config(known.objective, known.arch, known.model_family))

    for n, p in model.named_parameters():
        if any(k in n for k in ["deinstruction_shift", "residual_weight"]):
            p.requires_grad = True

    # ---- Data ----
    data_module = build_data_module(
        known, tokenizer, data_args, training_args, frontend_delimiters)

    if (getattr(model_args, "window_size", 0) > 0
            and hasattr(model.config, "window")):
        model.config.window = model_args.window_size

    # ---- Trainer (with ref model for DPO) ----
    needs_ref = known.objective in ("dpo")
    ref_model = (build_ref_model(Model, model_args, config, bnb_config=bnb_config)
                 if needs_ref else None)
    trainer = build_trainer(
        known, model, ref_model, tokenizer, training_args, data_module)

    if hasattr(trainer.model, "print_trainable_parameters"):
        print("=== Trainable params (PEFT) ===")
        trainer.model.print_trainable_parameters()

    if hasattr(trainer, "accelerator") and hasattr(trainer.model, "_set_static_graph"):
        pass  # HF Trainer wraps later; use the callback below instead

    # ---- Sanity check: print one sample before training ----
    dl = trainer.get_train_dataloader()
    batch = next(iter(dl))

    if trainer.is_world_process_zero():
        print("\n" + "=" * 80)
        print("SANITY CHECK: one sample before training")
        print("=" * 80)

        cfg = getattr(model, "config", config)
        dcfg = getattr(cfg, "text_config", None) or cfg
        print(f"\n[Config]")
        print(f"  num_labels       = {getattr(dcfg, 'num_labels', '?')}")
        print(f"  instruct_label   = {getattr(dcfg, 'instruct_label', '?')}")
        print(f"  data_label       = {getattr(dcfg, 'data_label', '?')}")
        print(f"  response_label   = {getattr(dcfg, 'response_label', '?')}")
        print(f"  inst_delm_ids    = {getattr(dcfg, 'inst_delm_ids', None)}")
        print(f"  data_delm_ids    = {getattr(dcfg, 'data_delm_ids', None)}")
        print(f"  response_delm_ids= {getattr(dcfg, 'response_delm_ids', None)}")
        print(f"  residual         = {getattr(dcfg, 'residual', '?')}")
        print(f"  bit_flip         = {getattr(dcfg, 'bit_flip', '?')}")

        print(f"\n[Trainable params]")
        trainable = 0
        total = 0
        key_params = []
        for n, p in model.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
                if any(k in n for k in ["deinstruction_shift", "residual_weight"]):
                    key_params.append((n, p.numel(), p.dtype, p.requires_grad))
        print(f"  trainable: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")
        for n, num, dt, rg in key_params:
            print(f"    {n}  numel={num:,}  dtype={dt}  requires_grad={rg}")

        if batch is not None:
            print(f"\n[Batch keys] {list(batch.keys())}")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: shape={tuple(v.shape)}  dtype={v.dtype}")

            c_ids = batch["chosen_input_ids"][0]
            c_ex = batch["chosen_expert_labels"][0]
            c_am = batch["chosen_attention_mask"][0]
            valid_len = int(c_am.sum().item())

            print(f"\n[Chosen sample 0] valid_len={valid_len}")
            text = tokenizer.decode(c_ids[:valid_len], skip_special_tokens=False)
            print(text[:2000] + ("...[TRUNCATED]" if len(text) > 2000 else ""))

            print(f"\n[Expert label distribution (chosen[0], valid tokens only)]")
            ex_valid = c_ex[:valid_len]
            dcfg_ref = getattr(getattr(model, "config", config), "text_config", None) or getattr(model, "config",
                                                                                                 config)
            for lab in range(getattr(dcfg_ref, "num_labels", 4)):
                cnt = int((ex_valid == lab).sum().item())
                print(f"  label {lab}: {cnt} tokens ({100 * cnt / valid_len:.1f}%)")

            print(f"\n[Label transitions (first 5)]")
            transitions = []
            for i in range(1, valid_len):
                if c_ex[i] != c_ex[i - 1]:
                    transitions.append(i)
                if len(transitions) >= 5:
                    break
            for pos in transitions:
                ctx = tokenizer.decode(c_ids[max(0, pos - 3):pos + 5])
                print(f"  pos={pos}  {int(c_ex[pos - 1])} -> {int(c_ex[pos])}  ctx={repr(ctx)}")

            unique_labels = set(c_ex[:valid_len].tolist())
            expected = set(range(getattr(dcfg_ref, "num_labels", 4)))
            ok = unique_labels == expected
            print(f"\n[Sanity checks]")
            print(f"  all {len(expected)} labels present: {ok}  (got {sorted(unique_labels)})")

        print("=" * 80 + "\n")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # ---- Train ----
    trainer.model.enable_input_require_grads()
    trainer.train(resume_from_checkpoint=resolve_resume(training_args, trainer))

    # ---- Save ----
    # For LoRA runs, merge the adapter into the base weights so evaluation can
    # load a full checkpoint directly (the default path in testing/test.py).
    # QLoRA/4-bit adapters cannot be merged, so those are saved as adapters
    # (load them with --load_as_adapter). AIR/full-finetune (non-PEFT) just
    # saves the full model.
    trainer.save_state()
    if trainer.is_world_process_zero():
        logger.info("Training done. Saving...")
        to_save = trainer.accelerator.unwrap_model(trainer.model)
        if bnb_config is None and hasattr(to_save, "merge_and_unload"):
            try:
                to_save = to_save.merge_and_unload()
                logger.info("Merged LoRA adapter into base weights.")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"merge_and_unload failed ({e}); saving adapter instead.")
        to_save.save_pretrained(training_args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(training_args.output_dir)
        # Persist the (DRIP) config — incl. delimiter ids — alongside the weights.
        getattr(to_save, "base_model", to_save).config.save_pretrained(training_args.output_dir)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()