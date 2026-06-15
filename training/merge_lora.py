"""Merge a trained LoRA adapter into its base model and save a full checkpoint.
Usage:
    python -m training.merge_lora \
        --adapter_path  out/Meta-Llama-3-8B-Instruct-TextTextText-sep-drip \
        --output_path   out/Meta-Llama-3-8B-Instruct-TextTextText-sep-drip-merged

    # base path / model class are inferred from the adapter path; override if needed:
    python -m training.merge_lora \
        --adapter_path <dir> \
        --output_path <dir> \
        --base_model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --customized_model_class LlamaForCausalLMDRIP
"""

import argparse
from testing.test import load_full_model


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--adapter_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--base_model_path", default=None)
    ap.add_argument("--customized_model_class", default="")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device_map = int(args.device) if args.device.isdigit() else args.device

    model, tokenizer, _, _ = load_full_model(
        args.adapter_path,
        customized_model_class=args.customized_model_class,
        load_as_adapter=True,
        base_model_path=args.base_model_path,
        device_map=device_map,
    )

    if not hasattr(model, "merge_and_unload"):
        raise SystemExit("Loaded model is not a PEFT adapter. Check --adapter_path / --customized_model_class.")

    TRACKED = ["deinstruction_shift", "lm_head", "embed_tokens"]

    # Snapshot base weights (inside the PeftModel, base_model.model.*)
    base_snapshots = {}
    for n, p in model.named_parameters():
        if "base_model" in n and not "modules_to_save" in n and any(k in n for k in TRACKED):
            base_snapshots[n] = p.detach().clone()
            print(f"BASE:      {n}  norm={p.norm().item():.4f}")

    # Snapshot adapter-overridden weights (modules_to_save.default.*)
    pre_merge_snapshots = {}
    for n, p in model.named_parameters():
        if "modules_to_save.default" in n and any(k in n for k in TRACKED):
            pre_merge_snapshots[n] = p.detach().clone()
            print(f"PRE-MERGE: {n}  norm={p.norm().item():.4f}")

    print("\nMerging adapter into base weights ...")
    merged = model.merge_and_unload()

    print("\n=== POST-MERGE diffs vs BASE ===")
    for n, p in merged.named_parameters():
        if any(k in n for k in TRACKED):
            parts = n.rsplit(".", 1)  # ["lm_head", "weight"]
            base_key = f"base_model.model.{parts[0]}.original_module.{parts[1]}"
            if base_key in base_snapshots:
                diff = (p - base_snapshots[base_key]).abs().max().item()
                changed = diff > 1e-6
                print(
                    f"{n}: norm={p.norm().item():.4f}  max_diff={diff:.4e}  {'CHANGED ✓' if changed else 'UNCHANGED ✗'}")
            else:
                print(f"{n}: norm={p.norm().item():.4f}  (no base match, tried {base_key})")

    merged.save_pretrained(args.output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.output_path)
    print(f"\nMerged checkpoint saved to {args.output_path}")


if __name__ == "__main__":
    main()