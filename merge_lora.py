"""Merge a trained LoRA adapter into its base model and save a full checkpoint.

Use this for checkpoints saved as adapters — e.g. QLoRA runs, or models trained
before train_unified.py started merging at save time. It loads the model exactly
the way evaluation does (testing/test.py `load_full_model`, load_as_adapter=True),
so the merged result is guaranteed consistent with eval, then merges and saves a
standalone checkpoint that the default eval path can load directly.

Usage:
    python merge_lora.py \
        --adapter_path  out/Meta-Llama-3-8B-Instruct-TextTextText-sep-drip \
        --output_path   out/Meta-Llama-3-8B-Instruct-TextTextText-sep-drip-merged

    # base path / model class are inferred from the adapter path; override if needed:
    python merge_lora.py --adapter_path <dir> --output_path <dir> \
        --base_model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --customized_model_class LlamaForCausalLMDRIP
"""

import argparse

from testing.test import load_full_model


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--adapter_path", required=True,
                    help="Directory of the trained LoRA adapter checkpoint.")
    ap.add_argument("--output_path", required=True,
                    help="Where to write the merged full checkpoint.")
    ap.add_argument("--base_model_path", default=None,
                    help="Base model (inferred from --adapter_path if omitted).")
    ap.add_argument("--customized_model_class", default="",
                    help="REGISTRY class key (auto-detected from the path if omitted).")
    ap.add_argument("--device", default="auto",
                    help="device_map for loading (e.g. 'auto', '0', 'cpu').")
    args = ap.parse_args()

    device_map = int(args.device) if args.device.isdigit() else args.device

    # Same loading path as evaluation; load_as_adapter attaches the adapter onto
    # the base model. The model class / delimiters are resolved internally.
    model, tokenizer = load_full_model(
        args.adapter_path,
        customized_model_class=args.customized_model_class,
        load_as_adapter=True,
        base_model_path=args.base_model_path,
        device_map=device_map,
    )

    if not hasattr(model, "merge_and_unload"):
        raise SystemExit(
            "Loaded model is not a PEFT adapter (nothing to merge). "
            "Check --adapter_path / --customized_model_class."
        )

    print("Merging adapter into base weights ...")
    merged = model.merge_and_unload()

    merged.save_pretrained(args.output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.output_path)
    # Persist the (DRIP) config — including delimiter ids — alongside the weights.
    getattr(merged, "base_model", merged).config.save_pretrained(args.output_path)
    print(f"Merged checkpoint saved to {args.output_path}")


if __name__ == "__main__":
    main()
