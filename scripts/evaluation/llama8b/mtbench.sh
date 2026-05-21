#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

# === Prompt for CUDA device ===
read -p "Enter CUDA device ID to use (default: 0): " CUDA_ID
CUDA_ID=${CUDA_ID:-0}  # Default to 0 if empty

if ! [[ "$CUDA_ID" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Invalid CUDA device ID: $CUDA_ID"
    exit 1
fi
echo "Using CUDA device $CUDA_ID"

# === Prompt for model_name_or_path ===
read -p "Enter model_name_or_path: " MODEL_PATH

if [ -z "$MODEL_PATH" ]; then
    echo "[ERROR] model_name_or_path cannot be empty"
    exit 1
fi
echo "Model path: $MODEL_PATH"
MODEL_ID=$(basename "$MODEL_PATH")

# === Detect flags based on model path ===
EXTRA_FLAGS=""

case "$MODEL_PATH" in
    *instfuse*nofusion*)
        EXTRA_FLAGS="--customized_model_class LlamaForCausalLMNoFuse"
        ;;
    *instfuse*concatfusion*)
        EXTRA_FLAGS="--customized_model_class LlamaForCausalLMConcatFuse"
        ;;
    *instfuse*embeddingshift*)
        EXTRA_FLAGS="--customized_model_class LlamaForCausalLMEmbeddingShift"
        ;;
    *instfuse*)
        EXTRA_FLAGS="--customized_model_class LlamaForCausalLMDRIP"
        ;;
    *ise*)
        EXTRA_FLAGS="--customized_model_class LlamaForCausalLMISE"
        ;;
    *air*)
        EXTRA_FLAGS="--customized_model_class LlamaForCausalLMAIR"
        ;;
    *possep*)
        EXTRA_FLAGS="--customized_model_class LlamaForCausalLMPFT"
        ;;
esac

if [ -n "$EXTRA_FLAGS" ]; then
    echo "Detected special model type → Adding flags: $EXTRA_FLAGS"
else
    echo "No special model type detected → Running without extra flags"
fi

echo "Executing test..."
CMD="CUDA_VISIBLE_DEVICES=$CUDA_ID python -m testing.mt_bench.gen_model_answer \
--model-path $MODEL_PATH --model-id $MODEL_ID \
$EXTRA_FLAGS"

echo
echo "⚙ Running:"
echo "$CMD"
echo

# -----------------------------
# Execute
# -----------------------------
eval $CMD