#!/bin/bash

# Source Conda configuration
source ~/anaconda3/etc/profile.d/conda.sh
conda activate prompt  # Replace with your actual env name

set -euo pipefail

# === Prompt for CUDA device ===
read -p "Enter CUDA device ID to use (default: 0): " CUDA_ID
CUDA_ID=${CUDA_ID:-0}  # Default to 0 if empty

# === Prompt for model_name_or_path ===
read -p "Enter model_name_or_path: " MODEL_PATH

if [ -z "$MODEL_PATH" ]; then
    echo "[ERROR] model_name_or_path cannot be empty"
    exit 1
fi
echo "Model path: $MODEL_PATH"

# === Detect flags based on model path ===
EXTRA_FLAGS=""

case "$MODEL_PATH" in
    *instfuse*)
        EXTRA_FLAGS="--customized_model_class MistralForCausalLMDRIP"
        ;;
    *ise*)
        EXTRA_FLAGS="--customized_model_class MistralForCausalLMISE"
        ;;
    *air*)
        EXTRA_FLAGS="--customized_model_class MistralForCausalLMAIR"
        ;;
    *possep*)
        EXTRA_FLAGS="--customized_model_class MistralForCausalLMPFT"
        ;;
esac

if [ -n "$EXTRA_FLAGS" ]; then
    echo "Detected special model type → Adding flags: $EXTRA_FLAGS"
else
    echo "No special model type detected → Running without extra flags"
fi

# === Run command ===
echo "Executing test..."

CMD_ARGS="--model_name_or_path $MODEL_PATH $EXTRA_FLAGS"

echo
echo "⚙ Running:"
echo "CUDA_VISIBLE_DEVICES=$CUDA_ID python -m testing.mmlu.test_mmlu $CMD_ARGS"
echo

# -----------------------------
# Execute
# -----------------------------
CUDA_VISIBLE_DEVICES=$CUDA_ID python -m testing.mmlu.test_mmlu $CMD_ARGS