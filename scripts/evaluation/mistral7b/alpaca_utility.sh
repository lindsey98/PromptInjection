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
        EXTRA_FLAGS="--pass_expert_labels --customized_model_class MistralForCausalLMFuse"
        ;;
    *ise*)
        EXTRA_FLAGS="--pass_expert_labels --customized_model_class MistralForCausalLMMoE"
        ;;
    *possep*)
        EXTRA_FLAGS="--pass_expert_labels --customized_model_class MistralForCausalLMMoEV2"
        ;;
esac

if [ -n "$EXTRA_FLAGS" ]; then
    echo "Detected special model type → Adding flags: $EXTRA_FLAGS"
else
    echo "No special model type detected → Running without extra flags"
fi

# === Run command ===
echo "Executing test..."

CMD_ARGS="--model_name_or_path $MODEL_PATH $EXTRA_FLAGS --attack none"

echo
echo "⚙ Running:"
echo "CUDA_VISIBLE_DEVICES=$CUDA_ID python -m testing.test $CMD_ARGS"
echo

# -----------------------------
# Execute
# -----------------------------
CUDA_VISIBLE_DEVICES=$CUDA_ID python -m testing.test $CMD_ARGS