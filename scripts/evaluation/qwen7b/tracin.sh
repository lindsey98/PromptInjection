#!/bin/bash

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
        EXTRA_FLAGS="--pass_expert_labels --customized_model_class Qwen2ForCausalLMFuse"
        ;;
    *ise*)
        EXTRA_FLAGS="--pass_expert_labels --customized_model_class Qwen2ForCausalLMMoE"
        ;;
    *possep*)
        EXTRA_FLAGS="--pass_expert_labels --customized_model_class Qwen2ForCausalLMMoEV2"
        ;;
esac

if [ -n "$EXTRA_FLAGS" ]; then
    echo "Detected special model type → Adding flags: $EXTRA_FLAGS"
else
    echo "No special model type detected → Running without extra flags"
fi

echo "Executing test..."
CMD="torchrun --nproc_per_node=6 -m testing.tracin --model_name_or_path $MODEL_PATH $EXTRA_FLAGS"

echo
echo "⚙ Running:"
echo "$CMD"
echo

# -----------------------------
# Execute
# -----------------------------
eval $CMD