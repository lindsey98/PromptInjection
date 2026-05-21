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
        EXTRA_FLAGS="--customized_model_class LlamaForCausalLMDRIP"
        ;;
    *ise*)
        EXTRA_FLAGS="--customized_model_class LlamaForCausalLMISE"
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
CMD="torchrun --nproc_per_node=6 -m testing.tracin --model_name_or_path $MODEL_PATH $EXTRA_FLAGS"

echo
echo "⚙ Running:"
echo "$CMD"
echo

# -----------------------------
# Execute
# -----------------------------
eval $CMD