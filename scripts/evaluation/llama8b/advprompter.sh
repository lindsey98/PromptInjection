#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

########################################
# 1. CUDA device
########################################
read -p "Enter CUDA device ID to use (default: 0): " CUDA_ID
CUDA_ID=${CUDA_ID:-0}

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

DATASET_TEXTTEXTTEXT="./advprompter/data/prompt_injections/dataset/test_TextTextText_datasets_davinci_003_outputs_json.csv"
DATASET_SPCLSPCLSPCL="./advprompter/data/prompt_injections/dataset/test_SpclSpclSpcl_datasets_davinci_003_outputs_json.csv"
WANDB_FLAGS="wandb_params.enable_wandb=false"

case "$MODEL_PATH" in
    *instfuse*)
        BASE_CMD_SPCLSPCLSPCL="python -m advprompter.main --config-name test target_llm=spcl_delm_llm"
        ;;
    *ise*)
        EXTRA_FLAGS="--customized_model_class LlamaForCausalLMISE"
        ;;
    *possep*)
        EXTRA_FLAGS="--customized_model_class LlamaForCausalLMPFT"
        ;;
esac


CMD="$BASE_CMD \
target_llm.llm_params.model_name=$MODEL_NAME \
target_llm.llm_params.checkpoint=$MODEL_PATH \
train.prompter_optim_params.lr=1e-5 \
train.dataset_pth=$DATASET \
eval.data.dataset_pth_dct.train=$DATASET \
$WANDB_FLAGS"

# append auto-detected flags (if any)
if [[ -n "$EXTRA_MODEL_FLAGS" ]]; then
    CMD="$CMD $EXTRA_MODEL_FLAGS"
fi

########################################
# 7. Final run
########################################
FULL_CMD="CUDA_VISIBLE_DEVICES=$CUDA_ID $CMD"

echo
echo "⚙ Running:"
echo "$FULL_CMD"
echo

eval "$FULL_CMD"