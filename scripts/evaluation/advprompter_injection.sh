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
# Extract last directory name (model name)
MODEL_NAME=$(basename "$MODEL_PATH")
echo "Model name: $MODEL_NAME"

# === Detect flags based on model path ===
TARGET_LLM="text_delm_llm"
DATASET="./advprompter/data/prompt_injections/dataset/test_TextTextText_datasets_davinci_003_outputs_json.csv"
EXTRA_FLAGS=()

# --- Auto-flags based on model path ---
case "$MODEL_PATH" in
  *struq*)
    TARGET_LLM="spcl_delm_llm"
    DATASET="./advprompter/data/prompt_injections/dataset/test_SpclSpclSpcl_datasets_davinci_003_outputs_json.csv"
    ;;
  *secalign*)
    TARGET_LLM="spcl_delm_llm"
    DATASET="./advprompter/data/prompt_injections/dataset/test_SpclSpclSpcl_datasets_davinci_003_outputs_json.csv"
    ;;
  *instfuse*)
    EXTRA_FLAGS+=(
      "target_llm.llm_params.customized_model_class=LlamaForCausalLMFuse"
      "target_llm.llm_params.pass_expert_labels=true"
    )
    ;;
  *ise*)
    EXTRA_FLAGS+=(
      "target_llm.llm_params.customized_model_class=LlamaForCausalLMMoE"
      "target_llm.llm_params.pass_expert_labels=true"
    )
    ;;
  *possep*)
    EXTRA_FLAGS+=(
      "target_llm.llm_params.customized_model_class=LlamaForCausalLMMoEV2"
      "target_llm.llm_params.pass_expert_labels=true"
    )
    ;;
esac

if (( ${#EXTRA_FLAGS[@]} )); then
  echo "Detected special model type → adding flags: ${EXTRA_FLAGS[*]}"
else
  echo "No special model type detected → running without extra flags"
fi



CMD="CUDA_VISIBLE_DEVICES=$CUDA_ID python -m advprompter.main \
--config-name test \
target_llm=$TARGET_LLM \
target_llm.llm_params.model_name=$MODEL_NAME \
target_llm.llm_params.checkpoint=$MODEL_PATH \
${EXTRA_FLAGS[@]} \
train.prompter_optim_params.lr=1e-5 \
train.dataset_pth=$DATASET \
eval.data.dataset_pth_dct.train=$DATASET \
wandb_params.enable_wandb=true"

echo
echo "⚙ Running:"
echo "$CMD"
echo

# -----------------------------
# Execute
# -----------------------------
eval $CMD