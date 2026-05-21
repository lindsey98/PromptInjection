#!/bin/bash
# Source Conda configuration
source ~/anaconda3/etc/profile.d/conda.sh
conda activate prompt

set -euo pipefail

# === Prompt for CUDA device ===
read -p "Enter CUDA device ID to use (default: 0): " CUDA_ID
CUDA_ID=${CUDA_ID:-0}

if command -v nvidia-smi >/dev/null 2>&1; then
  NGPUS=$(nvidia-smi -L | wc -l | tr -d ' ')
else
  NGPUS=1
fi
[ "$NGPUS" -ge 1 ] || NGPUS=1

# Calculate device IDs cyclically
DEV0=$(( (CUDA_ID + 0) % NGPUS ))
DEV1=$(( (CUDA_ID + 1) % NGPUS ))
DEV2=$(( (CUDA_ID + 2) % NGPUS ))
DEV3=$(( (CUDA_ID + 3) % NGPUS ))
DEV4=$(( (CUDA_ID + 4) % NGPUS ))
DEV5=$(( (CUDA_ID + 5) % NGPUS ))

echo "GPU mapping → dev0:$DEV0  dev1:$DEV1  dev2:$DEV2  dev3:$DEV3  dev4:$DEV4 dev5:$DEV5  (NGPUS=$NGPUS)"

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
    *instfuse*nofusion*)      EXTRA_FLAGS="--customized_model_class LlamaForCausalLMNoFuse" ;;
    *instfuse*concatfusion*)  EXTRA_FLAGS="--customized_model_class LlamaForCausalLMConcatFuse" ;;
    *instfuse*embeddingshift*) EXTRA_FLAGS="--customized_model_class LlamaForCausalLMEmbeddingShift" ;;
    *instfuse*)               EXTRA_FLAGS="--customized_model_class LlamaForCausalLMDRIP" ;;
    *ise*)                    EXTRA_FLAGS="--customized_model_class LlamaForCausalLMISE" ;;
    *possep*)                 EXTRA_FLAGS="--customized_model_class LlamaForCausalLMPFT" ;;
esac

if [ -n "$EXTRA_FLAGS" ]; then
    echo "Detected special model type → Adding flags: $EXTRA_FLAGS"
else
    echo "No special model type detected → Running without extra flags"
fi

# === Define base arguments ===
COMMON_ARGS="--model_name_or_path $MODEL_PATH $EXTRA_FLAGS"

# === Define specific tasks ===
#task0() { CUDA_VISIBLE_DEVICES=$DEV0 python -m testing.sep.test_sep $COMMON_ARGS --attack inject_pos_0 inject_pos_10 inject_pos_20 inject_pos_30; }
#task1() { CUDA_VISIBLE_DEVICES=$DEV1 python -m testing.sep.test_sep $COMMON_ARGS --attack inject_pos_40 inject_pos_50 inject_pos_60 inject_pos_70; }
#task2() { CUDA_VISIBLE_DEVICES=$DEV2 python -m testing.sep.test_sep $COMMON_ARGS --attack inject_pos_80 inject_pos_90 inject_pos_100; }
#task3() { CUDA_VISIBLE_DEVICES=$DEV3 python -m testing.sep.test_sep $COMMON_ARGS --attack stress_repeat_2 stress_repeat_4 stress_repeat_6; }
#task4() { CUDA_VISIBLE_DEVICES=$DEV4 python -m testing.sep.test_sep $COMMON_ARGS --attack stress_repeat_8 stress_repeat_10 stress_repeat_12; }
#task5() { CUDA_VISIBLE_DEVICES=$DEV5 python -m testing.sep.test_sep $COMMON_ARGS --attack stress_repeat_14 stress_repeat_16 stress_repeat_18 stress_repeat_20; }

task0() { CUDA_VISIBLE_DEVICES=$DEV3 python -m testing.sep.test_sep $COMMON_ARGS --base_model_path meta-llama/Meta-Llama-3-8B-Instruct --attack stress_repeat_6; }
task1() { CUDA_VISIBLE_DEVICES=$DEV4 python -m testing.sep.test_sep $COMMON_ARGS --base_model_path meta-llama/Meta-Llama-3-8B-Instruct --attack stress_repeat_12; }

echo
echo "⚙ Running in parallel..."

cleanup() {
  echo "⚠ Caught termination signal → killing children..."
  jobs -p | xargs -r kill
}
trap cleanup SIGINT SIGTERM EXIT

# -----------------------------
# Execute in parallel
# -----------------------------
set +e

task0 & pid0=$!
task1 & pid1=$!
#task2 & pid2=$!
#task3 & pid3=$!
#task4 & pid4=$!
#task5 & pid5=$!

wait $pid0; s0=$?
wait $pid1; s1=$?
#wait $pid2; s2=$?
#wait $pid3; s3=$?
#wait $pid4; s4=$?
#wait $pid5; s5=$?

echo "Exit codes"

#if [ $s0 -ne 0 ] || [ $s1 -ne 0 ] || [ $s2 -ne 0 ] || [ $s3 -ne 0 ] || [ $s4 -ne 0 ] || [ $s5 -ne 0 ]; then
if [ $s0 -ne 0 ] || [ $s1 -ne 0 ]; then
  exit 1
fi