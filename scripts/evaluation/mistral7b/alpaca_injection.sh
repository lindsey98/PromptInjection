#!/bin/bash
# Source Conda configuration
source ~/anaconda3/etc/profile.d/conda.sh
conda activate prompt  # Replace with your actual env name

set -euo pipefail

# === Prompt for CUDA device ===
read -p "Enter CUDA device ID to use (default: 0): " CUDA_ID
CUDA_ID=${CUDA_ID:-0}  # Default to 0 if empty

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
    *instfuse*)
        EXTRA_FLAGS="--customized_model_class MistralForCausalLMDRIP"
        ;;
    *ise*)
        EXTRA_FLAGS="--customized_model_class MistralForCausalLMISE"
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

# === Define Common Arguments ===
COMMON_ARGS="--model_name_or_path $MODEL_PATH $EXTRA_FLAGS"

# === Define Tasks (Functions inherit current Conda env) ===
task0() { CUDA_VISIBLE_DEVICES=$DEV0 python -m testing.test $COMMON_ARGS --base_model_path mistralai/Mistral-7B-Instruct-v0.3 --attack ignore_0 ignore_1 ignore_2 ignore_3 ignore_4; }
task1() { CUDA_VISIBLE_DEVICES=$DEV1 python -m testing.test $COMMON_ARGS --base_model_path mistralai/Mistral-7B-Instruct-v0.3 --attack ignore_5 ignore_6 ignore_7 ignore_8 ignore_9 ignore_10; }
task2() { CUDA_VISIBLE_DEVICES=$DEV2 python -m testing.test $COMMON_ARGS --base_model_path mistralai/Mistral-7B-Instruct-v0.3 --attack completion_real completion_realcmb completion_real_chinese completion_real_spanish completion_real_base64 completion_other; }
task3() { CUDA_VISIBLE_DEVICES=$DEV3 python -m testing.test $COMMON_ARGS --base_model_path mistralai/Mistral-7B-Instruct-v0.3 --attack completion_othercmb completion_close_1hash completion_close_2hash completion_close_0hash completion_close_upper completion_close_title; }
task4() { CUDA_VISIBLE_DEVICES=$DEV4 python -m testing.test $COMMON_ARGS --base_model_path mistralai/Mistral-7B-Instruct-v0.3 --attack completion_close_nospace completion_close_nocolon completion_close_typo completion_close_similar completion_close_ownlower completion_close_owntitle; }
task5() { CUDA_VISIBLE_DEVICES=$DEV5 python -m testing.test $COMMON_ARGS --base_model_path mistralai/Mistral-7B-Instruct-v0.3 --attack naive completion_close_ownhash completion_close_owndouble escape_separation escape_deletion hackaprompt; }

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
set +e  # allow background jobs to fail without killing the script immediately

# Execute functions in background
task0 & pid0=$!
task1 & pid1=$!
task2 & pid2=$!
task3 & pid3=$!
task4 & pid4=$!
task5 & pid5=$!

# Wait and collect statuses
wait $pid0; s0=$?
wait $pid1; s1=$?
wait $pid2; s2=$?
wait $pid3; s3=$?
wait $pid4; s4=$?
wait $pid5; s5=$?

echo "Exit codes → cmd0:$s0  cmd1:$s1  cmd2:$s2  cmd3:$s3  cmd4:$s4 cmd5:$s5"

# Make the whole script fail if any failed
if [ $s0 -ne 0 ] || [ $s1 -ne 0 ] || [ $s2 -ne 0 ] || [ $s3 -ne 0 ] || [ $s4 -ne 0 ] || [ $s5 -ne 0 ]; then
  exit 1
fi