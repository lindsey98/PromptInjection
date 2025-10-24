#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# =========================
# Configurable: defenses
# =========================
DEFENSES=(
  "sandwich"
  "reminder"
  "fakecompletion"
  "thinkintervene"
  "spotlight_delimit"
  "spotlight_datamark"
  "spotlight_encode"
)

# =========================
# Prompt for model path
# =========================
read -p "Enter model_name_or_path: " MODEL_PATH
if [ -z "${MODEL_PATH}" ]; then
  echo "[ERROR] model_name_or_path cannot be empty"
  exit 1
fi
echo "Model path: $MODEL_PATH"

# =========================
# Prompt for CUDA devices
# =========================
read -p "Enter comma-separated CUDA device IDs to use (default: 0,1,2,3): " CUDA_LIST_RAW
CUDA_LIST_RAW=${CUDA_LIST_RAW:-"0,1,2,3"}

# Parse to array
IFS=',' read -r -a CUDA_IDS <<< "$CUDA_LIST_RAW"
# Validate device IDs
for id in "${CUDA_IDS[@]}"; do
  if ! [[ "$id" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Invalid CUDA device ID: $id"
    exit 1
  fi
done
echo "Using CUDA devices: ${CUDA_IDS[*]}"

# =========================
# Extra flags (same as your script)
# =========================
EXTRA_FLAGS="--pass_expert_labels --customized_model_class LlamaForCausalLMFuse"
if [ -n "$EXTRA_FLAGS" ]; then
  echo "Detected special model type → Adding flags: $EXTRA_FLAGS"
else
  echo "No special model type detected → Running without extra flags"
fi

# =========================
# Setup
# =========================
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Track PIDs and names for summary
declare -A PID_NAME=()
declare -A PID_DEVICE=()

# Ensure children are cleaned up on ctrl-c/exit
cleanup() {
  echo
  echo "Cleaning up background jobs..."
  for pid in "${!PID_NAME[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping ${PID_NAME[$pid]} (pid $pid) on GPU ${PID_DEVICE[$pid]}..."
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup INT TERM

# =========================
# Launch jobs round-robin
# =========================
echo
echo "=== Launching parallel runs ==="
idx=0
for defense in "${DEFENSES[@]}"; do
  gpu="${CUDA_IDS[$((idx % ${#CUDA_IDS[@]}))]}"
  idx=$((idx+1))

  LOG_FILE="${LOG_DIR}/${defense}.log"
  CMD="CUDA_VISIBLE_DEVICES=${gpu} python -m testing.sep.test_sep \
    --model_name_or_path \"$MODEL_PATH\" \
    $EXTRA_FLAGS \
    --defense \"$defense\""

  echo
  echo "⚙ Launching: $defense on GPU $gpu"
  echo "→ Log: $LOG_FILE"
  echo "→ Cmd: $CMD"
  echo

  # Run in background with stdout/stderr to log
  bash -c "$CMD" > \"$LOG_FILE\" 2>&1 &
  pid=$!
  PID_NAME[$pid]="$defense"
  PID_DEVICE[$pid]="$gpu"
done

# =========================
# Wait and summarize
# =========================
echo
echo "Waiting for all jobs to finish..."
declare -i FAILS=0
for pid in "${!PID_NAME[@]}"; do
  defense="${PID_NAME[$pid]}"
  gpu="${PID_DEVICE[$pid]}"

  if wait "$pid"; then
    echo "✅ ${defense} (GPU ${gpu}) finished successfully."
  else
    echo "❌ ${defense} (GPU ${gpu}) failed. See logs/${defense}.log"
    FAILS=$((FAILS+1))
  fi
done

echo
if (( FAILS == 0 )); then
  echo "🎉 All runs completed successfully."
else
  echo "⚠ ${FAILS} run(s) failed. Check corresponding logs in ./logs/"
fi
