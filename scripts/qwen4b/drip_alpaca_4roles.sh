#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/nvme0n1/ruofan/hf_hub/
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NCCL_NTHREADS=8
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

# === NCCL hang protection ===
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TIMEOUT_MS=1800000
export TORCH_NCCL_TRACE_BUFFER_SIZE=20480
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=3,4,5,6

SCRIPT_PATH="train_unified.py"
BASELINE="drip"
BASE_MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
BASE_MODEL="/mnt/nvme0n1/ruofan/hf_hub/Qwen3-4B-Instruct-2507"
DATA_PATH="datasets/alpaca_injecagent_dpo_combined.json"
FILENAME=$(basename "$DATA_PATH")
PREFIX=${FILENAME%%_*}
FSDP_CONFIG="training/config/fsdp_config_qwen.json"
DELIMITER="TextTextTextQwen"
SAVE_PATH="${BASE_MODEL_NAME}-${DELIMITER}-alpaca-injecagent-${BASELINE}"

BATCH_SIZE=4
EPOCH=1
SEED=42

OBJECTIVE="dpo"
MODEL_FAMILY="qwen3"
ARCH="fuse"

python -m torch.distributed.run --nproc_per_node=4 --master_port=29951 "$SCRIPT_PATH" \
  --objective "${OBJECTIVE}" \
  --model-family "${MODEL_FAMILY}" \
  --arch "${ARCH}" \
  --model_name_or_path "$BASE_MODEL" \
  --data_path "$DATA_PATH" \
  --output_dir "$SAVE_PATH" \
  --num_train_epochs "$EPOCH" \
  --bf16 True \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --save_strategy "steps" \
  --save_steps 50 \
  --learning_rate 5e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --attack "${DELIMITER}_None" \
  --model_max_length 2048 \
  --dataloader_num_workers 1 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_config "$FSDP_CONFIG" \
  --resume_from_checkpoint True \
  --seed "$SEED" \
  --data_seed "$SEED"
