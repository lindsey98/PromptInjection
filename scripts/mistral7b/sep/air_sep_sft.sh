#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NCCL_NTHREADS=8
export TOKENIZERS_PARALLELISM=false

# === NCCL hang protection ===
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TIMEOUT_MS=1800000
export TORCH_NCCL_TRACE_BUFFER_SIZE=20480
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6

SCRIPT_PATH="train_unified.py"
BASELINE="air"
BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
DATA_PATH="datasets/sep/sep_data_cleaned.json"
FILENAME=$(basename "$DATA_PATH")
PREFIX=${FILENAME%%_*}
FSDP_CONFIG="training/config/fsdp_config_mistral.json"
DELIMITER="TextTextTextMistral-3roles"

SAVE_PATH="${BASE_MODEL}-${DELIMITER}-${BASELINE}-${PREFIX}-sft"

BATCH_SIZE=4
EPOCH=1

OBJECTIVE="sft"
MODEL_FAMILY="mistral"
ARCH="air"

python -m torch.distributed.run --nproc_per_node=6 --master_port=29951 "$SCRIPT_PATH" \
  --objective "${OBJECTIVE}" \
  --model-family "${MODEL_FAMILY}" \
  --arch "${ARCH}" \
  --use_qlora \
  --qlora_bits 4 \
  --model_name_or_path "$BASE_MODEL" \
  --data_path "$DATA_PATH" \
  --output_dir "$SAVE_PATH" \
  --num_train_epochs "$EPOCH" \
  --bf16 True \
  --gradient_checkpointing True \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --save_strategy "steps" \
  --save_steps 100 \
  --save_total_limit 3 \
  --learning_rate 1e-4 \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.0 \
  --weight_decay 0. \
  --lr_scheduler_type "linear" \
  --logging_steps 1 \
  --tf32 True \
  --attack "${DELIMITER}_None" \
  --model_max_length 512 \
  --dataloader_num_workers 2 \
  --optim "adamw_torch" \
  --ddp_find_unused_parameters False