#!/bin/bash

SCRIPT_PATH="train_unified.py"
BASELINE="struq"
BASE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
DATA_PATH="datasets/sep/sep_data_cleaned.json"
FILENAME=$(basename "$DATA_PATH")
PREFIX=${FILENAME%%_*}
FSDP_CONFIG="training/config/fsdp_config.json"
DELIMITER="SpclSpclSpcl"

SAVE_PATH="${BASE_MODEL}-${DELIMITER}-${BASELINE}-${PREFIX}-none"

BATCH_SIZE=4
EPOCH=1

OBJECTIVE="struq_sft"
MODEL_FAMILY="llama"
ARCH="base"

http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 \
python -m torch.distributed.run --nproc_per_node=6 --master_port=29951 "$SCRIPT_PATH" \
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
  --save_strategy "epoch" \
  --learning_rate 1e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --attack "${DELIMITER}_None" \
  --model_max_length 512 \
  --dataloader_num_workers 4 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_config "$FSDP_CONFIG"
