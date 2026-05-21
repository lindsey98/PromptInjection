#!/bin/bash

SCRIPT_PATH="train_unified.py"
BASELINE="air"
BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.3-air"
DATA_PATH="datasets/sep/sep_data_origdata_dpo.json"
FILENAME=$(basename "$DATA_PATH")
PREFIX=${FILENAME%%_*}
DELIMITER="TextTextTextMistral"

SAVE_PATH="${BASE_MODEL}-${DELIMITER}-${BASELINE}-${PREFIX}-none"

BATCH_SIZE=4
EPOCH=1

OBJECTIVE="dpo"
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
  --learning_rate 5e-6 \
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