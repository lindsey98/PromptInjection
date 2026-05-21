#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate prompt  # Replace with your actual env name

CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --nproc_per_node=3 -m testing.pismith.train_sep \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct-log-TextTextText-instfuse-alpaca-dpo \
    --customized_model_class LlamaForCausalLMDRIP \
    --attack_model_name Qwen/Qwen3-4B-Instruct-2507 \
    --attack_model_path Qwen/Qwen3-4B-Instruct-2507 \
    --output_dir ./pismith_ckpt/alpaca \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_modules q_proj v_proj k_proj o_proj