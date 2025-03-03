
# Training

## Adv Train Baseline (without special delimiters)
```bash
python -m torch.distributed.run --nproc_per_node=4 --master_port=29951 train.py
--model_name_or_path meta-llama/Meta-Llama-3-8B
--data_path datasets/alpaca_data_cleaned.json
--output_dir meta-llama/Llama-3-8B-Instruct
--num_train_epochs 3
--per_device_train_batch_size 4
--per_device_eval_batch_size 1
--gradient_accumulation_steps 8
--evaluation_strategy "no"
--save_strategy "epoch"
--learning_rate 2e-4
--weight_decay 0.
--warmup_ratio 0.03
--lr_scheduler_type "cosine"
--logging_steps 1
--tf32 True
--attack NaiveCompletion
--model_max_length 512
--bf16 True
--dataloader_num_workers 4
```

## Adv Train Struq
```bash
python -m torch.distributed.run --nproc_per_node=4 --master_port=29951 train_struq_lora.py \
--model_name_or_path meta-llama/Meta-Llama-3-8B \
--data_path datasets/alpaca_data_cleaned.json \
--output_dir meta-llama/Llama-3-8B-SpclSpclSpcl_NaiveCompletion-struq \
--num_train_epochs 3 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 8 \
--evaluation_strategy "no" \
--save_strategy "epoch" \
--learning_rate 2e-4 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--attack SpclSpclSpcl_NaiveCompletion \
--model_max_length 512 \
--bf16 True \
--dataloader_num_workers 4
```

## Adv Train ISE
```bash
python -m torch.distributed.run --nproc_per_node=3 --master_port=29951 train_ise.py \
--model_name_or_path meta-llama/Meta-Llama-3-8B \
--data_path datasets/alpaca_data_cleaned.json \
--resume_from_checkpoint True \
--output_dir meta-llama/Llama-3-8B-SpclSpclSpcl_NaiveCompletion-ise \
--num_train_epochs 3 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 8 \
--evaluation_strategy "no" --save_strategy "epoch" \
--learning_rate 2e-4 --weight_decay 0. --warmup_ratio 0.03 \
--lr_scheduler_type "cosine" --logging_steps 1 --tf32 True \
--attack SpclSpclSpcl_NaiveCompletion \
--model_max_length 512 \
--bf16 True --dataloader_num_workers 4 \
--extra_config_path training/config/Llama-3-8B-SpclSpclSpcl_NaiveCompletion-ise.json
```

## Train InstSep
```bash
python -m torch.distributed.run --nproc_per_node=4 --master_port=29951 train_instsep.py \
--model_name_or_path meta-llama/Meta-Llama-3-8B \
--data_path datasets/alpaca_data_cleaned.json \
--output_dir meta-llama/Llama-3-8B-SpclSpclSpcl_NaiveCompletion-instsep \
--num_train_epochs 3 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 8 \
--evaluation_strategy "no" \
--save_strategy "epoch" \
--learning_rate 2e-4 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--attack SpclSpclSpcl_NaiveCompletion \
--model_max_length 512 \
--bf16 True \
--dataloader_num_workers 4 \
--extra_config_path training/config/Llama-3-8B-SpclSpclSpcl_NaiveCompletion-instsep.json
```

## Adv Train InstFuse
```bash
python -m torch.distributed.run --nproc_per_node=4 --master_port=29951 train_instfuse.py \
--model_name_or_path meta-llama/Meta-Llama-3-8B \
--data_path datasets/alpaca_data_cleaned.json \
--output_dir meta-llama/Llama-3-8B-SpclSpclSpcl_NaiveCompletion-instfuse \
--num_train_epochs 3 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 8 \
--evaluation_strategy "no" \
--save_strategy "epoch" \
--learning_rate 2e-4 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--attack SpclSpclSpcl_NaiveCompletion \
--model_max_length 512 \
--bf16 True \
--dataloader_num_workers 4
```

# Evaluation
## StruQ
```bash
python -m testing.test --model_name_or_path meta-llama/Llama-3-8B-SpclSpclSpcl_NaiveCompletion-struq \
--attack none naive ignore completion_real escape_separation
```

## ISE
```bash
python -m testing.test --model_name_or_path meta-llama/Llama-3-8B-SpclSpclSpcl_NaiveCompletion-ise \
--attack none naive ignore completion_real escape_separation \
--pass_expert_labels \
--customized_model_class "LlamaForCausalLMMoE"
```