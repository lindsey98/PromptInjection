import transformers
import torch
from train import smart_tokenizer_and_embedding_resize, ModelArguments, DataArguments, TrainingArguments, AttackArguments
from transformers.utils import logging
from modeling import MistralForCausalLMFuse, MistralFuseConfig
from peft import LoraConfig, get_peft_model
import os
from data_generation.struq import jload, make_supervised_data_module
from train_instfuse_lora import InstFuseTrainer
logger = logging.get_logger(__name__)

os.environ["WANDB_WATCH"]="all" # log all parameters gradients

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, AttackArguments))
    model_args, data_args, training_args, attack_args = parser.parse_args_into_dataclasses()
    data_args.attack = attack_args.attack 
    print('\n\n' + training_args.output_dir + '\n\n')

    config = MistralFuseConfig.from_pretrained(
        model_args.model_name_or_path,
    )

    config.residual = True
    config.bit_flip = True

    model = MistralForCausalLMFuse.from_pretrained(
        model_args.model_name_or_path,
        ignore_mismatched_sizes=True,
        config=config,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=True,
        batched=True
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
        r=16,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        modules_to_save=["lm_head", "embed_tokens", "deinstruction_shift"],
    )
    # Create the PEFT model
    peft_model = get_peft_model(model, lora_config)

    # Construct dataloader
    frontend_delimiters = data_args.attack.split("_")[0]
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              downsample=training_args.downsample,
                                              frontend_delimiters=frontend_delimiters)
    if not training_args.downsample and training_args.lr_scale:
        training_args.learning_rate /= data_module["train_dataset"].data_copy_count

    trainer = InstFuseTrainer(
        model=peft_model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )
    trainer.model.print_trainable_parameters()
    trainer.train()

    # Merge and unload LoRA
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(training_args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(training_args.output_dir)

    config = trainer.model.config  # Access the config of the model
    config.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()
