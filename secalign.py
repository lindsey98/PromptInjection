from trl import DPOTrainer
import transformers
from config import PROMPT_FORMAT, DEFAULT_TOKENS, DELIMITERS, SPECIAL_DELM_TOKENS
from train import ModelArguments, DataArguments, AttackArguments, smart_tokenizer_and_embedding_resize, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

def align():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, AttackArguments))
    model_args, training_args, data_args, attack_args = parser.parse_args_into_dataclasses()

    train_dataset = load_dataset('json', data_files=data_args.data_path_list, split="train")
    print(f"Dataset length = {len(train_dataset)}")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    if model_args.window_size > 0:
        model.config.window = model_args.window_size

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=True,
        batched=True
    )

    special_tokens_dict = dict()
    special_tokens_dict["pad_token"] = DEFAULT_TOKENS['pad_token'] ###
    special_tokens_dict["eos_token"] = DEFAULT_TOKENS['eos_token']
    special_tokens_dict["bos_token"] = DEFAULT_TOKENS['bos_token']
    special_tokens_dict["unk_token"] = DEFAULT_TOKENS['unk_token']
    special_tokens_dict["additional_special_tokens"] = SPECIAL_DELM_TOKENS ###
    smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict, tokenizer=tokenizer, model=model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["lm_head", "embed_tokens"],
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    print(training_args.output_dir, '\n\n\n')

    trainer = DPOTrainer(
        peft_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    # Save the LoRA-adapted model
    trainer.model.save_pretrained(training_args.output_dir)

    tokenizer.save_pretrained(training_args.output_dir)

    config = trainer.model.config  # Access the config of the model
    config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    align()