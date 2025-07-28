import transformers
from train import smart_tokenizer_and_embedding_resize, ModelArguments, DataArguments, TrainingArguments, AttackArguments, find_closest_match
from config import DEFAULT_TOKENS, SPECIAL_DELM_TOKENS, DELIMITERS
from transformers import Trainer
from transformers.utils import logging
from modeling.llama_instsep import LlamaForCausalLMMoE, LlamaMoEConfig, LlamaForCausalLMMoEV2
from data_generation.struq import jload, make_supervised_data_module

logger = logging.get_logger(__name__)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, AttackArguments))
    model_args, data_args, training_args, attack_args = parser.parse_args_into_dataclasses()
    data_args.attack = attack_args.attack 
    if 'Instruct' in model_args.model_name_or_path:
        assert 'SpclSpclSpcl' not in data_args.attack
    print('\n\n' + training_args.output_dir + '\n\n')

    config = LlamaMoEConfig.from_pretrained(
        model_args.model_name_or_path,
    )

    model = LlamaForCausalLMMoEV2.from_pretrained(
        model_args.model_name_or_path,
        ignore_mismatched_sizes=True,
        config=config,
    )

    if model_args.window_size > 0:
        model.config.window = model_args.window_size

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=True,
        batched=True
    )

    # add new tokens into the dictionary => update model embedding layer
    special_tokens_dict = {"pad_token": DEFAULT_TOKENS['pad_token'],
                           "eos_token": DEFAULT_TOKENS['eos_token'],
                           "bos_token": DEFAULT_TOKENS['bos_token'],
                           "unk_token": DEFAULT_TOKENS['unk_token'],
                           "additional_special_tokens": SPECIAL_DELM_TOKENS}

    smart_tokenizer_and_embedding_resize(special_tokens_dict=special_tokens_dict, tokenizer=tokenizer, model=model)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"✅ {name} is trainable")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Params: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")

    # Construct dataloader
    frontend_delimiters = 'SpclSpclSpcl' if 'SpclSpclSpcl' in data_args.attack else find_closest_match(model_args.model_name_or_path, DELIMITERS)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, frontend_delimiters=frontend_delimiters, downsample=training_args.downsample)
    if not training_args.downsample and training_args.lr_scale:
        training_args.learning_rate /= data_module["train_dataset"].data_copy_count

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    config = trainer.model.config  # Access the config of the model
    config.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()
