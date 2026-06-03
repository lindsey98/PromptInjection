
from __future__ import annotations
import os

import argparse
import json
import os
import re
import random
from typing import Dict, List

import torch
import torch.distributed as dist
from tqdm import tqdm

from testing.test import load_full_model
from config import PROMPT_FORMAT, DELIMITERS, DEFAULT_SYSTEM_PROMPT
from testing.tap.utils import _inject, _generate
from testing.pismith.trainer import PISmithConfig, PISmithTrainer
from testing.pismith.utils import extract_injected_task, make_witness_judge

class SEPTrainer(PISmithTrainer):

    def train(self, samples: List[Dict]) -> None:
        cfg  = self.config
        start_step = 0
        shard = samples[self.rank::self.world_size] if self.is_ddp else samples

        start_step, _ = self._resume_state()
        samples_per_epoch = max(len(shard), 1)
        start_epoch = start_step // samples_per_epoch

        step = start_step
        if self.is_main and start_step > 0:
            print(f"[PISmith] Resumed: step={start_step}, "
                  f"samples/epoch={samples_per_epoch}, starting at epoch={start_epoch}")

        for epoch in range(start_epoch, cfg.num_epochs):
            random.shuffle(shard)
            pbar = tqdm(shard, desc=f"PISmith SEP epoch {epoch + 1}", disable=not self.is_main)

            for sample in pbar:
                target_inst   = sample["target_inst"]
                context       = sample["context"]
                injected_task = sample["injected_task"]
                witness       = sample.get("witness", "")

                # Per-sample judge: witness word must appear in response
                new_judge = make_witness_judge(witness)

                # Temporarily override judge_fn for this sample
                orig_judge    = self.judge_fn
                self.judge_fn = new_judge

                messages  = self._build_messages(target_inst, context, injected_task, witness)
                input_ids = self._tokenize_messages(messages)

                with torch.no_grad():
                    attack_prompts = self._generate_group(input_ids, cfg.group_size)

                rewards = self._compute_rewards(
                    attack_prompts=attack_prompts,
                    target_inst=target_inst,
                    context=context,
                    injected_task=injected_task,
                    witness=witness,
                )

                self.judge_fn = orig_judge   # restore
                if self.is_main and step % 10 == 0:
                    best_idx = rewards.index(max(rewards))
                    print(f"\n[Step {step}] mean_r={sum(rewards) / len(rewards):.3f}")
                    print(f"  injected_task: {injected_task[:100]}")
                    print(f"  best_injection (r={rewards[best_idx]:.1f}): {attack_prompts[best_idx][:300]}")

                mean_r = sum(rewards) / len(rewards)
                if self.is_main:
                    pbar.set_postfix(mean_reward=f"{mean_r:.3f}", step=step)

                self.attack_model.train()
                self.optimizer.zero_grad()
                loss = self._pismith_loss(input_ids, attack_prompts, rewards)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.attack_model.parameters(), 1.0)
                self.optimizer.step()
                torch.cuda.empty_cache()

                step += 1
                if self.is_main and step % cfg.save_steps == 0:
                    self.save(suffix=f"step{step}", step=step, epoch=epoch)

            if self.is_ddp:
                dist.barrier()

        if self.is_main:
            print(f"[PISmith SEP] Training done. Total steps: {step}")




# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name_or_path", type=str, nargs="+", required=True)
    parser.add_argument("--base_model_path", type=str, default=None,
                   help="Explicit base model path; required when adapter path "
                        "does not encode the base path via the usual suffix convention.")
    parser.add_argument("--customized_model_class",   type=str, default="")
    parser.add_argument("--train_data_path", type=str, default="./datasets/SEP_dataset.json")
    parser.add_argument("--max_train_samples", type=int, default=100)
    parser.add_argument("--attack_model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--attack_model_path", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output_dir",   type=str, default="./pismith_ckpt/sep")
    parser.add_argument("--group_size",   type=int,   default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--save_steps",   type=int,   default=50)
    parser.add_argument("--entropy_coef_max", type=float, default=0.01)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--inject_position", type=str, default="end")
    parser.add_argument("--target_max_new_tokens", type=int, default=512)
    parser.add_argument("--lora_r",       type=int,   default=16)
    parser.add_argument("--lora_alpha",   type=int,   default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, nargs="+")
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    args = parser.parse_args()
    args.model_name_or_path = args.model_name_or_path[0]

    # ── DDP init ──────────────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_ddp     = "LOCAL_RANK" in os.environ
    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        is_main = (dist.get_rank() == 0)
    else:
        is_main = True
    this_gpu = local_rank

    # ── Load target model (one copy per rank on its own GPU) ──────────────────
    print(f"[rank {local_rank}] Loading target model on GPU {this_gpu}")
    model, tokenizer, frontend_delimiters, _ = load_full_model(
        args.model_name_or_path,
        customized_model_class=args.customized_model_class,
        base_model_path=args.base_model_path,
        device_map={"": this_gpu},
    )

    def target_query_fn(
        target_inst: str,
        context: str,
        attack_prompt: str,
    ) -> str:

        injected_context = _inject(context, attack_prompt, position=args.inject_position)
        prompt = PROMPT_FORMAT[frontend_delimiters]["prompt_input"].format_map({
            "system": DEFAULT_SYSTEM_PROMPT,
            "instruction": target_inst,
            "input": injected_context,
        })
        return _generate(prompt,
                         model,
                         tokenizer,
                         frontend_delimiters,
                         max_new_tokens=args.target_max_new_tokens,
                         customized_model_class=args.customized_model_class
                         )

    # ── Build training samples from SEP ──────────────────────────────────────
    with open(args.train_data_path, "r") as f:
        raw = json.load(f)

    train_samples = []
    for sample in raw:
        witness = sample.get("witness", "").strip()
        probe = extract_injected_task(sample)
        if len(probe) == 0:
            continue
        if not witness:
            continue   # skip samples without a witness (can't measure success)
        train_samples.append({
            "target_inst":   sample["system_prompt_clean"],
            "context":       sample["prompt_clean"],
            "injected_task": extract_injected_task(sample),
            "witness":       witness,
        })
        if len(train_samples) >= args.max_train_samples:
            break

    if is_main:
        print(f"[PISmith train_sep] {len(train_samples)} SEP samples loaded")

    # ── Config & trainer ──────────────────────────────────────────────────────
    config = PISmithConfig(
        attack_model_name=args.attack_model_name,
        attack_model_path=args.attack_model_path,
        group_size=args.group_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        save_steps=args.save_steps,
        entropy_coef_max=args.entropy_coef_max,
        output_dir=args.output_dir,
        max_train_samples=args.max_train_samples,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        resume_from_checkpoint=args.resume_from_checkpoint,
        clip_eps=args.clip_eps,
    )

    # Placeholder judge — SEPTrainer overrides it per-sample during train()
    trainer = SEPTrainer(
        config=config,
        target_query_fn=target_query_fn,
        judge_fn=lambda response: False,
    )
    trainer.train(train_samples)
    trainer.save(suffix="final")

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()