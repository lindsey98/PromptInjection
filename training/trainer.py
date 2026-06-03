import logging
import transformers
from transformers import Trainer
from transformers.trainer import PreTrainedModel, is_sagemaker_mp_enabled, Adafactor, is_bitsandbytes_available
from typing import Optional, Tuple, Any, List
from torch import nn
from trl import DPOTrainer
import trl
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from contextlib import contextmanager, nullcontext
from torch import autocast

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    window_size: int = field(default=0, metadata={"help": "Window size for the sliding window attention."})
    padding_side: str = field(default="right", metadata={"help": "Padding side for tokenization."})

@dataclass
class DataArguments:
    data_path_list: List[str] = field(
        default_factory=list,
        metadata={"help": "List of file paths to the training data.", "nargs": "+"}
    )

@dataclass
class AttackArguments:
    attack: str = field(default='TextTextText_None', metadata={"help": "Attack type for SFT/Align"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    downsample: Optional[bool] = field(default=True)
    lr_scale: Optional[bool] = field(default=True)
    beta: float = field(default=0.1)
    ref_model_init_kwargs: Optional[str] = field(default=None)
    precompute_ref_log_probs: Optional[bool] = field(default=False)
    report_to: Optional[str] = "wandb"
    resume_from_checkpoint: Optional[bool] = field(default=True)

@dataclass
class DPOArgsSecAlign(trl.DPOConfig):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    downsample: Optional[bool] = field(default=True)
    lr_scale: Optional[bool] = field(default=True)
    beta: float = field(default=0.1)
    ref_model_init_kwargs: Optional[str] = field(default=None)
    report_to: Optional[str] = "wandb"
    resume_from_checkpoint: bool = False


@dataclass
class DPOArgsDRIP(trl.DPOConfig):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    downsample: Optional[bool] = field(default=True)
    lr_scale: Optional[bool] = field(default=True)
    beta: float = field(default=0.1)
    ref_model_init_kwargs: Optional[str] = field(default=None)
    report_to: Optional[str] = "wandb"
    resume_from_checkpoint: bool = False
    remove_unused_columns: bool = field(default=False)

class DPOTrainerOurs(DPOTrainer):

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        return dataset

    @staticmethod
    def _seq_logps(model, ids, attention_mask, expert_labels, prompt_length, return_logits=False):
        out = model(
            input_ids=ids,
            attention_mask=attention_mask,
            expert_labels=expert_labels,
            use_cache=False
        )
        lp = torch.log_softmax(out.logits, dim=-1) # logprob
        lp  = lp[:, :-1, :]
        target = ids[:, 1:] # ground-truth IDs (next token)
        tok = torch.gather(lp, 2, target.unsqueeze(-1)).squeeze(-1) # gather the logprobs at ground-truth IDs
        B, Tm1 = tok.size()
        ar = torch.arange(Tm1, device=tok.device).unsqueeze(0).expand(B, -1)

        start = (prompt_length - 1).clamp_min(0).unsqueeze(1)
        resp_mask = (ar >= start).to(tok.dtype)
        mask = resp_mask * attention_mask[:, 1:].to(tok.dtype) # only compute loss at response part
        tok = tok * mask

        logp_sum = tok.sum(-1)  # shape [B]

        if return_logits:
            # mean over the response positions (before log-softmax)
            chosen_logits = (out.logits[:, :-1, :].max(-1).values * mask).sum(-1) / mask.sum(-1).clamp(min=1)
            return logp_sum, chosen_logits
        return logp_sum

    def _safe_gather(self, tensor):
        return self.accelerator.gather_for_metrics(
            tensor.detach().to(torch.float32).contiguous()
        ).mean().item()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        metrics = {}

        compute_loss_context_manager = (
            autocast(self.accelerator.device.type)
            if getattr(self, "_peft_has_been_casted_to_bf16", False) else nullcontext()
        )
        with compute_loss_context_manager:
            chosen_logps, chosen_mean_logits = self._seq_logps(
                self.model, inputs["chosen_input_ids"], inputs["chosen_attention_mask"],
                inputs["chosen_expert_labels"], inputs["prompt_lens"], return_logits=True)
            rejected_logps, rejected_mean_logits = self._seq_logps(
                self.model, inputs["rejected_input_ids"], inputs["rejected_attention_mask"],
                inputs["rejected_expert_labels"], inputs["prompt_lens"], return_logits=True)

            with torch.no_grad():
                if self.ref_model is not None:
                    ref_chosen_logps = self._seq_logps(self.ref_model, inputs["chosen_input_ids"],
                                                       inputs["chosen_attention_mask"], inputs["chosen_expert_labels"],
                                                       inputs["prompt_lens"])
                    ref_rejected_logps = self._seq_logps(self.ref_model, inputs["rejected_input_ids"],
                                                         inputs["rejected_attention_mask"],
                                                         inputs["rejected_expert_labels"], inputs["prompt_lens"])
                else:
                    with self.null_ref_context():
                        ref_chosen_logps = self._seq_logps(self.model, inputs["chosen_input_ids"],
                                                           inputs["chosen_attention_mask"],
                                                           inputs["chosen_expert_labels"], inputs["prompt_lens"])
                        ref_rejected_logps = self._seq_logps(self.model, inputs["rejected_input_ids"],
                                                             inputs["rejected_attention_mask"],
                                                             inputs["rejected_expert_labels"], inputs["prompt_lens"])

            logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logits = logratios - ref_logratios
            losses = (
                    - F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        loss = losses.mean()

        chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps).detach()
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if self.control.should_evaluate else ""
        metrics[f"{prefix}rewards/chosen"] = self._safe_gather(chosen_rewards.mean())
        metrics[f"{prefix}rewards/rejected"] = self._safe_gather(rejected_rewards.mean())
        metrics[f"{prefix}rewards/accuracies"] = self._safe_gather(reward_accuracies.mean())
        metrics[f"{prefix}rewards/margins"] = self._safe_gather((chosen_rewards - rejected_rewards).mean())
        metrics[f"{prefix}logps/chosen"] = self._safe_gather(chosen_logps.mean())
        metrics[f"{prefix}logps/rejected"] = self._safe_gather(rejected_logps.mean())
        metrics[f"{prefix}logits/chosen"] = self._safe_gather(chosen_mean_logits.mean())
        metrics[f"{prefix}logits/rejected"] = self._safe_gather(rejected_mean_logits.mean())

        self.store_metrics(metrics, train_eval="train" if not self.control.should_evaluate else "eval")

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # self.store_metrics(metrics, train_eval="train")
        loss = loss.to(torch.float32)
        if return_outputs:
            return loss, metrics

        return loss

    # def create_optimizer(self, special_params_list = ["deinstruction_shift"]):
    def create_optimizer(self, special_params_list=None):
        if special_params_list is None:
            special_params_list = []

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if any(kw in n for kw in special_params_list) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate * 10, ## move 10 times faster
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (not any(kw in n for kw in special_params_list)) and (n in decay_parameters) and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (not any(kw in n for kw in special_params_list)) and (n not in decay_parameters) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
            ]

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters,
                                           **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer


class DPOTrainerMOE(DPOTrainerOurs):
    """DPOTrainerOurs + MoE load-balancing aux_loss."""

    @staticmethod
    def _seq_logps_moe(model, ids, attention_mask, expert_labels, prompt_length,
                       return_logits=False, output_router_logits=False):
        out = model(
            input_ids=ids,
            attention_mask=attention_mask,
            expert_labels=expert_labels,
            output_router_logits=output_router_logits,
            use_cache=False,
        )
        lp = torch.log_softmax(out.logits, dim=-1)[:, :-1, :]
        target = ids[:, 1:]
        tok = torch.gather(lp, 2, target.unsqueeze(-1)).squeeze(-1)

        B, Tm1 = tok.size()
        ar = torch.arange(Tm1, device=tok.device).unsqueeze(0).expand(B, -1)
        start = (prompt_length - 1).clamp_min(0).unsqueeze(1)
        resp_mask = (ar >= start).to(tok.dtype)
        mask = resp_mask * attention_mask[:, 1:].to(tok.dtype)
        tok = tok * mask
        logp_sum = tok.sum(-1)

        aux_loss = getattr(out, "aux_loss", None) if output_router_logits else None

        if return_logits:
            chosen_logits = (out.logits[:, :-1, :].max(-1).values * mask).sum(-1) / mask.sum(-1).clamp(min=1)
            return logp_sum, chosen_logits, aux_loss
        return logp_sum, aux_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        metrics = {}

        ctx = (autocast(self.accelerator.device.type)
               if getattr(self, "_peft_has_been_casted_to_bf16", False) else nullcontext())

        with ctx:
            # Policy forward: chosen + rejected, capture aux_loss from both
            chosen_logps, chosen_mean_logits, chosen_aux = self._seq_logps_moe(
                self.model, inputs["chosen_input_ids"], inputs["chosen_attention_mask"],
                inputs["chosen_expert_labels"], inputs["prompt_lens"],
                return_logits=True, output_router_logits=True)

            rejected_logps, rejected_mean_logits, rejected_aux = self._seq_logps_moe(
                self.model, inputs["rejected_input_ids"], inputs["rejected_attention_mask"],
                inputs["rejected_expert_labels"], inputs["prompt_lens"],
                return_logits=True, output_router_logits=True)

            # Ref forward: NO aux_loss needed (ref is frozen)
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_chosen_logps, _ = self._seq_logps_moe(
                        self.ref_model, inputs["chosen_input_ids"],
                        inputs["chosen_attention_mask"], inputs["chosen_expert_labels"],
                        inputs["prompt_lens"], output_router_logits=False)
                    ref_rejected_logps, _ = self._seq_logps_moe(
                        self.ref_model, inputs["rejected_input_ids"],
                        inputs["rejected_attention_mask"], inputs["rejected_expert_labels"],
                        inputs["prompt_lens"], output_router_logits=False)
                else:
                    with self.null_ref_context():
                        ref_chosen_logps, _ = self._seq_logps_moe(
                            self.model, inputs["chosen_input_ids"],
                            inputs["chosen_attention_mask"], inputs["chosen_expert_labels"],
                            inputs["prompt_lens"], output_router_logits=False)
                        ref_rejected_logps, _ = self._seq_logps_moe(
                            self.model, inputs["rejected_input_ids"],
                            inputs["rejected_attention_mask"], inputs["rejected_expert_labels"],
                            inputs["prompt_lens"], output_router_logits=False)

            # DPO loss
            logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logits = logratios - ref_logratios
            dpo_losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                -F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
            dpo_loss = dpo_losses.mean()

            # MoE aux loss (average over chosen + rejected forward passes)
            aux_coef = getattr(self.model.config, "router_aux_loss_coef", 0.001)
            if chosen_aux is not None and rejected_aux is not None:
                aux_loss = 0.5 * (chosen_aux + rejected_aux)
                loss = dpo_loss + aux_coef * aux_loss.to(dpo_loss.device)
                metrics["aux_loss"] = self._safe_gather(aux_loss)
            else:
                loss = dpo_loss

        # Metrics
        chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps).detach()
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if self.control.should_evaluate else ""
        metrics[f"{prefix}rewards/chosen"]     = self._safe_gather(chosen_rewards.mean())
        metrics[f"{prefix}rewards/rejected"]   = self._safe_gather(rejected_rewards.mean())
        metrics[f"{prefix}rewards/accuracies"] = self._safe_gather(reward_accuracies.mean())
        metrics[f"{prefix}rewards/margins"]    = self._safe_gather((chosen_rewards - rejected_rewards).mean())
        metrics[f"{prefix}logps/chosen"]       = self._safe_gather(chosen_logps.mean())
        metrics[f"{prefix}logps/rejected"]     = self._safe_gather(rejected_logps.mean())
        metrics[f"{prefix}dpo_loss"]           = self._safe_gather(dpo_loss.detach())

        self.store_metrics(metrics, train_eval="train" if not self.control.should_evaluate else "eval")

        loss = loss.to(self.args.device).to(torch.float32)
        if return_outputs:
            return loss, metrics
        return loss


class DPOTrainerAIR(DPOTrainerOurs):
    def create_optimizer(self, special_params_list = ["intermediate_shifts"]):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if any(kw in n for kw in special_params_list) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate * 10, ## move 10 times faster
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (not any(kw in n for kw in special_params_list)) and (n in decay_parameters) and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (not any(kw in n for kw in special_params_list)) and (n not in decay_parameters) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
            ]

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters,
                                           **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

class SFTTrainerParams(Trainer):
    def create_optimizer(self, special_params_list: List[str]):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if any(kw in n for kw in special_params_list) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate * 10,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (not any(kw in n for kw in special_params_list)) and (n in decay_parameters) and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (not any(kw in n for kw in special_params_list)) and (n not in decay_parameters) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
            ]

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters,
                                           **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

class SFTTrainerOurs(SFTTrainerParams):

    def create_optimizer(self, special_params_list = ["deinstruction_shifts"]):
        return super().create_optimizer(special_params_list)

class SFTTrainerISE(SFTTrainerParams):
    def create_optimizer(self, special_params_list = ["input_shifts"]):
        return super().create_optimizer(special_params_list)



