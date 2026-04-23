# Policy Gradient Training Loop
#
# Original implementation by Zafir Stojanovski (@zafstojano)
# Source: https://github.com/zafstojano/policy-gradients
# License: Apache 2.0
#
# Adapted for RLHF Book (https://rlhfbook.com) by Nathan Lambert
# - Added SDPA fallback for platforms without flash-attn (e.g., DGX Spark)

import argparse
import os
import platform
import random
import re
import time
from itertools import batched  # Requires Python 3.12+
from typing import Any, NamedTuple

import numpy as np
import reasoning_gym as rg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from reasoning_gym.composite import DatasetSpec
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer
from rich.console import Console
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import wandb

from .buffer import Experience, ReplayBuffer, join_experiences_batch
from .config import Config, load_config
from .loss import (
    CISPOLoss,
    DAPOLoss,
    GRPOLoss,
    GSPOLoss,
    PPOLoss,
    ReinforceLoss,
    SAPOLoss,
    approx_kl3,
    masked_mean,
)
from .utils import print_model_info, print_rollout_sample, print_step_header, progress_bar


def get_attn_implementation() -> str:
    """Determine the best attention implementation for this platform.

    Returns 'flash_attention_2' on x86_64 with flash-attn installed,
    otherwise 'sdpa' (PyTorch's native SDPA, faster on DGX Spark/Blackwell).
    """
    if platform.machine() != "x86_64":
        return "sdpa"  # aarch64 / DGX Spark - use SDPA with cuDNN

    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model_name: str, device_map: Any, gradient_checkpointing: bool = True):
    """Load model and tokenizer with automatic attention implementation selection."""
    attn_impl = get_attn_implementation()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    # Many decoder-only models (LLaMA, GPT-2) don't define pad_token
    # Set it to eos_token to enable batch padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        trust_remote_code=False,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, tokenizer


def get_ref_model(model_name: str, device_map: Any, beta: float):
    """Load reference model for KL penalty (only if beta > 0)."""
    if not beta:
        return None
    ref_model, _ = load_model(model_name, device_map, gradient_checkpointing=False)
    ref_model.eval()
    return ref_model


def get_val_model(model_name: str, device_map: Any, loss: str, gradient_checkpointing: bool = True):
    """Load value model for PPO (only if loss == 'ppo')."""
    if loss not in ["ppo"]:
        return None
    val_model, _ = load_model(model_name, device_map, gradient_checkpointing)
    val_model.lm_head = nn.Linear(
        val_model.lm_head.in_features, 1, bias=False, device=val_model.device, dtype=torch.bfloat16
    )
    return val_model


def get_loss_objective(loss: str, **kwargs) -> nn.Module:
    """Get the loss function module for the specified algorithm."""
    if loss in ["grpo", "drgrpo"]:
        return GRPOLoss(**kwargs)
    elif loss == "gspo":
        return GSPOLoss(**kwargs)
    elif loss in ["rloo", "reinforce"]:
        return ReinforceLoss(**kwargs)
    elif loss == "cispo":
        return CISPOLoss(**kwargs)
    elif loss == "sapo":
        return SAPOLoss(**kwargs)
    elif loss == "ppo":
        return PPOLoss(**kwargs)
    elif loss == "dapo":
        return DAPOLoss(**kwargs)
    raise ValueError(f"Unsupported loss type: {loss}")


def _accuracy_reward(
    dataset: ProceduralDataset,
    completions: list[str],
    entries: list[dict],
    cfg: Config,
    lengths: list[int],
) -> list[float]:
    """Compute accuracy reward based on extracted answers.

    For DAPO, apply an overlong penalty based on completion length:
    - no penalty when length <= l_max - l_cache
    - linear penalty in the buffer region
    - full penalty of -1 when length > l_max
    """

    def dapo_length_penalty(completion_len: int, l_cache: int, l_max: int) -> float:
        safe_len = l_max - l_cache
        if completion_len <= safe_len:
            return 0.0
        if completion_len <= l_max:
            return (safe_len - completion_len) / l_cache
        return -1.0

    def score_answer(completion: str, entry: dict, completion_len: int) -> float:
        answer = extract_answer(completion)
        score = float(dataset.score_answer(answer, entry))

        if cfg.loss == "dapo":
            score += dapo_length_penalty(
                completion_len=completion_len,
                l_cache=cfg.l_cache,
                l_max=cfg.l_max,
            )

        return score

    return [
        score_answer(
            completion=completion,
            entry=entry,
            completion_len=completion_len,
        )
        for completion, entry, completion_len in zip(completions, entries, lengths, strict=True)
    ]


def _format_reward(completions: list[str], **kwargs) -> list[float]:
    """Compute format reward based on presence of thinking/answer tags."""

    def count_tags(text: str) -> float:
        count = 0.0
        if re.search(r"\s*<think>\s*", text):
            count += 0.25
        if re.search(r"\s*</think>\s*", text):
            count += 0.25
        if re.search(r"\s*<answer>\s*", text):
            count += 0.25
        if re.search(r"\s*</answer>\s*", text):
            count += 0.25
        return count

    return [count_tags(c) for c in completions]


def compute_rewards(
    dataset: ProceduralDataset,
    completions: list[str],
    entries: list[dict],
    cfg: Config,
    lengths: list[int],
    format_weight: float = 0.5,
) -> list[float]:
    """Compute combined accuracy + format rewards."""
    accuracy_rewards = _accuracy_reward(dataset, completions, entries, cfg, lengths)
    format_rewards = _format_reward(completions)
    combined_rewards = [
        acc + format_weight * fmt for acc, fmt in zip(accuracy_rewards, format_rewards, strict=True)
    ]
    return combined_rewards


def apply_reward_kl(
    rewards: torch.Tensor,
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: torch.Tensor,
    beta: float,
    loss: str,
) -> torch.Tensor:
    """Apply KL penalty to rewards (for REINFORCE/RLOO/PPO)."""
    if not beta or loss not in ["ppo", "rloo", "reinforce"]:
        return rewards
    kl_div = approx_kl3(log_probs, log_probs_ref, action_mask)
    kl_div = masked_mean(kl_div, mask=action_mask, dim=-1, keepdim=True)
    rewards = rewards - beta * kl_div
    return rewards


def compute_standardized_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute standardized advantages (GRPO, GSPO, CISPO, DAPO)"""
    return (rewards - rewards.mean(dim=0, keepdim=True)) / (rewards.std(dim=0, keepdim=True) + eps)


def compute_nonstandardized_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Compute non-standardized advantages (Dr. GRPO)."""
    return rewards - rewards.mean(dim=0, keepdim=True)


def compute_loo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Compute leave-one-out advantages (RLOO)."""
    K = rewards.shape[0]
    return (K / (K - 1)) * (rewards - rewards.mean(dim=0, keepdim=True))


def compute_gae(
    rewards: torch.Tensor, action_mask: torch.Tensor, values: torch.Tensor, gamma: float, lam: float
) -> torch.Tensor:
    """Compute Generalized Advantage Estimation (PPO)."""
    B, S = action_mask.size()
    device = action_mask.device
    last_action_indices = action_mask.long().cumsum(dim=-1).argmax(dim=-1, keepdim=True)
    indices = torch.arange(S, device=device).unsqueeze(0)
    done = (indices >= last_action_indices).float()

    rewards = torch.zeros_like(action_mask, device=device, dtype=torch.float32).scatter_(
        dim=-1, index=last_action_indices, src=rewards
    )

    values = values.to(device)
    advantages = torch.zeros_like(action_mask, dtype=torch.float32, device=device)
    next_values = torch.zeros(B, device=device, dtype=torch.float32)
    running = torch.zeros(B, device=device, dtype=torch.float32)

    for t in reversed(range(S)):
        not_done = 1.0 - done[:, t]
        delta = rewards[:, t] + not_done * gamma * next_values - values[:, t]
        running = delta + not_done * gamma * lam * running
        advantages[:, t] = running
        next_values = values[:, t]

    advantages = advantages * action_mask
    return advantages


def compute_advantages(
    rewards: torch.Tensor,
    loss: str,
    action_mask: torch.Tensor | None = None,
    values: torch.Tensor | None = None,
    gamma: float | None = None,
    lam: float | None = None,
) -> torch.Tensor:
    """Compute advantages using the appropriate method for the loss function."""
    if loss in ["grpo", "gspo", "cispo", "sapo", "dapo"]:
        return compute_standardized_advantages(rewards)
    elif loss in ["drgrpo"]:
        return compute_nonstandardized_advantages(rewards)
    elif loss in ["rloo"]:
        return compute_loo_advantages(rewards)
    elif loss in ["ppo"]:
        return compute_gae(rewards, action_mask, values, gamma, lam)
    else:
        return rewards


def compute_log_probs(
    model, sequence_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Compute log probabilities for each token in the sequence."""
    if not model:
        return None
    sequence_ids, attention_mask = sequence_ids.to(model.device), attention_mask.to(model.device)
    output = model(input_ids=sequence_ids, attention_mask=attention_mask, use_cache=False)
    logits = output.logits[:, :-1, :].to(torch.float32)
    log_probs = F.log_softmax(logits, dim=-1)
    targets = sequence_ids[:, 1:].unsqueeze(-1)
    target_log_probs = torch.gather(log_probs, dim=-1, index=targets).squeeze(-1)
    return target_log_probs


def compute_values(model, sequence_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute value estimates for each position (PPO)."""
    if not model:
        return None
    sequence_ids, attention_mask = sequence_ids.to(model.device), attention_mask.to(model.device)
    output = model(input_ids=sequence_ids, attention_mask=attention_mask, use_cache=False)
    values = output.logits[:, :-1, :].squeeze(-1).to(torch.float32)
    return values


class RolloutOutput(NamedTuple):
    sequence_ids: torch.Tensor  # [B, T]      (LongTensor)
    action_mask: torch.Tensor  # [B, T-1]    (BoolTensor)
    attention_mask: torch.Tensor  # [B, T]      (BoolTensor)
    rewards: torch.Tensor  # [B, 1]      (FloatTensor)
    completions: list[str]  # length B


def rollout(
    model,
    entries: list[dict],
    dataset: ProceduralDataset,
    tokenizer: AutoTokenizer,
    cfg: Config,
    console: Console,
) -> RolloutOutput | None:
    """Generate completions and compute rewards."""
    # 1. Format prompts
    message_templates = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPTS["DeepSeekZero"]},
                {"role": "user", "content": entry["question"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for entry in entries
    ]
    model_inputs = tokenizer(
        message_templates,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to(model.device)

    # 2. Generate responses
    pad_token_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    generation_config = GenerationConfig(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        min_p=cfg.min_p,
        do_sample=True,
        max_new_tokens=cfg.max_new_tokens,
        pad_token_id=pad_token_id,
    )
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
    completion_ids = sequence_ids[:, model_inputs["input_ids"].shape[1] :]
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    # 3. Obtain the generated tokens only
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, model_inputs["input_ids"].shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]
    # Per-completion generated length
    lengths = action_mask.sum(dim=1).tolist()
    # 4. Compute rewards
    rewards_list = compute_rewards(dataset, completions, entries, cfg, lengths)
    rewards = torch.tensor(rewards_list, dtype=torch.float32, device=model.device).unsqueeze(-1)

    # 5. Compute attention mask
    attention_mask = sequence_ids != tokenizer.pad_token_id
    # 6. DAPO group filtering: drop groups with all rewards 0 or all rewards 1
    if cfg.loss == "dapo":
        group_size = cfg.num_rollouts
        batch_size = rewards.shape[0]

        if batch_size % group_size != 0:
            raise ValueError(
                f"Batch size {batch_size} is not divisible by num_rollouts={group_size}."
            )

        num_groups = batch_size // group_size

        rewards_grouped = rewards.squeeze(-1).view(num_groups, group_size)
        sequence_ids_grouped = sequence_ids.view(num_groups, group_size, -1)
        action_mask_grouped = action_mask.view(num_groups, group_size, -1)
        attention_mask_grouped = attention_mask.view(num_groups, group_size, -1)

        all_zero = (rewards_grouped == 0).all(dim=1)
        all_one = (rewards_grouped == 1).all(dim=1)
        valid_groups = ~(all_zero | all_one)

        num_filtered = (~valid_groups).sum().item()
        console.print(
            f"[bold yellow]DAPO filtering:[/bold yellow] "
            f"filtered {num_filtered}/{num_groups} groups "
            f"({num_filtered / max(num_groups, 1):.2%})"
        )

        if not valid_groups.any():
            console.print("[bold red]All DAPO groups were filtered out.[/bold red]")
            return None

        sequence_ids = sequence_ids_grouped[valid_groups].reshape(-1, sequence_ids.shape[-1])
        action_mask = action_mask_grouped[valid_groups].reshape(-1, action_mask.shape[-1])
        attention_mask = attention_mask_grouped[valid_groups].reshape(-1, attention_mask.shape[-1])
        rewards = rewards_grouped[valid_groups].reshape(-1, 1)

        kept_group_indices = valid_groups.nonzero(as_tuple=False).squeeze(-1).tolist()
        filtered_completions: list[str] = []
        for group_idx in kept_group_indices:
            start = group_idx * group_size
            end = start + group_size
            filtered_completions.extend(completions[start:end])
        completions = filtered_completions
    return RolloutOutput(
        sequence_ids=sequence_ids,
        action_mask=action_mask,
        attention_mask=attention_mask,
        rewards=rewards,
        completions=completions,
    )


def create_dataset(cfg: Config) -> ProceduralDataset:
    """Create the training dataset from config."""
    specs = [DatasetSpec(name=s.name, weight=s.weight, config=s.config) for s in cfg.data.specs]
    return rg.create_dataset("composite", size=cfg.data.size, seed=cfg.seed, datasets=specs)


def main(cfg: Config):
    """Main training loop."""
    seed_everything(cfg.seed)
    console = Console()

    # Print attention implementation info
    attn_impl = get_attn_implementation()
    console.print(f"[dim]Using attention implementation: {attn_impl}[/dim]")

    cpu_device = torch.device("cpu")
    if torch.cuda.is_available():
        model_device = torch.device(f"cuda:{cfg.model_device_id}")
        ref_model_device = torch.device(f"cuda:{cfg.ref_model_device_id}")
        val_model_device = torch.device(f"cuda:{cfg.val_model_device_id}")
    else:
        model_device = ref_model_device = val_model_device = cpu_device

    dataset = create_dataset(cfg)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.prompts_per_step,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
        collate_fn=lambda x: x,
    )
    model, tokenizer = load_model(cfg.model_name, model_device, gradient_checkpointing=True)
    ref_model = get_ref_model(cfg.model_name, ref_model_device, cfg.beta)
    val_model = get_val_model(
        cfg.model_name, val_model_device, cfg.loss, gradient_checkpointing=True
    )
    objective = get_loss_objective(
        loss=cfg.loss,
        clip_eps_lo=cfg.clip_eps_lo,
        clip_eps_hi=cfg.clip_eps_hi,
        clip_eps_val=cfg.clip_eps_val,
        vf_coef=cfg.vf_coef,
        beta=cfg.beta,
        sapo_temp_pos=cfg.sapo_temp_pos,
        sapo_temp_neg=cfg.sapo_temp_neg,
    ).to(model.device)
    params = list(model.parameters()) + (list(val_model.parameters()) if val_model else [])
    optimizer = optim.Adam(params, lr=cfg.lr)
    replay_buffer = ReplayBuffer()

    # wandb project can be set via env var WANDB_PROJECT or config file
    wandb_project = os.environ.get("WANDB_PROJECT", cfg.wandb_project)
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", cfg.wandb_run_name)

    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=wandb_project, name=wandb_run_name, config=vars(cfg))
    print_model_info(console, model)

    start_time = time.time()
    for step, batch in enumerate(dataloader):
        print_step_header(console, step=step, total=len(dataloader))
        model.eval()
        if val_model:
            val_model.eval()
        replay_buffer.clear()
        rollout_rewards, rollout_completions = [], []

        with progress_bar(console) as progress:
            entries = [entry for entry in batch for _ in range(cfg.num_rollouts)]
            task = progress.add_task(
                "Generating rollouts", total=len(entries) // cfg.rollout_batch_size
            )

            for batch in batched(entries, cfg.rollout_batch_size):
                with torch.no_grad():
                    rollout_output = rollout(
                        model=model,
                        entries=batch,
                        dataset=dataset,
                        tokenizer=tokenizer,
                        cfg=cfg,
                        console=console,
                    )
                    if not rollout_output:
                        continue
                    rollout_rewards.append(rollout_output.rewards.cpu())
                    rollout_completions.extend(
                        [
                            (entry["question"], entry["answer"], completion)
                            for entry, completion in zip(
                                batch, rollout_output.completions, strict=True
                            )
                        ]
                    )

                    log_probs_old = compute_log_probs(
                        model, rollout_output.sequence_ids, rollout_output.attention_mask
                    )
                    log_probs_ref = compute_log_probs(
                        ref_model, rollout_output.sequence_ids, rollout_output.attention_mask
                    )
                    rewards = apply_reward_kl(
                        rollout_output.rewards,
                        log_probs_old,
                        log_probs_ref,
                        rollout_output.action_mask,
                        cfg.beta,
                        cfg.loss,
                    )
                    values_old = compute_values(
                        val_model, rollout_output.sequence_ids, rollout_output.attention_mask
                    )

                    advantages = compute_advantages(
                        rewards,
                        cfg.loss,
                        rollout_output.action_mask,
                        values_old,
                        cfg.gamma,
                        cfg.lam,
                    )

                    experience = Experience(
                        sequence_ids=rollout_output.sequence_ids,
                        attention_mask=rollout_output.attention_mask,
                        action_mask=rollout_output.action_mask,
                        advantages=advantages,
                        log_probs_old=log_probs_old,
                        log_probs_ref=log_probs_ref,
                        values_old=values_old,
                    ).to(cpu_device)
                    replay_buffer.add(experience)

                progress.update(task, advance=1)

        # Summarize rollouts
        avg_reward = torch.cat(rollout_rewards, dim=0).mean().item()
        hours_elapsed = (time.time() - start_time) / 3600
        wandb.log({"avg_reward": avg_reward, "hours_elapsed": hours_elapsed})
        print_rollout_sample(console, reward=avg_reward, rollout_completions=rollout_completions)

        torch.cuda.empty_cache()
        model.train()
        if val_model:
            val_model.train()

        experience_sampler = DataLoader(
            dataset=replay_buffer.buffer,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            collate_fn=join_experiences_batch,
        )

        with progress_bar(console) as progress:
            task = progress.add_task("Training", total=len(experience_sampler))

            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0

            for batch_idx, experience in enumerate(experience_sampler):
                experience: Experience
                experience = experience.to(model.device)

                # Compute loss
                log_probs = compute_log_probs(
                    model, experience.sequence_ids, experience.attention_mask
                )
                values = compute_values(
                    val_model, experience.sequence_ids, experience.attention_mask
                )
                loss = objective(log_probs=log_probs, experience=experience, values=values)
                if not loss.isfinite():
                    continue
                scaled_loss = loss / cfg.batch_acc
                scaled_loss.backward()
                accumulated_loss += loss.item()

                # Update weights every batch_acc steps
                if (batch_idx + 1) % cfg.batch_acc == 0 or (batch_idx + 1) == len(
                    experience_sampler
                ):
                    grad_norm = clip_grad_norm_(params, max_norm=cfg.max_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()

                    num_accumulated = min(cfg.batch_acc, (batch_idx % cfg.batch_acc) + 1)
                    avg_loss = accumulated_loss / num_accumulated
                    hours_elapsed = (time.time() - start_time) / 3600
                    wandb.log(
                        {"loss": avg_loss, "grad_norm": grad_norm, "hours_elapsed": hours_elapsed}
                    )
                    progress.update(task, advance=1, description=f"[dim]Loss: {avg_loss:.4f}[/dim]")
                    accumulated_loss = 0.0
                else:
                    progress.update(task, advance=1)


def main_cli():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train policy gradient models for RLHF")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)


if __name__ == "__main__":
    main_cli()
