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
import time
from typing import Any

import numpy as np
import reasoning_gym as rg
import torch
import torch.nn as nn
import torch.optim as optim
from reasoning_gym.composite import DatasetSpec
from reasoning_gym.dataset import ProceduralDataset
from rich.console import Console
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

from .algorithms import compute_log_probs, compute_values
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
)
from .rollout import TransformerRolloutEngine, collect_rollouts_for_step
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
    """Load value model."""
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


def create_dataset(cfg: Config) -> ProceduralDataset:
    """Create the training dataset from config."""
    specs = [DatasetSpec(name=s.name, weight=s.weight, config=s.config) for s in cfg.data.specs]
    return rg.create_dataset("composite", size=cfg.data.size, seed=cfg.seed, datasets=specs)


def main(cfg: Config):
    seed_everything(cfg.seed)
    console = Console()

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
    model, tokenizer = load_model(cfg.model_name, model_device)
    ref_model = get_ref_model(cfg.model_name, ref_model_device, cfg.beta)
    val_model = get_val_model(cfg.model_name, val_model_device, cfg.loss)
    rollout_engine = TransformerRolloutEngine(
        tokenizer=tokenizer,
        cfg=cfg,
        ref_model=ref_model,
        val_model=val_model,
        cpu_device=cpu_device,
    )
    objective = get_loss_objective(
        loss=cfg.loss,
        clip_eps_lo=cfg.clip_eps_lo,
        clip_eps_hi=cfg.clip_eps_hi,
        clip_eps_val=cfg.clip_eps_val,
        vf_coef=cfg.vf_coef,
        beta=cfg.beta,
        kl_estimator=cfg.kl_estimator,
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
    dataloader_iter = iter(dataloader)
    for step in range(len(dataloader)):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            break
        print_step_header(console, step=step, total=len(dataloader))
        model.eval()
        if val_model:
            val_model.eval()
        replay_buffer.clear()
        rollout_rewards, rollout_completions = [], []

        with progress_bar(console) as progress:
            entries = [entry for entry in batch for _ in range(cfg.num_rollouts)]
            task = progress.add_task("Generating rollouts", total=1)
            replay_buffer = collect_rollouts_for_step(
                model=model,
                entries=entries,
                dataset=dataset,
                dataloader_iter=dataloader_iter,
                rollout_engine=rollout_engine,
                console=console,
                replay_buffer=replay_buffer,
                rollout_rewards=rollout_rewards,
                rollout_completions=rollout_completions,
            )
            progress.update(task, completed=1)

        if len(replay_buffer) == 0:
            console.print("[bold red]No valid experiences collected; skipping step.[/bold red]")
            continue

        # Summarize rollouts
        avg_reward = torch.cat(rollout_rewards, dim=0).mean().item()
        hours_elapsed = (time.time() - start_time) / 3600
        wandb.log({"avg_reward": avg_reward, "hours_elapsed": hours_elapsed})
        if rollout_completions:
            print_rollout_sample(
                console, reward=avg_reward, rollout_completions=rollout_completions
            )
        else:
            console.print(f"[bold green]Average Reward:[/bold green] {avg_reward:.4f}")

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

            for batch_idx, exp in enumerate(experience_sampler):
                exp: Experience
                exp = exp.to(model.device)

                log_probs = compute_log_probs(model, exp.sequence_ids, exp.attention_mask)
                values = compute_values(val_model, exp.sequence_ids, exp.attention_mask)
                loss = objective(log_probs=log_probs, experience=exp, values=values)
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
