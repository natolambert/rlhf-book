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
import time

import torch
import torch.optim as optim
from rich.console import Console
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

import wandb

from .buffer import Experience, join_experiences_batch
from .config import Config, load_config
from .rollout import TransformerRolloutEngine
from .utils import (
    compute_log_probs,
    compute_values,
    create_dataset,
    get_loss_objective,
    get_ref_model,
    get_val_model,
    load_model,
    print_model_info,
    print_step_header,
    progress_bar,
    seed_everything,
)


def main(cfg: Config):
    seed_everything(cfg.seed)
    console = Console()

    cpu_device = torch.device("cpu")
    if torch.cuda.is_available():
        model_device = torch.device(f"cuda:{cfg.model_device_id}")
        ref_model_device = torch.device(f"cuda:{cfg.ref_model_device_id}")
        val_model_device = torch.device(f"cuda:{cfg.val_model_device_id}")
    else:
        model_device = ref_model_device = val_model_device = cpu_device

    dataset = create_dataset(cfg)
    model, tokenizer = load_model(cfg.model_name, model_device)
    ref_model = get_ref_model(cfg.model_name, ref_model_device, cfg.beta)
    val_model = get_val_model(cfg.model_name, val_model_device, cfg.loss)
    rollout_engine = TransformerRolloutEngine(
        cfg=cfg,
        dataset=dataset,
        model=model,
        tokenizer=tokenizer,
        ref_model=ref_model,
        val_model=val_model,
        cpu_device=cpu_device,
        console=console,
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

    # wandb project can be set via env var WANDB_PROJECT or config file
    wandb_project = os.environ.get("WANDB_PROJECT", cfg.wandb_project)
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", cfg.wandb_run_name)

    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=wandb_project, name=wandb_run_name, config=vars(cfg))
    print_model_info(console, model)

    start_time = time.time()
    for step, replay_buffer in enumerate(rollout_engine):
        print_step_header(console, step=step, total=len(rollout_engine))
        model.eval()
        if val_model:
            val_model.eval()

        rewards = torch.stack([e.rewards for e in replay_buffer.buffer])
        avg_reward = rewards.mean().item()
        hours = (time.time() - start_time) / 3600
        wandb.log({"avg_reward": avg_reward, "hours": hours})

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

            for idx, exp in enumerate(experience_sampler):
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
                if (idx + 1) % cfg.batch_acc == 0 or (idx + 1) == len(experience_sampler):
                    grad_norm = clip_grad_norm_(params, max_norm=cfg.max_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()

                    num_accumulated = min(cfg.batch_acc, (idx % cfg.batch_acc) + 1)
                    avg_loss = accumulated_loss / num_accumulated
                    hours = (time.time() - start_time) / 3600
                    wandb.log({"loss": avg_loss, "grad_norm": grad_norm, "hours": hours})
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
