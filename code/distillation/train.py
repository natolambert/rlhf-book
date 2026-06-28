import argparse
import os
import time

import torch
import torch.optim as optim
import wandb
from torch.nn.utils import clip_grad_norm_

from .config import Config, load_config
from .data import build_dataloader, create_dataset
from .rollout import generate_batch
from .utils import (
    get_loss_objective,
    load_model,
    print_model_info,
    print_step_header,
    print_step_metrics,
    seed_everything,
)


def main(cfg: Config):
    seed_everything(cfg.seed)
    device = (
        torch.device(f"cuda:{cfg.model_device_id}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model, tokenizer = load_model(cfg.model_name, device)
    dataset = create_dataset(cfg)
    loader = build_dataloader(dataset, batch_size=cfg.prompts_per_step, shuffle=True)
    objective = get_loss_objective(
        cfg.loss, kl_top_k=cfg.kl_top_k, rollout_chunk=cfg.rollout_chunk
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    num_warmup_steps = int(cfg.num_steps * cfg.warmup_ratio)

    def lr_lambda(step):
        # Linear warmup to the target LR, then held constant.
        if step < num_warmup_steps:
            # Start at 1/(num_warmup_steps + 1) so the first step still moves.
            return float(step + 1) / float(max(1, num_warmup_steps + 1))
        return 1.0

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    wandb_entity = os.environ.get("WANDB_ENTITY", cfg.wandb_entity)
    wandb_project = os.environ.get("WANDB_PROJECT", cfg.wandb_project)
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", cfg.wandb_run_name)
    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=wandb_run_name,
            config=vars(cfg),
        )
    print_model_info(model)

    start_time = time.time()
    step = 0
    for prompts in loader:
        if step >= cfg.num_steps:
            break
        print_step_header(step=step, total=cfg.num_steps)

        torch.cuda.empty_cache()
        model.eval()
        batches = [
            generate_batch(model, tokenizer, dataset, entry, cfg, idx=i, total=len(prompts))
            for i, entry in enumerate(prompts)
        ]
        batches = [b for b in batches if b is not None]
        if not batches:
            continue

        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for b in batches:
            accumulated_loss += objective.backward_loss(model, b, scale=1.0 / len(batches))

        grad_norm = clip_grad_norm_(model.parameters(), cfg.max_norm)
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()

        metrics = {
            "step": step,
            "loss": accumulated_loss,
            "grad_norm": float(grad_norm),
            "lr": scheduler.get_last_lr()[0],
            "avg_reward": torch.cat([b["reward"] for b in batches]).mean().item(),
            "hours": (time.time() - start_time) / 3600,
        }
        wandb.log(metrics)
        print_step_metrics(step, metrics)
        step += 1


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(load_config(args.config))


if __name__ == "__main__":
    main_cli()
