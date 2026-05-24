import argparse
import os
import time

import torch
import torch.optim as optim
import wandb
from torch.nn.utils import clip_grad_norm_

from .config import Config, load_config
from .data import build_dataloader
from .rollout import generate_batch
from .utils import (
    get_loss_objective,
    load_model,
    print_model_info,
    print_step_header,
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
    loader = build_dataloader(cfg.split, batch_size=cfg.prompts_per_step, shuffle=True)
    objective = get_loss_objective(cfg.loss, kl_top_k=cfg.kl_top_k).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

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
        batches = [generate_batch(model, tokenizer, entry, cfg) for entry in prompts]

        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for b in batches:
            loss = objective(model, b) / len(batches)
            if loss.isfinite():
                loss.backward()
                accumulated_loss += loss.item()

        grad_norm = clip_grad_norm_(model.parameters(), cfg.max_norm)
        optimizer.step()
        torch.cuda.empty_cache()

        wandb.log(
            {
                "loss": accumulated_loss,
                "grad_norm": float(grad_norm),
                "avg_reward": torch.cat([b["reward"] for b in batches]).mean().item(),
                "avg_acc": torch.cat([b["acc"] for b in batches]).mean().item(),
                "hours": (time.time() - start_time) / 3600,
            }
        )
        step += 1


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(load_config(args.config))


if __name__ == "__main__":
    main_cli()
