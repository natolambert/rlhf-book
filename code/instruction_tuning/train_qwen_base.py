import argparse
import os
import time

import torch
import wandb
from torch.nn.utils import clip_grad_norm_

from .config import Config, load_config
from .utils import (
    compute_loss,
    console,
    create_dataloader,
    generate_samples,
    load_model,
    make_lr_scheduler,
    print_epoch_header,
    print_training_info,
    progress_bar,
    seed_everything,
)


def main(cfg: Config):
    seed_everything(cfg.seed)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cfg.model_device_id}")
    else:
        device = torch.device("cpu")

    # Minimal ChatML. Note the Jinja `{%-`/`-%}` markers strip the adjacent literal
    # '\n's, so this renders as '...<|im_end|><|im_start|>...' with the generation
    # prompt ending at 'assistant' (no trailing newline) and responses learning to
    # start with '\n'. That is deliberate: the bare word 'assistant' is a BPE-safe
    # mask boundary, whereas a trailing '\n' can merge with a response's leading
    # newline ('\n'+'\n' -> one '\n\n' token) and silently misalign the labels.
    chat_template = (
        "{%- for message in messages -%}"
        "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}"
    )

    model, tokenizer = load_model(
        cfg,
        device,
        chat_template=chat_template,
        eos_token="<|im_end|>",
        generation_eos_ids=[151643, 151645],  # <|endoftext|>, <|im_end|>
    )

    # Warm-start the stop token. <|im_end|> is untrained in the base model (embedding
    # row norm ~0.38 vs ~1.6 typical) and at lr 5e-6 SFT cannot train it from scratch:
    # the first run of this script ended with P(<|im_end|>)~1e-4 at supervised stop
    # positions while actively suppressing the base's natural <|endoftext|> stop.
    # Copying the well-trained document-EOS row means SFT only re-purposes an existing
    # direction (embeddings are tied, so this also fixes the output logit).
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        emb[151645] = emb[151643]  # <|im_end|> := <|endoftext|>
    dataloader = create_dataloader(cfg, tokenizer)

    accum = cfg.gradient_accumulation_steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = len(dataloader) // accum
    total_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = make_lr_scheduler(optimizer, total_steps, cfg.warmup_ratio)

    wandb_project = os.environ.get("WANDB_PROJECT", cfg.wandb_project)
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", cfg.wandb_run_name)
    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=wandb_project, name=wandb_run_name, config=vars(cfg))

    print_training_info(model, cfg, total_steps, warmup_steps)

    start_time = time.time()
    global_step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    micro_in_step = 0

    for epoch in range(cfg.num_epochs):
        print_epoch_header(epoch, cfg.num_epochs)
        with progress_bar() as progress:
            task = progress.add_task("Training", total=len(dataloader))

            for batch_idx, batch in enumerate(dataloader):
                batch = batch.to(device)
                loss = compute_loss(model, batch)
                if loss.isfinite():
                    scaled = loss / accum
                    scaled.backward()
                    accumulated_loss += loss.item()
                    micro_in_step += 1

                if (batch_idx + 1) % accum == 0:
                    if cfg.sample_every > 0 and global_step % cfg.sample_every == 0:
                        generate_samples(model, tokenizer, cfg, step=global_step)

                    grad_norm = clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    avg_loss = accumulated_loss / max(1, micro_in_step)
                    wandb.log(
                        {
                            "loss": avg_loss,
                            "grad_norm": float(grad_norm),
                            "learning_rate": scheduler.get_last_lr()[0],
                            "epoch": epoch + (batch_idx + 1) / len(dataloader),
                            "hours": (time.time() - start_time) / 3600,
                        },
                        step=global_step,
                    )
                    progress.update(task, advance=1, description=f"[dim]Loss: {avg_loss:.4f}[/dim]")
                    accumulated_loss = 0.0
                    micro_in_step = 0
                else:
                    progress.update(task, advance=1)

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        micro_in_step = 0

    # Final samples from the trained model, then persist it (the saved tokenizer
    # carries the custom chat template and eos so the checkpoint is self-describing).
    if cfg.sample_every > 0:
        generate_samples(model, tokenizer, cfg, step=global_step)
    if cfg.output_dir:
        model.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        console.print(f"[bold green]Saved model + tokenizer to {cfg.output_dir}[/bold green]")

    wandb.finish()


def main_cli():
    parser = argparse.ArgumentParser(description="Instruction-tune a base model (SFT).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(load_config(args.config))


if __name__ == "__main__":
    main_cli()
