# Stage 3: selection + SFT + GSM8k exact-match eval. Auto-runs preprocess on
# cache miss, so one command per config is enough:
#   uv run python -m rejection_sampling.train --config configs/top_per_prompt.yaml

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from datasets import load_dataset
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from policy_gradients.train import get_attn_implementation, load_model, seed_everything
from policy_gradients.utils import print_model_info, print_step_header, progress_bar

from . import preprocess
from .config import Config, load_config
from .selection import select
from .utils import (
    ACEMATH_SYSTEM_PROMPT,
    cuda_memory_gb,
    extract_gsm8k_answer,
    format_gsm8k_gold,
    free_memory,
)


class SFTDataset(Dataset):
    """Tokenise (prompt, completion) pairs for causal SFT, masking prompt tokens."""

    def __init__(self, pairs: list[tuple[str, str]], tokenizer, max_seq_length: int):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        question, completion = self.pairs[idx]

        prompt_messages = [
            {"role": "system", "content": ACEMATH_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt_str = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        full_messages = prompt_messages + [{"role": "assistant", "content": completion}]
        full_str = self.tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )

        prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False).input_ids
        full_ids = self.tokenizer(full_str, add_special_tokens=False).input_ids
        prompt_len = len(prompt_ids)

        # Response-priority left truncation: keep the completion, drop prompt tokens.
        if len(full_ids) > self.max_seq_length:
            trim = len(full_ids) - self.max_seq_length
            full_ids = full_ids[trim:]
            prompt_len = max(0, prompt_len - trim)

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[:prompt_len] = -100  # mask prompt tokens from the loss
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }


def sft_collate(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Right-pad variable-length SFT examples into a single batch tensor."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    def pad(t: torch.Tensor, value: int) -> torch.Tensor:
        return F.pad(t, (0, max_len - t.size(0)), value=value)

    return {
        "input_ids": torch.stack([pad(item["input_ids"], pad_token_id) for item in batch]),
        "labels": torch.stack([pad(item["labels"], -100) for item in batch]),
        "attention_mask": torch.stack([pad(item["attention_mask"], 0) for item in batch]),
    }


def sft(
    cfg: Config,
    model,
    tokenizer,
    selected_pairs: list[tuple[str, str]],
    console: Console,
    start_time: float,
) -> None:
    """Full-parameter causal-LM SFT on the selected rejection-sampling pairs."""
    dataset = SFTDataset(selected_pairs, tokenizer, cfg.max_seq_length)
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
        collate_fn=lambda b: sft_collate(b, pad_id),
    )

    # Plain Adam (not AdamW) to match policy_gradients/train.py.
    params = list(model.parameters())
    optimizer = optim.Adam(params, lr=cfg.lr)

    model.train()
    global_step = 0
    for epoch in range(cfg.num_epochs):
        print_step_header(console, step=epoch, total=cfg.num_epochs)
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        with progress_bar(console) as progress:
            task = progress.add_task("SFT", total=len(dataloader))

            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(model.device)
                labels = batch["labels"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                loss = outputs.loss
                if not torch.isfinite(loss):
                    progress.update(task, advance=1)
                    continue

                scaled_loss = loss / cfg.batch_acc
                scaled_loss.backward()
                accumulated_loss += loss.item()

                is_last_in_epoch = (batch_idx + 1) == len(dataloader)
                if (batch_idx + 1) % cfg.batch_acc == 0 or is_last_in_epoch:
                    grad_norm = clip_grad_norm_(params, max_norm=cfg.max_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()

                    num_accumulated = min(cfg.batch_acc, (batch_idx % cfg.batch_acc) + 1)
                    avg_loss = accumulated_loss / num_accumulated
                    hours_elapsed = (time.time() - start_time) / 3600
                    wandb.log(
                        {
                            "sft/loss": avg_loss,
                            "sft/grad_norm": grad_norm,
                            "sft/epoch": epoch,
                            "sft/step": global_step,
                            "hours_elapsed": hours_elapsed,
                        }
                    )
                    progress.update(task, advance=1, description=f"[dim]Loss: {avg_loss:.4f}[/dim]")
                    accumulated_loss = 0.0
                    global_step += 1
                else:
                    progress.update(task, advance=1)

    # Release Adam moment buffers (~2x model size) + gradients before eval,
    # otherwise they stay resident and steal VRAM from generation.
    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    del optimizer, params
    free_memory()
    console.print(f"[dim]VRAM after SFT cleanup: {cuda_memory_gb():.2f} GB[/dim]")


def _build_eval_prompt(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": ACEMATH_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def evaluate(cfg: Config, model, tokenizer, console: Console) -> float:
    """Greedy-decode exact-match eval on the GSM8k test split."""
    ds = load_dataset(cfg.data.name, cfg.data.subset, split=cfg.data.test_split)
    if cfg.data.max_test_samples is not None:
        ds = ds.select(range(min(cfg.data.max_test_samples, len(ds))))

    eval_rows = [
        {"question": row["question"], "gold": format_gsm8k_gold(row["answer"])} for row in ds
    ]

    model.eval()
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )

    correct = 0
    total = 0
    samples: list[tuple[str, str, str, str]] = []  # (question, gold, prediction, completion)

    batch_size = cfg.eval_batch_size
    with progress_bar(console) as progress:
        total_batches = (len(eval_rows) + batch_size - 1) // batch_size
        task = progress.add_task("Evaluating", total=total_batches)
        for start in range(0, len(eval_rows), batch_size):
            batch = eval_rows[start : start + batch_size]
            chat_strings = [_build_eval_prompt(row["question"], tokenizer) for row in batch]
            inputs = tokenizer(
                chat_strings,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                return_attention_mask=True,
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=cfg.eval_max_new_tokens,
                    do_sample=False,
                    temperature=cfg.eval_temperature,
                    pad_token_id=pad_id,
                )

            prompt_len = inputs["input_ids"].shape[1]
            completion_ids = output_ids[:, prompt_len:]
            completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

            for row, completion in zip(batch, completions, strict=True):
                pred = extract_gsm8k_answer(completion)
                gold = row["gold"]
                is_correct = pred is not None and pred.strip() == gold.strip()
                correct += int(is_correct)
                total += 1
                samples.append((row["question"], gold, pred or "<none>", completion))

            progress.update(task, advance=1)

    accuracy = correct / max(total, 1)
    console.print(
        Panel(
            f"[bold green]Exact match:[/bold green] {correct}/{total} = {accuracy:.2%}",
            title="[bold cyan]Eval[/bold cyan]",
            border_style="cyan",
        )
    )

    table = Table(title="Sample predictions", show_lines=False)
    table.add_column("Question", overflow="fold", max_width=50)
    table.add_column("Gold", style="green")
    table.add_column("Pred", style="yellow")
    for question, gold, pred, _ in samples[:5]:
        table.add_row(question[:100] + ("..." if len(question) > 100 else ""), gold, pred)
    console.print(table)

    return accuracy


def main(cfg: Config) -> None:
    console = Console()
    seed_everything(cfg.seed)
    console.print(f"[dim]Attention implementation: {get_attn_implementation()}[/dim]")

    # Stage 1+2: rollouts + scoring (cache-aware).
    if not preprocess.rollouts_path(cfg).exists():
        console.print("[dim]Cache miss — running preprocess.run(cfg)[/dim]")
        preprocess.run(cfg)
    records = preprocess.load_cached_records(cfg)
    console.print(f"[dim]Loaded {len(records)} scored records.[/dim]")

    # Stage 3a: selection.
    selected_pairs = select(records, cfg)
    console.print(
        Panel(
            f"[bold]Strategy:[/bold] {cfg.selection.strategy}\n"
            f"[bold]Training pairs:[/bold] {len(selected_pairs)}",
            title="[bold magenta]Selection[/bold magenta]",
            border_style="magenta",
        )
    )

    wandb_project = os.environ.get("WANDB_PROJECT", cfg.wandb_project)
    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project=wandb_project,
            name=os.environ.get("WANDB_RUN_NAME", cfg.wandb_run_name)
            or f"rs-{cfg.selection.strategy}",
            config=cfg.model_dump(),
        )
    wandb.log(
        {"selection/strategy": cfg.selection.strategy, "selection/num_pairs": len(selected_pairs)}
    )

    # Stage 3b: load policy model and SFT on the selected pairs.
    model_device = torch.device(
        f"cuda:{cfg.model_device_id}" if torch.cuda.is_available() else "cpu"
    )
    console.print(f"[dim]Loading policy model for SFT: {cfg.model_name}[/dim]")
    model, tokenizer = load_model(cfg.model_name, model_device, gradient_checkpointing=True)
    print_model_info(console, model)
    console.print(f"[dim]VRAM after policy load: {cuda_memory_gb():.2f} GB[/dim]")

    sft(cfg, model, tokenizer, selected_pairs, console, start_time=time.time())

    # Gradient checkpointing slows generate() (recomputes activations) and
    # disables the KV cache — flip both off before eval.
    model.gradient_checkpointing_disable()
    model.config.use_cache = True

    # Stage 3c: evaluate on GSM8k test split.
    accuracy = evaluate(cfg, model, tokenizer, console)
    wandb.log({"test_accuracy": accuracy, "selection/strategy": cfg.selection.strategy})

    if cfg.save_checkpoint:
        ckpt_dir = Path(cfg.output_dir) / "checkpoints" / f"rs-{cfg.selection.strategy}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[dim]Saving checkpoint to {ckpt_dir}[/dim]")
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

    wandb.finish()


def main_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Rejection sampling: selection + SFT + GSM8k eval."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(load_config(args.config))


if __name__ == "__main__":
    main_cli()
