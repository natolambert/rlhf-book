# Utilities for Instruction Tuning (SFT).

import os
import platform
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from .config import Config


console = Console()

IGNORE_INDEX = -100

DEFAULT_SAMPLE_PROMPTS: list[str] = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a haiku about programming.",
    "How does photosynthesis work?",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_attn_implementation() -> str:
    if platform.machine() != "x86_64":
        return "sdpa"
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def load_model(cfg: Config, device: torch.device):
    attn_impl = get_attn_implementation()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=False)

    if tokenizer.chat_template is None and cfg.chat_template_source:
        donor = AutoTokenizer.from_pretrained(cfg.chat_template_source, trust_remote_code=False)
        if donor.chat_template is None:
            raise ValueError(
                f"chat_template_source {cfg.chat_template_source} has no chat_template."
            )
        tokenizer.chat_template = donor.chat_template

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=False,
        attn_implementation=attn_impl,
        torch_dtype=dtype,
    ).to(device)

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    return model, tokenizer


@dataclass
class SFTBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

    def to(self, device: torch.device | str) -> "SFTBatch":
        return SFTBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
        )


def compute_loss(model, batch: "SFTBatch") -> torch.Tensor:
    """Causal-LM cross-entropy with prompt-masked labels (-100 ignored)."""
    out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
    shift_logits = out.logits[:, :-1, :].contiguous()
    shift_labels = batch.labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


def _encode_row(
    messages: list[dict],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> dict[str, torch.Tensor] | None:
    """Render ``messages`` with the chat template and mask all but the final assistant turn."""
    if not messages or messages[-1]["role"] != "assistant":
        return None

    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=False
    )
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=False
    )
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(full_ids[len(prompt_ids) :])

    if max_length is not None and len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        labels = labels[:max_length]

    if all(label == IGNORE_INDEX for label in labels):
        return None

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class SFTDataset(Dataset):
    def __init__(self, encoded: list[dict[str, torch.Tensor]]):
        self.encoded = encoded

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.encoded[idx]


def _collate(examples: list[dict[str, torch.Tensor]], pad_token_id: int) -> SFTBatch:
    max_len = max(ex["input_ids"].size(0) for ex in examples)
    input_ids, attention_mask, labels = [], [], []
    for ex in examples:
        ids, lbl = ex["input_ids"], ex["labels"]
        pad = max_len - ids.size(0)
        input_ids.append(torch.cat([ids, torch.full((pad,), pad_token_id, dtype=torch.long)]))
        attention_mask.append(
            torch.cat(
                [torch.ones(ids.size(0), dtype=torch.long), torch.zeros(pad, dtype=torch.long)]
            )
        )
        labels.append(torch.cat([lbl, torch.full((pad,), IGNORE_INDEX, dtype=torch.long)]))
    return SFTBatch(
        input_ids=torch.stack(input_ids),
        attention_mask=torch.stack(attention_mask),
        labels=torch.stack(labels),
    )


def create_dataloader(cfg: Config, tokenizer: PreTrainedTokenizer) -> DataLoader:
    raw = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    if cfg.max_samples is not None and len(raw) > cfg.max_samples:
        raw = raw.select(range(cfg.max_samples))

    encoded: list[dict[str, torch.Tensor]] = []
    skipped = 0
    for example in raw:
        row = _encode_row(example["messages"], tokenizer, cfg.max_length)
        if row is None:
            skipped += 1
            continue
        encoded.append(row)

    if not encoded:
        raise RuntimeError("No trainable rows after tokenization.")
    if skipped:
        console.print(
            f"[dim]Skipped {skipped}/{len(raw)} rows (no trainable assistant tokens).[/dim]"
        )

    pad_id = tokenizer.pad_token_id
    return DataLoader(
        SFTDataset(encoded),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate(batch, pad_id),
        num_workers=0,
        pin_memory=False,
    )


def generate_samples(
    model,
    tokenizer: PreTrainedTokenizer,
    cfg: Config,
    step: int,
    prompts: list[str] | None = None,
    max_new_tokens: int | None = None,
) -> None:
    was_training = model.training
    model.eval()
    new_tokens = max_new_tokens if max_new_tokens is not None else cfg.sample_max_tokens
    if prompts is None:
        prompts = DEFAULT_SAMPLE_PROMPTS

    console.print(f"\n[bold yellow]Samples @ step {step}:[/bold yellow]")

    for prompt_id, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.sample_max_input_tokens,
        ).to(model.device)
        kwargs = dict(
            **inputs,
            max_new_tokens=new_tokens,
            do_sample=cfg.sample_do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        if cfg.sample_do_sample:
            kwargs.update(temperature=cfg.sample_temperature, top_p=cfg.sample_top_p)
        with torch.no_grad():
            out = model.generate(**kwargs)
        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        console.print(f"[bold cyan]Prompt {prompt_id}:[/bold cyan] {prompt}")
        console.print(f"[bold green]Response:[/bold green] {response[:500]}")

    if was_training:
        model.train()


def progress_bar() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("*"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def print_training_info(model, cfg: Config, total_steps: int, warmup_steps: int) -> None:
    console.print(
        Panel(
            f"[bold magenta]Model:[/bold magenta] {cfg.model_name}\n"
            f"[dim]Parameters:[/dim] {sum(p.numel() for p in model.parameters()):,}\n"
            f"[dim]Device:[/dim] {model.device}\n"
            f"[dim]Dataset:[/dim] {cfg.dataset_name} (split={cfg.dataset_split})\n"
            f"[dim]Effective batch:[/dim] {cfg.batch_size} x {cfg.gradient_accumulation_steps}"
            f" = {cfg.batch_size * cfg.gradient_accumulation_steps}\n"
            f"[dim]Steps:[/dim] {total_steps} total, {warmup_steps} warmup",
            title="[bold magenta]SFT Configuration[/bold magenta]",
            border_style="magenta",
        )
    )


def print_epoch_header(epoch_idx: int, total_epochs: int) -> None:
    console.rule(f"[bold cyan]Epoch {epoch_idx + 1}/{total_epochs}[/bold cyan]", style="cyan")


def make_lr_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps + 1))
        remaining = total_steps - step
        return max(0.0, remaining / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
