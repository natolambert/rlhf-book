# Direct Alignment Training Loop
#
# Educational implementations of DPO and related algorithms for RLHF Book.
# See Chapter 8 for mathematical derivations.
#
# Usage:
#   uv run python -m direct_alignment.train --config configs/dpo.yaml
#   uv run python -m direct_alignment.train --loss dpo --max_samples 1000
#
# References:
# - DPO: https://github.com/eric-mitchell/direct-preference-optimization
# - TRL DPOTrainer: https://huggingface.co/docs/trl/dpo_trainer

import argparse
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config, load_config
from .data import PreferenceBatch, create_dataloader
from .loss import compute_logprobs, get_loss_function, ORPOLoss


def get_attn_implementation() -> str:
    """Determine the best attention implementation for this platform."""
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


def load_model(
    model_name: str,
    device: str,
    gradient_checkpointing: bool = True,
    bf16: bool = True,
):
    """Load model and tokenizer."""
    attn_impl = get_attn_implementation()
    dtype = torch.bfloat16 if bf16 else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        torch_dtype=dtype,
    )
    model = model.to(device)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    return model, tokenizer


def load_ref_model(model_name: str, device: str, bf16: bool = True):
    """Load reference model (frozen, no gradient checkpointing)."""
    attn_impl = get_attn_implementation()
    dtype = torch.bfloat16 if bf16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        torch_dtype=dtype,
    )
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


def forward_pass(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    average_log_prob: bool = False,
) -> torch.Tensor:
    """Compute log probabilities for a sequence.

    Args:
        model: Language model
        input_ids: Token ids (batch, seq_len)
        attention_mask: Attention mask (batch, seq_len) - for model forward
        response_mask: Response mask (batch, seq_len) - 1 for response tokens only
        average_log_prob: If True, return average log prob (for SimPO)

    Returns:
        Log probabilities (batch,)
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits

    # Use response_mask for computing log probs (only response tokens contribute)
    return compute_logprobs(
        logits=logits,
        labels=input_ids,
        mask=response_mask,
        average_log_prob=average_log_prob,
    )


def compute_nll_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute negative log likelihood loss (for ORPO SFT component).

    Only computes loss on response tokens (not prompt).
    """
    # Shift for autoregressive
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = response_mask[:, 1:].contiguous()

    # Compute per-token loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    loss = loss.view(shift_labels.size())

    # Mask and average over response tokens only
    loss = (loss * shift_mask).sum(dim=-1) / shift_mask.sum(dim=-1).clamp(min=1)
    return loss


def train_step(
    policy_model,
    ref_model,
    batch: PreferenceBatch,
    loss_fn,
    optimizer,
    max_grad_norm: float,
    gradient_accumulation_steps: int,
    step_in_accumulation: int,
    use_average_logprob: bool = False,
) -> dict:
    """Perform a single training step.

    Returns:
        Dictionary with loss and metrics
    """
    device = next(policy_model.parameters()).device
    batch = batch.to(device)

    # Forward pass through policy model (use response_mask for loss computation)
    policy_chosen_logps = forward_pass(
        policy_model,
        batch.chosen_input_ids,
        batch.chosen_attention_mask,
        batch.chosen_response_mask,
        average_log_prob=use_average_logprob,
    )
    policy_rejected_logps = forward_pass(
        policy_model,
        batch.rejected_input_ids,
        batch.rejected_attention_mask,
        batch.rejected_response_mask,
        average_log_prob=use_average_logprob,
    )

    # Forward pass through reference model (if needed)
    if ref_model is not None:
        with torch.no_grad():
            ref_chosen_logps = forward_pass(
                ref_model,
                batch.chosen_input_ids,
                batch.chosen_attention_mask,
                batch.chosen_response_mask,
                average_log_prob=use_average_logprob,
            )
            ref_rejected_logps = forward_pass(
                ref_model,
                batch.rejected_input_ids,
                batch.rejected_attention_mask,
                batch.rejected_response_mask,
                average_log_prob=use_average_logprob,
            )
    else:
        ref_chosen_logps = None
        ref_rejected_logps = None

    # Compute loss
    # Handle ORPO which needs NLL loss
    if isinstance(loss_fn, ORPOLoss):
        # Need to compute NLL loss for ORPO (only on response tokens)
        policy_outputs = policy_model(
            input_ids=batch.chosen_input_ids,
            attention_mask=batch.chosen_attention_mask,
            use_cache=False,
        )
        chosen_nll = compute_nll_loss(
            policy_outputs.logits,
            batch.chosen_labels,
            batch.chosen_response_mask,
        )
        loss, metrics = loss_fn(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            chosen_nll_loss=chosen_nll,
        )
    else:
        loss, metrics = loss_fn(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
        )

    # Scale loss for gradient accumulation
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()

    # Optimizer step (only at end of accumulation)
    grad_norm = None
    if (step_in_accumulation + 1) % gradient_accumulation_steps == 0:
        grad_norm = clip_grad_norm_(policy_model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    metrics["loss"] = loss.item()
    if grad_norm is not None:
        metrics["grad_norm"] = grad_norm.item()

    return metrics


def print_training_info(console: Console, cfg: Config, num_samples: int):
    """Print training configuration."""
    console.print("\n[bold]Direct Alignment Training[/bold]")
    console.print(f"  Model: {cfg.model_name}")
    console.print(f"  Loss: {cfg.loss}")
    console.print(f"  Beta: {cfg.beta}")
    console.print(f"  Dataset: {cfg.dataset_name}")
    console.print(f"  Samples: {num_samples}")
    console.print(f"  Batch size: {cfg.batch_size} x {cfg.gradient_accumulation_steps} = {cfg.batch_size * cfg.gradient_accumulation_steps}")
    console.print(f"  Learning rate: {cfg.learning_rate}")
    console.print(f"  Epochs: {cfg.num_epochs}")
    console.print(
        "  In-loop samples: "
        f"every {cfg.sample_every} steps, "
        f"{cfg.sample_num_prompts} prompts/event, "
        f"strategy={cfg.sample_prompt_strategy}"
    )
    console.print()


# Default prompt pool for in-loop generation logging
DEFAULT_SAMPLE_PROMPT_POOL = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a haiku about programming.",
    "Give me a three-step plan to learn probability basics.",
    "What are the pros and cons of remote work for small teams?",
    "Summarize why gradient descent works in plain English.",
    "Write a short email declining a meeting politely.",
    "How does photosynthesis work?",
    "Propose a weekend itinerary for Seattle on a rainy day.",
    "What is overfitting in machine learning?",
    "Compare TCP and UDP with one practical example each.",
    "Draft a bedtime story about a curious robot.",
    "Explain inflation to a high school student.",
    "Give debugging steps when a Python script hangs intermittently.",
    "How would you evaluate a chatbot for safety and helpfulness?",
    "Write SQL to find the top 5 customers by revenue.",
]


def load_sample_prompt_pool(cfg: Config, console: Console) -> list[str]:
    """Load prompt pool for in-loop sampling."""
    if cfg.sample_prompts_file is None:
        return DEFAULT_SAMPLE_PROMPT_POOL.copy()

    path = Path(cfg.sample_prompts_file)
    if not path.exists():
        raise FileNotFoundError(f"sample_prompts_file not found: {path}")

    if path.suffix.lower() == ".json":
        raw_data = json.loads(path.read_text())
        if not isinstance(raw_data, list) or not all(isinstance(x, str) for x in raw_data):
            raise ValueError("sample_prompts_file JSON must be a list of strings")
        prompts = [p.strip() for p in raw_data if p.strip()]
    else:
        prompts = [line.strip() for line in path.read_text().splitlines() if line.strip()]

    if not prompts:
        raise ValueError(f"No prompts found in {path}")

    console.print(f"[dim]Loaded {len(prompts)} prompts from {path}[/dim]")
    return prompts


def select_sample_prompts(
    prompt_pool: list[str],
    sample_num_prompts: int,
    strategy: str,
    sample_event_idx: int,
    rng: random.Random,
) -> list[dict]:
    """Select prompts for this sampling event."""
    if not prompt_pool:
        raise ValueError("Prompt pool cannot be empty")

    k = min(sample_num_prompts, len(prompt_pool))

    if strategy == "fixed":
        indices = list(range(k))
    elif strategy == "round_robin":
        start = (sample_event_idx * k) % len(prompt_pool)
        indices = [(start + i) % len(prompt_pool) for i in range(k)]
    elif strategy == "random":
        indices = rng.sample(list(range(len(prompt_pool))), k=k)
    else:
        raise ValueError(f"Unknown sample_prompt_strategy: {strategy}")

    return [{"prompt_id": idx, "prompt": prompt_pool[idx]} for idx in indices]


def generate_samples(
    model,
    tokenizer,
    prompt_entries: list[dict],
    max_new_tokens: int = 128,
    max_input_tokens: int = 512,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
    console: Console | None = None,
) -> list[dict]:
    """Generate sample outputs from the model for inspection.

    Args:
        model: The policy model
        tokenizer: Tokenizer
        prompt_entries: List of {"prompt_id": int, "prompt": str}
        max_new_tokens: Maximum tokens to generate
        max_input_tokens: Max prompt tokens fed to generation
        do_sample: Whether to sample from next-token distribution
        temperature: Sampling temperature (used when do_sample=True)
        top_p: Top-p nucleus sampling threshold (used when do_sample=True)
        console: Rich console for printing (optional)

    Returns:
        List of dicts with prompt/response and generation metadata
    """
    model.eval()
    samples = []

    for entry in prompt_entries:
        prompt_id = entry["prompt_id"]
        prompt = entry["prompt"]

        # Format as chat
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        ).to(model.device)

        # Generate
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        if do_sample:
            generation_kwargs.update(
                temperature=temperature,
                top_p=top_p,
            )

        with torch.no_grad():
            outputs = model.generate(**generation_kwargs)

        # Decode response only (not the prompt)
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        samples.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "response": response,
                "max_new_tokens": max_new_tokens,
                "max_input_tokens": max_input_tokens,
                "do_sample": do_sample,
                "temperature": temperature if do_sample else None,
                "top_p": top_p if do_sample else None,
            }
        )

        # Print to console if provided
        if console:
            console.print(f"\n[bold cyan]Prompt {prompt_id}:[/bold cyan] {prompt}")
            console.print(f"[bold green]Response:[/bold green] {response[:500]}{'...' if len(response) > 500 else ''}")

    model.train()
    return samples


def log_samples_to_wandb(samples: list[dict], step: int, strategy: str, prompt_pool_size: int):
    """Log generated samples to wandb as a table."""
    table = wandb.Table(
        columns=[
            "prompt_id",
            "prompt",
            "response",
            "do_sample",
            "temperature",
            "top_p",
            "max_new_tokens",
            "max_input_tokens",
        ]
    )
    for sample in samples:
        table.add_data(
            sample["prompt_id"],
            sample["prompt"],
            sample["response"][:1000],  # Truncate long responses
            sample["do_sample"],
            sample["temperature"],
            sample["top_p"],
            sample["max_new_tokens"],
            sample["max_input_tokens"],
        )
    wandb.log(
        {
            "samples": table,
            "samples_count": len(samples),
            "samples_prompt_strategy": strategy,
            "samples_prompt_pool_size": prompt_pool_size,
        },
        step=step,
    )


def main(cfg: Config):
    """Main training loop."""
    seed_everything(cfg.seed)
    console = Console()

    # Print attention implementation
    attn_impl = get_attn_implementation()
    console.print(f"[dim]Using attention implementation: {attn_impl}[/dim]")

    # Load models
    console.print(f"[dim]Loading policy model: {cfg.model_name}[/dim]")
    policy_model, tokenizer = load_model(
        cfg.model_name,
        cfg.device,
        gradient_checkpointing=cfg.gradient_checkpointing,
        bf16=cfg.bf16,
    )

    # Reference model (not needed for SimPO and ORPO)
    if cfg.loss in ["simpo", "orpo"]:
        ref_model = None
        console.print("[dim]No reference model needed for this loss[/dim]")
    else:
        console.print(f"[dim]Loading reference model: {cfg.ref_model_name}[/dim]")
        ref_model = load_ref_model(cfg.ref_model_name, cfg.device, bf16=cfg.bf16)

    # Create dataloader
    console.print(f"[dim]Loading dataset: {cfg.dataset_name}[/dim]")
    dataloader = create_dataloader(
        dataset_name=cfg.dataset_name,
        tokenizer=tokenizer,
        split=cfg.dataset_split,
        max_samples=cfg.max_samples,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    print_training_info(console, cfg, len(dataloader.dataset))

    # Get loss function
    # ORPO and SimPO both use average sequence log-probs (TRL-style).
    # For ORPO this avoids extreme log-odds magnitudes that show up with summed log-probs.
    use_average_logprob = cfg.loss in ["simpo", "orpo"]
    loss_fn = get_loss_function(
        cfg.loss,
        beta=cfg.beta,
        gamma=cfg.gamma if cfg.loss == "simpo" else None,
        label_smoothing=cfg.label_smoothing if cfg.loss in ["dpo", "cdpo"] else None,
    )

    # Optimizer with warmup
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # Learning rate scheduler
    # Note: num_training_steps is optimizer steps, not micro-batches
    steps_per_epoch = len(dataloader) // cfg.gradient_accumulation_steps
    num_training_steps = steps_per_epoch * cfg.num_epochs
    num_warmup_steps = int(num_training_steps * cfg.warmup_ratio)
    console.print(f"[dim]LR schedule: {num_training_steps} steps, {num_warmup_steps} warmup[/dim]")

    def lr_lambda(step):
        if step < num_warmup_steps:
            # Start warmup at 1/num_warmup_steps, not 0 (avoids first step doing nothing)
            return float(step + 1) / float(max(1, num_warmup_steps + 1))
        return max(0.0, float(num_training_steps - step) / float(max(1, num_training_steps - num_warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Initialize wandb
    wandb_project = os.environ.get("WANDB_PROJECT", cfg.wandb_project)
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", cfg.wandb_run_name)

    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name or f"{cfg.loss}-{cfg.model_name.split('/')[-1]}",
            config=vars(cfg),
        )

    # Training loop
    global_step = 0
    start_time = time.time()
    policy_model.train()
    sample_prompt_pool = load_sample_prompt_pool(cfg, console)
    sample_event_idx = 0
    sample_rng = random.Random(cfg.seed)
    console.print(
        "[dim]Sample generation settings: "
        f"pool={len(sample_prompt_pool)}, "
        f"do_sample={cfg.sample_do_sample}, "
        f"temperature={cfg.sample_temperature}, "
        f"top_p={cfg.sample_top_p}, "
        f"max_new_tokens={cfg.sample_max_tokens}[/dim]"
    )

    for epoch in range(cfg.num_epochs):
        console.print(f"\n[bold]Epoch {epoch + 1}/{cfg.num_epochs}[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Training", total=len(dataloader))

            for batch_idx, batch in enumerate(dataloader):
                metrics = train_step(
                    policy_model=policy_model,
                    ref_model=ref_model,
                    batch=batch,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    max_grad_norm=cfg.max_grad_norm,
                    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                    step_in_accumulation=batch_idx,
                    use_average_logprob=use_average_logprob,
                )

                # Update progress bar (advance by 1 for each micro-batch)
                progress.update(task, advance=1)

                # Update scheduler after optimizer step
                if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                    scheduler.step()
                    global_step += 1

                    # Log to wandb
                    metrics["learning_rate"] = scheduler.get_last_lr()[0]
                    metrics["epoch"] = epoch + (batch_idx + 1) / len(dataloader)
                    metrics["hours_elapsed"] = (time.time() - start_time) / 3600
                    wandb.log(metrics, step=global_step)

                    # Update progress description with loss
                    progress.update(
                        task,
                        description=f"[dim]Loss: {metrics['loss']:.4f}[/dim]",
                    )

                    # Generate and log samples periodically
                    if cfg.sample_every > 0 and global_step % cfg.sample_every == 0:
                        console.print(f"\n[bold yellow]Generating samples at step {global_step}...[/bold yellow]")
                        prompt_entries = select_sample_prompts(
                            prompt_pool=sample_prompt_pool,
                            sample_num_prompts=cfg.sample_num_prompts,
                            strategy=cfg.sample_prompt_strategy,
                            sample_event_idx=sample_event_idx,
                            rng=sample_rng,
                        )
                        sample_event_idx += 1
                        samples = generate_samples(
                            model=policy_model,
                            tokenizer=tokenizer,
                            prompt_entries=prompt_entries,
                            max_new_tokens=cfg.sample_max_tokens,
                            max_input_tokens=cfg.sample_max_input_tokens,
                            do_sample=cfg.sample_do_sample,
                            temperature=cfg.sample_temperature,
                            top_p=cfg.sample_top_p,
                            console=console,
                        )
                        log_samples_to_wandb(
                            samples=samples,
                            step=global_step,
                            strategy=cfg.sample_prompt_strategy,
                            prompt_pool_size=len(sample_prompt_pool),
                        )

            # Flush remaining gradients at end of epoch if partial batch exists
            remaining = len(dataloader) % cfg.gradient_accumulation_steps
            if remaining != 0:
                grad_norm = clip_grad_norm_(policy_model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                # Log final partial batch
                metrics["learning_rate"] = scheduler.get_last_lr()[0]
                metrics["epoch"] = epoch + 1.0
                metrics["grad_norm"] = grad_norm.item()
                wandb.log(metrics, step=global_step)

    # Final summary
    console.print("\n[bold green]Training complete![/bold green]")
    console.print(f"  Final loss: {metrics.get('loss', 'N/A'):.4f}")
    if "accuracy" in metrics:
        console.print(f"  Final accuracy: {metrics['accuracy']:.2%}")

    # Generate final samples
    if cfg.sample_every > 0:
        console.print("\n[bold yellow]Final model samples:[/bold yellow]")
        prompt_entries = select_sample_prompts(
            prompt_pool=sample_prompt_pool,
            sample_num_prompts=cfg.sample_num_prompts,
            strategy=cfg.sample_prompt_strategy,
            sample_event_idx=sample_event_idx,
            rng=sample_rng,
        )
        samples = generate_samples(
            model=policy_model,
            tokenizer=tokenizer,
            prompt_entries=prompt_entries,
            max_new_tokens=cfg.sample_max_tokens,
            max_input_tokens=cfg.sample_max_input_tokens,
            do_sample=cfg.sample_do_sample,
            temperature=cfg.sample_temperature,
            top_p=cfg.sample_top_p,
            console=console,
        )
        log_samples_to_wandb(
            samples=samples,
            step=global_step,
            strategy=cfg.sample_prompt_strategy,
            prompt_pool_size=len(sample_prompt_pool),
        )

    # Save model if requested
    if cfg.save_model:
        output_path = Path(cfg.output_dir) / f"{cfg.loss}_{cfg.model_name.split('/')[-1]}"
        output_path.mkdir(parents=True, exist_ok=True)
        console.print(f"\n[dim]Saving model to {output_path}[/dim]")
        policy_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

    wandb.finish()


def main_cli():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train direct alignment models (DPO, IPO, etc.)")

    # Config file (optional)
    parser.add_argument("--config", type=str, help="Path to YAML config file")

    # Override individual settings
    parser.add_argument("--model_name", type=str, help="Model name or path")
    parser.add_argument("--loss", type=str, choices=["dpo", "cdpo", "ipo", "simpo", "orpo", "kto"])
    parser.add_argument("--beta", type=float, help="Beta parameter")
    parser.add_argument("--dataset_name", type=str, help="HuggingFace dataset name")
    parser.add_argument("--max_samples", type=int, help="Max training samples")
    parser.add_argument("--max_length", type=int, help="Max sequence length")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--wandb_project", type=str, help="Wandb project name")
    parser.add_argument("--sample_every", type=int, help="Generate samples every N steps (0 to disable)")
    parser.add_argument("--sample_num_prompts", type=int, help="Prompts per sample event")
    parser.add_argument("--sample_prompt_strategy", type=str, choices=["fixed", "round_robin", "random"])
    parser.add_argument("--sample_prompts_file", type=str, help="Optional .txt/.json prompt list for in-loop samples")
    parser.add_argument("--sample_max_tokens", type=int, help="Max new tokens per in-loop sample")
    parser.add_argument("--sample_max_input_tokens", type=int, help="Max prompt tokens for in-loop sample generation")
    parser.add_argument("--sample_temperature", type=float, help="Temperature for in-loop sample generation")
    parser.add_argument("--sample_top_p", type=float, help="Top-p for in-loop sample generation")
    parser.add_argument("--sample_do_sample", action=argparse.BooleanOptionalAction, help="Enable/disable stochastic in-loop sampling")
    parser.add_argument("--seed", type=int, help="Random seed")

    args = parser.parse_args()

    # Load config from file or use defaults
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = Config()

    # Override with CLI arguments
    for key, value in vars(args).items():
        if value is not None and key != "config":
            setattr(cfg, key, value)

    main(cfg)


if __name__ == "__main__":
    main_cli()
