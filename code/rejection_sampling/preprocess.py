# Stage 1 (rollouts) + Stage 2 (reward-model scoring) for rejection sampling.
# Can be run standalone or called from train.py on a cache miss.

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from rich.console import Console
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
)

from policy_gradients.train import get_attn_implementation, load_model, seed_everything
from policy_gradients.utils import progress_bar

from .config import Config, load_config
from .utils import (
    ACEMATH_SYSTEM_PROMPT,
    answers_match,
    cache_key,
    cuda_memory_gb,
    extract_gsm8k_answer,
    format_gsm8k_gold,
    free_memory,
)


Prompt = dict  # {"question": str, "gold": str}


def load_gsm8k_prompts(cfg: Config) -> list[Prompt]:
    ds = load_dataset(cfg.data.name, cfg.data.subset, split=cfg.data.train_split)
    if cfg.data.max_train_samples is not None:
        ds = ds.select(range(min(cfg.data.max_train_samples, len(ds))))
    return [{"question": row["question"], "gold": format_gsm8k_gold(row["answer"])} for row in ds]


def _build_generation_chat(question: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": ACEMATH_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def generate_completions(
    model,
    tokenizer,
    prompts: list[Prompt],
    cfg: Config,
    console: Console,
) -> list[list[str]]:
    """Generate N completions per prompt, returning an (M, N) list-of-lists."""
    model.eval()
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
        num_return_sequences=cfg.num_completions_per_prompt,
    )

    all_completions: list[list[str]] = []
    with progress_bar(console) as progress:
        total_batches = (len(prompts) + cfg.rollout_batch_size - 1) // cfg.rollout_batch_size
        task = progress.add_task("Generating rollouts", total=total_batches)

        for start in range(0, len(prompts), cfg.rollout_batch_size):
            batch = prompts[start : start + cfg.rollout_batch_size]
            chat_strings = [_build_generation_chat(p["question"], tokenizer) for p in batch]
            model_inputs = tokenizer(
                chat_strings,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                return_attention_mask=True,
            ).to(model.device)

            with torch.no_grad():
                sequence_ids = model.generate(
                    **model_inputs,
                    generation_config=generation_config,
                )

            prompt_len = model_inputs["input_ids"].shape[1]
            completion_ids = sequence_ids[:, prompt_len:]
            decoded = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

            # decoded is flat with num_return_sequences per prompt, grouped per prompt.
            n = cfg.num_completions_per_prompt
            for i in range(len(batch)):
                all_completions.append(decoded[i * n : (i + 1) * n])

            progress.update(task, advance=1)

    return all_completions


def load_reward_model(cfg: Config, device: torch.device):
    """Load AceMath-7B-RM with a single-scalar regression head."""
    attn_impl = get_attn_implementation()
    tokenizer = AutoTokenizer.from_pretrained(cfg.reward_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.reward_model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    # LlamaForSequenceClassification reads model.config.pad_token_id inside
    # its forward pass — setting it on the tokenizer alone is not enough and
    # it raises "Cannot handle batch sizes > 1 if no padding token is defined".
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)
    model.eval()
    return model, tokenizer


def _build_scoring_chat(question: str, completion: str, tokenizer) -> str:
    chat = [
        {"role": "system", "content": ACEMATH_SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": completion},
    ]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)


def score_rollouts(
    rm,
    rm_tokenizer,
    prompts: list[Prompt],
    completions: list[list[str]],
    cfg: Config,
    console: Console,
) -> list[list[float]]:
    """Score every (prompt, completion) pair and return an (M, N) reward matrix."""
    # Pre-tokenize all pairs, then length-sort so each batch has similar
    # sequence lengths and minimal padding waste.
    flat: list[tuple[int, int, list[int]]] = []
    for i, prompt in enumerate(prompts):
        for j, completion in enumerate(completions[i]):
            chat_str = _build_scoring_chat(prompt["question"], completion, rm_tokenizer)
            # Chat template already inserts special tokens; encoding with
            # add_special_tokens=True would double-prepend them.
            ids = rm_tokenizer.encode(chat_str, add_special_tokens=False)
            flat.append((i, j, ids))
    flat.sort(key=lambda item: len(item[2]))

    pad_id = (
        rm_tokenizer.pad_token_id
        if rm_tokenizer.pad_token_id is not None
        else rm_tokenizer.eos_token_id
    )
    rewards: list[list[float]] = [[0.0] * cfg.num_completions_per_prompt for _ in prompts]

    with progress_bar(console) as progress:
        total_batches = (len(flat) + cfg.score_batch_size - 1) // cfg.score_batch_size
        task = progress.add_task("Scoring rollouts", total=total_batches)

        for start in range(0, len(flat), cfg.score_batch_size):
            batch = flat[start : start + cfg.score_batch_size]
            max_len = max(len(item[2]) for item in batch)

            # Left-pad so the last hidden state (which holds the reward) is
            # at the rightmost position regardless of sequence length.
            input_ids = torch.full(
                (len(batch), max_len), pad_id, dtype=torch.long, device=rm.device
            )
            attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=rm.device)
            for row, (_, _, ids) in enumerate(batch):
                pad_len = max_len - len(ids)
                input_ids[row, pad_len:] = torch.tensor(ids, dtype=torch.long, device=rm.device)
                attention_mask[row, pad_len:] = 1

            with torch.no_grad():
                outputs = rm(input_ids=input_ids, attention_mask=attention_mask)
            scores = outputs.logits[:, 0].float().cpu().tolist()

            for (prompt_idx, completion_idx, _), score in zip(batch, scores, strict=True):
                rewards[prompt_idx][completion_idx] = score

            progress.update(task, advance=1)

    return rewards


def rollouts_path(cfg: Config) -> Path:
    return Path(cfg.output_dir) / "rollouts" / f"{cache_key(cfg)}.jsonl"


def rollouts_intermediate_path(cfg: Config) -> Path:
    """Pre-scoring intermediate cache (rollouts only, no rewards)."""
    return Path(cfg.output_dir) / "rollouts" / f"{cache_key(cfg)}.rollouts.jsonl"


def _write_rollouts_intermediate(
    path: Path, prompts: list[Prompt], completions: list[list[str]]
) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for prompt, comp_list in zip(prompts, completions, strict=True):
            record = {
                "question": prompt["question"],
                "answer": prompt["gold"],
                "completions": comp_list,
            }
            f.write(json.dumps(record) + "\n")
    os.replace(tmp, path)


def _load_rollouts_intermediate(
    path: Path,
) -> tuple[list[Prompt], list[list[str]]]:
    prompts: list[Prompt] = []
    completions: list[list[str]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompts.append({"question": rec["question"], "gold": rec["answer"]})
            completions.append(rec["completions"])
    return prompts, completions


def run(cfg: Config) -> Path:
    """Run Stage 1 + Stage 2 and write the JSONL cache. Skips if cache exists."""
    console = Console()
    seed_everything(cfg.seed)

    cache_path = rollouts_path(cfg)
    if cache_path.exists():
        console.print(f"[dim]Rollout cache hit at {cache_path} — skipping preprocess.[/dim]")
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    intermediate_path = rollouts_intermediate_path(cfg)

    # Stage 1: generation. We persist rollouts to an intermediate file before
    # touching the reward model so a Stage 2 OOM doesn't force regeneration.
    if intermediate_path.exists():
        console.print(f"[dim]Resuming from intermediate cache at {intermediate_path}[/dim]")
        prompts, completions = _load_rollouts_intermediate(intermediate_path)
    else:
        prompts = load_gsm8k_prompts(cfg)
        console.print(f"[dim]Loaded {len(prompts)} GSM8k prompts.[/dim]")

        model_device = torch.device(
            f"cuda:{cfg.model_device_id}" if torch.cuda.is_available() else "cpu"
        )
        console.print(f"[dim]Stage 1: loading policy model ({cfg.model_name})[/dim]")
        policy_model, policy_tokenizer = load_model(
            cfg.model_name, model_device, gradient_checkpointing=False
        )
        console.print(f"[dim]VRAM after policy load: {cuda_memory_gb():.2f} GB[/dim]")

        completions = generate_completions(policy_model, policy_tokenizer, prompts, cfg, console)
        _write_rollouts_intermediate(intermediate_path, prompts, completions)

        # Drop references before free_memory so CUDA buffers actually release.
        del policy_model, policy_tokenizer
        free_memory()
        console.print(f"[dim]VRAM after policy free: {cuda_memory_gb():.2f} GB[/dim]")

    # Stage 2: scoring.
    rm_device = torch.device(
        f"cuda:{cfg.reward_model_device_id}" if torch.cuda.is_available() else "cpu"
    )
    console.print(f"[dim]Stage 2: loading reward model ({cfg.reward_model_name})[/dim]")
    rm, rm_tokenizer = load_reward_model(cfg, rm_device)
    console.print(f"[dim]VRAM after reward-model load: {cuda_memory_gb():.2f} GB[/dim]")

    rewards = score_rollouts(rm, rm_tokenizer, prompts, completions, cfg, console)

    del rm, rm_tokenizer
    free_memory()

    # Atomic write: .tmp -> rename.
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        for prompt, comp_list, reward_list in zip(prompts, completions, rewards, strict=True):
            record = {
                "question": prompt["question"],
                "answer": prompt["gold"],
                "completions": comp_list,
                "rewards": reward_list,
            }
            f.write(json.dumps(record) + "\n")
    os.replace(tmp_path, cache_path)
    console.print(f"[bold green]Wrote rollout cache:[/bold green] {cache_path}")

    # Log decidable_fraction so the user knows how much signal top_per_prompt
    # actually has on this cache.  A decidable prompt has both correct and
    # incorrect completions; on all-correct or all-wrong rows, selection
    # strategy cannot matter.
    n_decidable = 0
    for prompt, comp_list, reward_list in zip(prompts, completions, rewards, strict=True):
        has_correct = any(answers_match(extract_gsm8k_answer(c), prompt["gold"]) for c in comp_list)
        has_wrong = any(
            not answers_match(extract_gsm8k_answer(c), prompt["gold"]) for c in comp_list
        )
        if has_correct and has_wrong:
            n_decidable += 1
    frac = n_decidable / len(prompts) if prompts else 0.0
    console.print(
        f"[dim]decidable_fraction: {n_decidable}/{len(prompts)} = {frac:.3f} "
        f"(prompts where selection strategy can matter)[/dim]"
    )

    # Drop the intermediate now that the final cache is on disk.
    intermediate_path.unlink(missing_ok=True)

    return cache_path


def load_cached_records(cfg: Config) -> list[dict]:
    path = rollouts_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"Rollout cache not found at {path}; run preprocess.run(cfg) first."
        )
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate + score rollouts for rejection sampling."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main_cli()
