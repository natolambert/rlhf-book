# Stage 1 (rollouts) + Stage 2 (reward-model scoring) for rejection sampling.
#
# Runnable as a module (``python -m rejection_sampling.preprocess --config ...``)
# or as a library (``preprocess.run(cfg)`` — which is what train.py calls on a
# cache miss).

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

# We deliberately import these helpers verbatim from policy_gradients — the
# attention-implementation selection, seeding, and policy-model loader are
# identical to what we need for Stage 1.
from policy_gradients.train import (
    get_attn_implementation,
    load_model,
    seed_everything,
)
from policy_gradients.utils import progress_bar

from .config import Config, load_config
from .utils import (
    ACEMATH_SYSTEM_PROMPT,
    cache_key,
    cuda_memory_gb,
    format_gsm8k_gold,
    free_memory,
)


Prompt = dict  # {"question": str, "gold": str}


def load_gsm8k_prompts(cfg: Config) -> list[Prompt]:
    """Load and truncate the GSM8k train split used for rollouts."""
    ds = load_dataset(cfg.data.name, cfg.data.subset, split=cfg.data.train_split)
    if cfg.data.max_train_samples is not None:
        ds = ds.select(range(min(cfg.data.max_train_samples, len(ds))))
    return [
        {"question": row["question"], "gold": format_gsm8k_gold(row["answer"])}
        for row in ds
    ]


def _build_generation_chat(question: str, tokenizer) -> str:
    """Apply Qwen3's chat template with the AceMath system prompt.

    AceMath-7B-RM expects completions that follow the
    "reason step by step ... within \\boxed{}" convention, so we must use
    that system prompt during generation too.
    """
    messages = [
        {"role": "system", "content": ACEMATH_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_completions(
    model,
    tokenizer,
    prompts: list[Prompt],
    cfg: Config,
    console: Console,
) -> list[list[str]]:
    """Generate ``cfg.num_completions_per_prompt`` completions for each prompt.

    Returns a list of length M where each element is a list of N completion
    strings. Generation is batched over prompts at ``cfg.rollout_batch_size``;
    the reference policy stays in sampling mode so the multiple rollouts per
    prompt are actually different.
    """
    model.eval()
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
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
    """Load AceMath-7B-RM as a sequence classifier with a single-scalar head."""
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
    # LlamaForSequenceClassification (AceMath's base class) reads
    # self.config.pad_token_id inside its forward pass to pick the last
    # non-pad hidden state. When it's None and batch > 1 it raises
    # "Cannot handle batch sizes > 1 if no padding token is defined." —
    # setting it on the tokenizer alone is not enough.
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)
    model.eval()
    return model, tokenizer


def _build_scoring_chat(question: str, completion: str, tokenizer) -> str:
    """Wrap a (question, completion) pair in AceMath's expected chat format."""
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
    """Score every (prompt, completion) pair with AceMath-7B-RM.

    We sort all (prompt, completion) pairs by encoded-length before batching
    (the length-sorting throughput trick noted in chapter.md §Implementation
    Details) and use left-padding so the last hidden state — which holds the
    scalar reward — stays at the rightmost position regardless of sequence
    length.
    """
    # Flatten (prompt_idx, completion_idx, encoded_ids) and pre-tokenize.
    flat: list[tuple[int, int, list[int]]] = []
    for i, prompt in enumerate(prompts):
        for j, completion in enumerate(completions[i]):
            chat_str = _build_scoring_chat(prompt["question"], completion, rm_tokenizer)
            # The chat template already inserts special tokens; passing
            # add_special_tokens=True here would double-prepend them and
            # corrupt the score.
            ids = rm_tokenizer.encode(chat_str, add_special_tokens=False)
            flat.append((i, j, ids))

    # Length-sort: batches end up with similar sequence lengths, minimising
    # padding waste on the reward-model forward pass.
    flat.sort(key=lambda item: len(item[2]))

    pad_id = (
        rm_tokenizer.pad_token_id
        if rm_tokenizer.pad_token_id is not None
        else rm_tokenizer.eos_token_id
    )

    # Pre-allocate an (M x N) output table so we can write scores back in
    # their original positions after sorting.
    rewards: list[list[float]] = [
        [0.0] * cfg.num_completions_per_prompt for _ in prompts
    ]

    with progress_bar(console) as progress:
        total_batches = (len(flat) + cfg.score_batch_size - 1) // cfg.score_batch_size
        task = progress.add_task("Scoring rollouts", total=total_batches)

        for start in range(0, len(flat), cfg.score_batch_size):
            batch = flat[start : start + cfg.score_batch_size]
            max_len = max(len(item[2]) for item in batch)

            input_ids = torch.full(
                (len(batch), max_len), pad_id, dtype=torch.long, device=rm.device
            )
            attention_mask = torch.zeros(
                (len(batch), max_len), dtype=torch.long, device=rm.device
            )
            for row, (_, _, ids) in enumerate(batch):
                pad_len = max_len - len(ids)
                input_ids[row, pad_len:] = torch.tensor(ids, dtype=torch.long, device=rm.device)
                attention_mask[row, pad_len:] = 1

            with torch.no_grad():
                outputs = rm(input_ids=input_ids, attention_mask=attention_mask)
            # AceMath's head returns (batch, num_labels=1). We take [:, 0].
            scores = outputs.logits[:, 0].float().cpu().tolist()

            for (prompt_idx, completion_idx, _), score in zip(batch, scores, strict=True):
                rewards[prompt_idx][completion_idx] = score

            progress.update(task, advance=1)

    return rewards


def rollouts_path(cfg: Config) -> Path:
    """Path to the cache file for this config's (generation + scoring) params."""
    out_dir = Path(cfg.output_dir) / "rollouts"
    return out_dir / f"{cache_key(cfg)}.jsonl"


def rollouts_intermediate_path(cfg: Config) -> Path:
    """Path to the pre-scoring intermediate cache (rollouts without rewards)."""
    out_dir = Path(cfg.output_dir) / "rollouts"
    return out_dir / f"{cache_key(cfg)}.rollouts.jsonl"


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
    """Run Stage 1 + Stage 2 end-to-end and write the result to disk.

    Returns the path to the JSONL cache file. If the cache already exists,
    Stage 1/2 are skipped entirely.
    """
    console = Console()
    seed_everything(cfg.seed)

    cache_path = rollouts_path(cfg)
    if cache_path.exists():
        console.print(f"[dim]Rollout cache hit at {cache_path} — skipping preprocess.[/dim]")
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    intermediate_path = rollouts_intermediate_path(cfg)

    # --- Stage 1: generation (resumable via intermediate cache) ----------------
    if intermediate_path.exists():
        console.print(
            f"[dim]Intermediate rollouts found at {intermediate_path} — "
            f"skipping Stage 1 and resuming at scoring.[/dim]"
        )
        prompts, completions = _load_rollouts_intermediate(intermediate_path)
        console.print(
            f"[dim]Loaded {len(prompts)} prompts × "
            f"{len(completions[0]) if completions else 0} completions from intermediate cache.[/dim]"
        )
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

        # Persist rollouts immediately, before we touch the reward model.
        # Stage 2 is fragile (OOMs, pad-token mismatches, etc.) and we do
        # not want to redo an hour of generation on every scoring failure.
        _write_rollouts_intermediate(intermediate_path, prompts, completions)
        console.print(f"[dim]Saved intermediate rollouts to {intermediate_path}[/dim]")

        # Drop references before calling free_memory so the underlying CUDA
        # buffers can actually be released.
        del policy_model, policy_tokenizer
        free_memory()
        vram_between = cuda_memory_gb()
        console.print(f"[dim]VRAM after policy free: {vram_between:.2f} GB[/dim]")
        assert vram_between < 1.0, (
            f"Expected <1 GB between stages, got {vram_between:.2f} GB — free_memory leak?"
        )

    # --- Stage 2: scoring ------------------------------------------------------
    rm_device = torch.device(
        f"cuda:{cfg.reward_model_device_id}" if torch.cuda.is_available() else "cpu"
    )
    console.print(f"[dim]Stage 2: loading reward model ({cfg.reward_model_name})[/dim]")
    rm, rm_tokenizer = load_reward_model(cfg, rm_device)
    console.print(f"[dim]VRAM after reward-model load: {cuda_memory_gb():.2f} GB[/dim]")

    rewards = score_rollouts(rm, rm_tokenizer, prompts, completions, cfg, console)

    del rm, rm_tokenizer
    free_memory()
    console.print(f"[dim]VRAM after reward-model free: {cuda_memory_gb():.2f} GB[/dim]")

    # --- Write JSONL atomically (.tmp -> rename) -------------------------------
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

    # Intermediate is redundant now that the final cache (which includes
    # rewards) is on disk — remove it so we don't double-store rollouts.
    if intermediate_path.exists():
        intermediate_path.unlink()

    return cache_path


def load_cached_records(cfg: Config) -> list[dict]:
    """Read back the JSONL cache into the in-memory records format."""
    path = rollouts_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"Rollout cache not found at {path}; run preprocess.run(cfg) first.")
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Generate + score rollouts for rejection sampling.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main_cli()
