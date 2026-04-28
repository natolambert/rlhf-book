import re
from itertools import batched  # Requires Python 3.12+
from typing import Any, Iterator

import torch
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .algorithms import apply_reward_kl, compute_advantages, compute_log_probs, compute_values
from .buffer import Experience, ReplayBuffer, join_experiences_batch
from .config import Config


def _correctness_reward(
    dataset: ProceduralDataset,
    completions: list[str],
    entries: list[dict],
) -> list[float]:
    """Compute raw reward scores from the environment"""

    def score_correctness_answer(dataset: ProceduralDataset, completion: str, entry: dict) -> float:
        answer = extract_answer(completion)
        return float(dataset.score_answer(answer, entry))

    return [
        score_correctness_answer(dataset=dataset, completion=completion, entry=entry)
        for completion, entry in zip(completions, entries, strict=True)
    ]


def _response_penalties(lengths: list[int], cfg: Config) -> list[float]:
    """Compute penalties of responses (e.g. based on length)"""

    def dapo_length_penalty(completion_len: int, l_cache: int, l_max: int) -> float:
        safe_len = l_max - l_cache
        if completion_len <= safe_len:
            return 0.0
        if completion_len <= l_max:
            return (safe_len - completion_len) / l_cache
        return -1.0

    if cfg.loss == "dapo":
        return [dapo_length_penalty(length, cfg.l_cache, cfg.l_max) for length in lengths]
    return [0 for _ in lengths]


def _format_reward(completions: list[str]) -> list[float]:
    """Compute format reward based on presence of thinking/answer tags."""

    def count_tags(text: str) -> float:
        count = 0.0
        if re.search(r"\s*<think>\s*", text):
            count += 0.25
        if re.search(r"\s*</think>\s*", text):
            count += 0.25
        if re.search(r"\s*<answer>\s*", text):
            count += 0.25
        if re.search(r"\s*</answer>\s*", text):
            count += 0.25
        return count

    return [count_tags(c) for c in completions]


def compute_rewards(
    dataset: ProceduralDataset,
    completions: list[str],
    entries: list[dict],
    cfg: Config,
    lengths: list[int],
) -> tuple[list[float], list[float]]:
    """Compute training rewards and raw correctness rewards for filtering."""
    correctness_rewards = _correctness_reward(dataset, completions, entries)
    response_penalties = _response_penalties(lengths, cfg)
    format_rewards = _format_reward(completions)
    combined_rewards = [
        acc + pen + cfg.format_weight * fmt
        for acc, pen, fmt in zip(
            correctness_rewards, response_penalties, format_rewards, strict=True
        )
    ]
    return combined_rewards, correctness_rewards


class TransformerRolloutEngine:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        cfg: Config,
        ref_model,
        val_model,
        cpu_device: torch.device,
    ):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.ref_model = ref_model
        self.val_model = val_model
        self.cpu_device = cpu_device
        self.tokenizer.pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
        self.tokenizer.padding_side = "left"
        self.generation_config = GenerationConfig(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            min_p=cfg.min_p,
            do_sample=True,
            max_new_tokens=cfg.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def __call__(
        self,
        model: AutoModelForCausalLM,
        entries: list[dict],
        dataset: ProceduralDataset,
        console: Console,
        rollout_rewards: list[torch.Tensor],
    ) -> ReplayBuffer | None:
        """Generate completions and package tensors for policy-gradient training."""
        # 1. Format prompts
        message_templates = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPTS["DeepSeekZero"]},
                    {"role": "user", "content": entry["question"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for entry in entries
        ]
        model_inputs = self.tokenizer(
            message_templates,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        ).to(model.device)

        # 2. Generate responses
        sequence_ids = model.generate(**model_inputs, generation_config=self.generation_config)
        completion_ids = sequence_ids[:, model_inputs["input_ids"].shape[1] :]
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # 3. Obtain the generated tokens only
        action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
        action_mask[:, model_inputs["input_ids"].shape[1] :] = True
        action_mask[sequence_ids == self.tokenizer.pad_token_id] = False
        action_mask = action_mask[:, 1:]
        lengths = action_mask.sum(dim=1).tolist()

        # 4. Compute rewards
        full_rewards, correctness_rewards = compute_rewards(
            dataset, completions, entries, self.cfg, lengths
        )
        rewards = torch.tensor(full_rewards, dtype=torch.float32, device=model.device).unsqueeze(-1)

        # 5. Compute attention mask
        attention_mask = sequence_ids != self.tokenizer.pad_token_id

        if self.cfg.loss == "dapo":
            rollout_output = self._filter_dapo_groups(
                sequence_ids=sequence_ids,
                action_mask=action_mask,
                attention_mask=attention_mask,
                rewards=rewards,
                correctness_rewards=correctness_rewards,
                console=console,
            )
            if rollout_output is None:
                return None
        else:
            rollout_output = Experience(
                sequence_ids=sequence_ids,
                action_mask=action_mask,
                attention_mask=attention_mask,
                rewards=rewards,
            )

        rollout_rewards.append(rollout_output.rewards.cpu())

        log_probs_old = compute_log_probs(
            model, rollout_output.sequence_ids, rollout_output.attention_mask
        )
        if self.cfg.beta and self.cfg.loss in ["ppo", "rloo", "reinforce"]:
            log_probs_ref = compute_log_probs(
                self.ref_model, rollout_output.sequence_ids, rollout_output.attention_mask
            )
            if log_probs_old is None or log_probs_ref is None:
                raise ValueError("beta>0 requires both rollout and reference log probs.")
            shaped_rewards = apply_reward_kl(
                rollout_output.rewards,
                log_probs_old,
                log_probs_ref,
                rollout_output.action_mask,
                self.cfg.beta,
                self.cfg.loss,
                self.cfg.kl_estimator,
            )
        else:
            log_probs_ref = None
            shaped_rewards = rollout_output.rewards
        values_old = compute_values(
            self.val_model, rollout_output.sequence_ids, rollout_output.attention_mask
        )
        advantages = compute_advantages(
            shaped_rewards,
            self.cfg.loss,
            rollout_output.action_mask,
            values_old,
            self.cfg.gamma,
            self.cfg.lam,
        )

        experience = Experience(
            sequence_ids=rollout_output.sequence_ids,
            attention_mask=rollout_output.attention_mask,
            action_mask=rollout_output.action_mask,
            advantages=advantages,
            log_probs_old=log_probs_old,
            log_probs_ref=log_probs_ref,
            values_old=values_old,
        ).to(self.cpu_device)
        replay_buffer = ReplayBuffer()
        replay_buffer.add(experience)
        return replay_buffer

    def _filter_dapo_groups(
        self,
        sequence_ids: torch.Tensor,
        action_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        rewards: torch.Tensor,
        correctness_rewards: list[float],
        console: Console,
    ) -> Experience | None:
        """Drop DAPO groups where every completion has min or max correctness."""
        group_size = self.cfg.num_rollouts
        batch_size = rewards.shape[0]

        if batch_size % group_size != 0:
            raise ValueError(f"Batch size {batch_size} is not divisible by {group_size=}.")

        num_groups = batch_size // group_size
        rewards_grouped = rewards.squeeze(-1).view(num_groups, group_size)
        correctness_rewards_tensor = torch.tensor(
            correctness_rewards, dtype=torch.float32, device=rewards.device
        )
        correctness_grouped = correctness_rewards_tensor.view(num_groups, group_size)
        sequence_ids_grouped = sequence_ids.view(num_groups, group_size, -1)
        action_mask_grouped = action_mask.view(num_groups, group_size, -1)
        attention_mask_grouped = attention_mask.view(num_groups, group_size, -1)

        all_min = (correctness_grouped == self.cfg.accuracy_min_reward).all(dim=1)
        all_max = (correctness_grouped == self.cfg.accuracy_max_reward).all(dim=1)
        valid_groups = ~(all_min | all_max)

        num_filtered = (~valid_groups).sum().item()
        num_all_min = all_min.sum().item()
        num_all_max = all_max.sum().item()
        console.print(
            f"[bold yellow]DAPO filtering:[/bold yellow] "
            f"filtered {num_filtered}/{num_groups} groups "
            f"({num_filtered / max(num_groups, 1):.2%}); "
            f"{num_all_min=}, {num_all_max=}"
        )

        if not valid_groups.any():
            correctness_rows = correctness_grouped.detach().cpu().tolist()
            console.print(
                "[bold red]For DAPO, all completions in this rollout batch were filtered out.[/bold red] "
                f"Here min={self.cfg.accuracy_min_reward} and max={self.cfg.accuracy_max_reward}. "
                f"Correctness rewards per group: {correctness_rows}. "
            )
            return None

        return Experience(
            sequence_ids=sequence_ids_grouped[valid_groups].reshape(-1, sequence_ids.shape[-1]),
            action_mask=action_mask_grouped[valid_groups].reshape(-1, action_mask.shape[-1]),
            attention_mask=attention_mask_grouped[valid_groups].reshape(
                -1, attention_mask.shape[-1]
            ),
            rewards=rewards_grouped[valid_groups].reshape(-1, 1),
        )


def decode_rollout_completions(tokenizer: Any, rollout_output: Experience) -> list[str]:
    """Decode generated tokens from a rollout experience batch."""
    target_ids = rollout_output.sequence_ids[:, 1:]
    completions = []
    for token_ids, action_mask in zip(target_ids, rollout_output.action_mask, strict=True):
        completion_ids = token_ids[action_mask.bool()]
        completions.append(tokenizer.decode(completion_ids, skip_special_tokens=True))
    return completions


def build_rollout_completions(
    tokenizer: Any, rollout_output: Experience, rollout_batch: list[dict]
) -> list[tuple[str, str, str]]:
    """Build rollout sample rows in the training loop."""
    completions = decode_rollout_completions(tokenizer, rollout_output)
    if len(completions) != len(rollout_batch):
        return []
    return [
        (entry["question"], entry["answer"], completion)
        for entry, completion in zip(rollout_batch, completions, strict=True)
    ]


def collect_rollout_batch(
    model,
    rollout_batch: list[dict],
    dataset: ProceduralDataset,
    rollout_engine: TransformerRolloutEngine,
    console: Console,
    replay_buffer: ReplayBuffer,
    rollout_rewards: list[torch.Tensor],
    rollout_completions: list[tuple[str, str, str]],
) -> int:
    """Generate one rollout batch and add valid experiences to the replay buffer."""
    with torch.no_grad():
        rollout_buffer = rollout_engine(
            model=model,
            entries=rollout_batch,
            dataset=dataset,
            console=console,
            rollout_rewards=rollout_rewards,
        )
        if rollout_buffer is None:
            return 0
        rollout_output = join_experiences_batch(rollout_buffer.buffer)
        rollout_completions.extend(
            build_rollout_completions(
                tokenizer=rollout_engine.tokenizer,
                rollout_output=rollout_output,
                rollout_batch=rollout_batch,
            )
        )
        replay_buffer.buffer.extend(rollout_buffer.buffer)
        return len(rollout_buffer)


def collect_rollouts_for_step(
    model,
    entries: list[dict],
    dataset: ProceduralDataset,
    dataloader_iter: Iterator[list[dict]],
    rollout_engine: TransformerRolloutEngine,
    console: Console,
    replay_buffer: ReplayBuffer,
    rollout_rewards: list[torch.Tensor],
    rollout_completions: list[tuple[str, str, str]],
) -> ReplayBuffer:
    """Collect rollouts for a single step.

    For DAPO, rejected groups are regenerated until the replay buffer reaches
    prompts_per_step * num_rollouts experiences.
    """
    cfg = rollout_engine.cfg
    rollout_batches = [
        list(rollout_batch) for rollout_batch in batched(entries, cfg.rollout_batch_size)
    ]
    next_prompt_batch: list[dict] = []
    expected_experiences = len(entries)

    if cfg.loss != "dapo":
        for rollout_batch in rollout_batches:
            collect_rollout_batch(
                model=model,
                rollout_batch=rollout_batch,
                dataset=dataset,
                rollout_engine=rollout_engine,
                console=console,
                replay_buffer=replay_buffer,
                rollout_rewards=rollout_rewards,
                rollout_completions=rollout_completions,
            )
        return replay_buffer

    rejected_prompt_keys: set[str] = set()
    pending_rollout_batches = list(rollout_batches)
    while len(replay_buffer) < expected_experiences:
        if not pending_rollout_batches:
            if not next_prompt_batch:
                try:
                    next_prompt_batch = next(dataloader_iter)
                except StopIteration as e:
                    raise RuntimeError(
                        "DAPO needs more prompts to replace filtered groups, but the dataloader "
                        "is exhausted."
                    ) from e
            while next_prompt_batch and next_prompt_batch[0]["question"] in rejected_prompt_keys:
                next_prompt_batch.pop(0)
            if not next_prompt_batch:
                continue
            next_entry = next_prompt_batch.pop(0)
            pending_rollout_batches.append([next_entry for _ in range(cfg.num_rollouts)])

        rollout_batch = pending_rollout_batches.pop(0)
        experiences_added = collect_rollout_batch(
            model=model,
            rollout_batch=rollout_batch,
            dataset=dataset,
            rollout_engine=rollout_engine,
            console=console,
            replay_buffer=replay_buffer,
            rollout_rewards=rollout_rewards,
            rollout_completions=rollout_completions,
        )
        if experiences_added == 0:
            rejected_prompt_keys.add(rollout_batch[0]["question"])
            console.print(
                f"[bold yellow]DAPO group rejected:[/bold yellow] "
                f"{experiences_added=}; continuing until "
                f"{len(replay_buffer)=}/{expected_experiences}"
            )
            continue

        console.print(
            f"[bold green]DAPO group accepted:[/bold green] "
            f"{experiences_added=}; continuing until "
            f"{len(replay_buffer)=}/{expected_experiences}"
        )
    return replay_buffer
