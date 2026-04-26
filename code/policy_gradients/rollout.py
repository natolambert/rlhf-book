from typing import Callable, NamedTuple

import torch
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.utils import SYSTEM_PROMPTS
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .config import Config


RewardFunction = Callable[
    [ProceduralDataset, list[str], list[dict], Config, list[int], Console],
    tuple[list[float], list[float]],
]


class RolloutOutput(NamedTuple):
    sequence_ids: torch.Tensor  # [B, T]      (LongTensor)
    action_mask: torch.Tensor  # [B, T-1]    (BoolTensor)
    attention_mask: torch.Tensor  # [B, T]      (BoolTensor)
    completions: list[str]  # length B
    entries: list[dict]  # length B, aligned to completions
    rewards: torch.Tensor  # [B, 1]      (FloatTensor)
    completions_lengths: list[int]  # length B


class TransformerRolloutEngine:
    def __init__(self, tokenizer: AutoTokenizer, cfg: Config):
        self.tokenizer = tokenizer
        self.cfg = cfg
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
        reward_fn: RewardFunction,
        console: Console,
    ) -> RolloutOutput | None:
        """Generate completions and package tensors for policy-gradient training."""
        valid_entries = entries
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
        # Per-completion generated length
        lengths = action_mask.sum(dim=1).tolist()

        rewards_list, correctness_rewards = reward_fn(
            dataset, completions, entries, self.cfg, lengths, console
        )
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=model.device).unsqueeze(-1)
        # 5. Compute attention mask
        attention_mask = sequence_ids != self.tokenizer.pad_token_id

        if self.cfg.loss == "dapo":
            filtered_output = self._filter_dapo_groups(
                sequence_ids=sequence_ids,
                action_mask=action_mask,
                attention_mask=attention_mask,
                rewards=rewards,
                correctness_rewards=correctness_rewards,
                completions=completions,
                entries=valid_entries,
                lengths=lengths,
                console=console,
            )
            if filtered_output is None:
                return None
            sequence_ids = filtered_output.sequence_ids
            action_mask = filtered_output.action_mask
            attention_mask = filtered_output.attention_mask
            completions = filtered_output.completions
            valid_entries = filtered_output.entries
            lengths = filtered_output.completions_lengths
            rewards = filtered_output.rewards

        return RolloutOutput(
            sequence_ids=sequence_ids,
            action_mask=action_mask,
            attention_mask=attention_mask,
            completions=completions,
            entries=valid_entries,
            completions_lengths=lengths,
            rewards=rewards,
        )

    def _filter_dapo_groups(
        self,
        sequence_ids: torch.Tensor,
        action_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        rewards: torch.Tensor,
        correctness_rewards: list[float],
        completions: list[str],
        entries: list[dict],
        lengths: list[int],
        console: Console,
    ) -> RolloutOutput | None:
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
            console.print("[bold red]All DAPO groups were filtered out.[/bold red]")
            return None

        kept_group_indices = valid_groups.nonzero(as_tuple=False).squeeze(-1).tolist()
        filtered_completions: list[str] = []
        filtered_entries: list[dict] = []
        filtered_lengths: list[int] = []
        for group_idx in kept_group_indices:
            start = group_idx * group_size
            end = start + group_size
            filtered_completions.extend(completions[start:end])
            filtered_entries.extend(entries[start:end])
            filtered_lengths.extend(lengths[start:end])

        return RolloutOutput(
            sequence_ids=sequence_ids_grouped[valid_groups].reshape(-1, sequence_ids.shape[-1]),
            action_mask=action_mask_grouped[valid_groups].reshape(-1, action_mask.shape[-1]),
            attention_mask=attention_mask_grouped[valid_groups].reshape(
                -1, attention_mask.shape[-1]
            ),
            completions=filtered_completions,
            entries=filtered_entries,
            completions_lengths=filtered_lengths,
            rewards=rewards_grouped[valid_groups].reshape(-1, 1),
        )
