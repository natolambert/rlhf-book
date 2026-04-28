from itertools import islice
from typing import Iterator

import torch
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.utils import SYSTEM_PROMPTS
from rich.console import Console
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .buffer import Experience, ReplayBuffer
from .config import Config
from .utils import (
    apply_reward_kl,
    compute_advantages,
    compute_log_probs,
    compute_rewards,
    compute_values,
    print_rollout_sample,
)


class TransformerRolloutEngine:
    """Iterable engine that yields one ReplayBuffer per training step."""

    def __init__(
        self,
        cfg: Config,
        dataset: ProceduralDataset,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        ref_model: AutoModelForCausalLM | None,
        val_model: AutoModelForCausalLM | None,
        cpu_device: torch.device,
        console: Console,
    ):
        self.cfg = cfg
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        self.val_model = val_model
        self.cpu_device = cpu_device
        self.console = console

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

        self._dataloader = DataLoader(
            dataset=dataset,
            batch_size=cfg.prompts_per_step,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            collate_fn=lambda x: x,
        )
        self._prompts = (prompt for batch in self._dataloader for prompt in batch)

    def __len__(self) -> int:
        return len(self._dataloader)

    def __iter__(self) -> Iterator[ReplayBuffer]:
        while True:
            buffer = self._step()
            if len(buffer) == 0:
                return
            yield buffer

    def _step(self) -> ReplayBuffer:
        buffer = ReplayBuffer()
        expected = self.cfg.prompts_per_step * self.cfg.num_rollouts

        while len(buffer) < expected:
            remaining = expected - len(buffer)
            n_prompts = remaining // self.cfg.num_rollouts
            prompts = list(islice(self._prompts, n_prompts))
            if not prompts:
                break

            with torch.no_grad():
                exp = self._generate_experience(prompts)
                exp = self._filter(exp)

            buffer.add(exp)

        print_rollout_sample(self.console, self.tokenizer, buffer)
        return buffer

    def _generate_experience(self, prompts: list[dict]) -> Experience:
        entries = [p for p in prompts for _ in range(self.cfg.num_rollouts)]

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
        ).to(self.model.device)

        sequence_ids = self.model.generate(**model_inputs, generation_config=self.generation_config)
        completion_ids = sequence_ids[:, model_inputs["input_ids"].shape[1] :]
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
        action_mask[:, model_inputs["input_ids"].shape[1] :] = True
        action_mask[sequence_ids == self.tokenizer.pad_token_id] = False
        action_mask = action_mask[:, 1:]
        attention_mask = sequence_ids != self.tokenizer.pad_token_id
        lens = action_mask.sum(dim=1).tolist()

        rewards, correctness = compute_rewards(entries, completions, lens, self.dataset, self.cfg)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.model.device).unsqueeze(-1)
        correctness = torch.tensor(correctness, dtype=torch.float32, device=self.model.device)

        log_probs_old = compute_log_probs(self.model, sequence_ids, attention_mask)
        log_probs_ref = compute_log_probs(self.ref_model, sequence_ids, attention_mask)
        values_old = compute_values(self.val_model, sequence_ids, attention_mask)

        rewards = apply_reward_kl(
            rewards,
            log_probs_old,
            log_probs_ref,
            action_mask,
            self.cfg.beta,
            self.cfg.loss,
            self.cfg.kl_estimator,
        )
        advantages = compute_advantages(
            rewards, self.cfg.loss, action_mask, values_old, self.cfg.gamma, self.cfg.lam
        )

        return Experience(
            sequence_ids=sequence_ids,
            attention_mask=attention_mask,
            action_mask=action_mask,
            rewards=rewards,
            correctness=correctness,
            log_probs_old=log_probs_old,
            log_probs_ref=log_probs_ref,
            values_old=values_old,
            advantages=advantages,
        ).to(self.cpu_device)

    def _filter(self, exp: Experience) -> Experience:
        if self.cfg.loss == "dapo":
            return self._dapo_filter(exp)
        return exp

    def _dapo_filter(self, exp: Experience) -> Experience:
        """Drop groups of num_rollouts where every completion has min or max correctness."""
        group_size = self.cfg.num_rollouts
        num_groups = exp.correctness.shape[0] // group_size

        correctness_grouped = exp.correctness.view(num_groups, group_size)
        all_min = (correctness_grouped == self.cfg.accuracy_min_reward).all(dim=1)
        all_max = (correctness_grouped == self.cfg.accuracy_max_reward).all(dim=1)
        valid = ~(all_min | all_max)

        def slice_field(t: torch.Tensor | None) -> torch.Tensor | None:
            if t is None:
                return None
            return t.view(num_groups, group_size, *t.shape[1:])[valid].reshape(-1, *t.shape[1:])

        return Experience(
            sequence_ids=slice_field(exp.sequence_ids),
            attention_mask=slice_field(exp.attention_mask),
            action_mask=slice_field(exp.action_mask),
            rewards=slice_field(exp.rewards),
            correctness=slice_field(exp.correctness),
            log_probs_old=slice_field(exp.log_probs_old),
            log_probs_ref=slice_field(exp.log_probs_ref),
            values_old=slice_field(exp.values_old),
            advantages=slice_field(exp.advantages),
        )
