from typing import Iterator

import torch
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.utils import SYSTEM_PROMPTS
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
    ):
        self.cfg = cfg
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        self.val_model = val_model

        self.cpu_device = torch.device("cpu")

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
        self._entries = (entry for batch in self._dataloader for entry in batch)

    def __len__(self) -> int:
        return len(self._dataloader)

    def __iter__(self) -> Iterator[ReplayBuffer]:
        while True:
            buffer = self._step()
            if len(buffer) == 0:
                return
            yield buffer

    def _step(self) -> ReplayBuffer:
        self.model.eval()
        if self.val_model is not None:
            self.val_model.eval()

        buffer = ReplayBuffer()
        expected = self.cfg.prompts_per_step * self.cfg.num_rollouts
        while len(buffer) < expected:
            entry = next(self._entries, None)
            if entry is None:
                break
            with torch.no_grad():
                exp = self._generate_experience(entry)
                exp = self._filter(exp)
            if exp is not None:
                buffer.add(exp)

        print_rollout_sample(buffer, self.tokenizer)
        return buffer

    def _generate_experience(self, entries: dict | list[dict]) -> Experience:
        entries = [entries] if isinstance(entries, dict) else entries
        entries = [e for e in entries for _ in range(self.cfg.num_rollouts)]

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
        correctness = torch.tensor(
            correctness, dtype=torch.float32, device=self.model.device
        ).unsqueeze(-1)

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

    def _filter(self, exp: Experience) -> Experience | None:
        if self.cfg.loss == "dapo":
            return self._dapo_filter(exp)
        return exp

    def _dapo_filter(self, exp: Experience) -> Experience | None:
        """Drop the group if every completion has min or max correctness."""
        all_min = (exp.correctness == self.cfg.accuracy_min_reward).all()
        all_max = (exp.correctness == self.cfg.accuracy_max_reward).all()
        if all_min or all_max:
            return None
        return exp
