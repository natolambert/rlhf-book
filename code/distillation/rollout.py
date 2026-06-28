import random

import torch
from reasoning_gym.dataset import ProceduralDataset
from transformers import GenerationConfig

from .data import compute_score
from .utils import print_rollout_sample


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the "
    "Assistant solves it. The Assistant provides the final solution enclosed within "
    "<answer> </answer> tags, i.e., <answer> solution here </answer>. Provide only the "
    "final answer inside the tags. When an example is provided, you should strictly "
    "follow the format of the output/answer in that example."
)


def build_teacher_prompt(question: str, demo: str) -> str:
    parts = [
        question,
        f"Correct solution:\n\n{demo}",
        "Correctly solve the original question.",
    ]
    return "\n\n".join(parts)


def _apply_template(tokenizer, content: str, chat_kwargs: dict) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        tokenize=False,
        add_generation_prompt=True,
        **chat_kwargs,
    )


def generate_batch(
    model, tokenizer, dataset: ProceduralDataset, entry: dict, cfg, idx: int = 0, total: int = 1
) -> dict | None:
    device = model.device
    pad_id = tokenizer.pad_token_id
    gen_config = GenerationConfig(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        min_p=cfg.min_p,
        repetition_penalty=cfg.repetition_penalty,
        do_sample=True,
        max_new_tokens=cfg.max_new_tokens,
        pad_token_id=pad_id,
    )

    chat_kwargs = {}
    if not cfg.enable_thinking:
        chat_kwargs["enable_thinking"] = False
    prompts = [_apply_template(tokenizer, entry["question"], chat_kwargs)] * cfg.num_rollouts
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        return_attention_mask=True,
        truncation=True,
        max_length=cfg.max_prompt_len,
    ).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        student_ids = model.generate(**inputs, generation_config=gen_config)
    completion_ids = student_ids[:, prompt_len:]
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    reward = torch.tensor([compute_score(c, dataset, entry) for c in completions], device=device)
    success_idx = [i for i, r in enumerate(reward) if r >= cfg.success_reward_threshold]

    # No correct demonstration in the group: nothing to distil from, skip the prompt.
    if not success_idx:
        print_rollout_sample(
            problem_id=entry["question"],
            reward=reward.mean().item(),
            completion=random.choice(completions),
            idx=idx,
            total=total,
            skipped=True,
        )
        return None

    action_mask = (completion_ids != pad_id).float()
    demo = completions[success_idx[0]]
    teacher_prompt = _apply_template(
        tokenizer, build_teacher_prompt(entry["question"], demo), chat_kwargs
    )
    teacher_prefix = tokenizer(
        teacher_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.max_reprompt_len,
    ).to(device)["input_ids"]
    teacher_ids = torch.cat([teacher_prefix.expand(cfg.num_rollouts, -1), completion_ids], dim=1)

    print_rollout_sample(
        problem_id=entry["question"],
        reward=reward.mean().item(),
        completion=random.choice(completions),
        idx=idx,
        total=total,
    )
    return {
        "s_ids": student_ids,
        "s_mask": (student_ids != pad_id).long(),
        "t_ids": teacher_ids,
        "t_mask": (teacher_ids != pad_id).long(),
        "action_mask": action_mask,
        "reward": reward,
    }
