import random

import torch
from transformers import GenerationConfig

from .data import compute_score
from .utils import print_rollout_sample


def build_teacher_prompt(prompt: str, demo: str, feedback: str) -> str:
    """Build the teacher's feedback-conditioned reprompt around the original problem."""
    parts = [prompt]
    if demo:
        parts.append(f"\nCorrect solution:\n\n{demo}\n")
    if feedback:
        parts.append(
            f"\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback}\n"
        )
    parts.append("\nCorrectly solve the original question.\n")
    return "".join(parts)


def generate_batch(model, tokenizer, entry: dict, cfg) -> dict:
    """Generate and score ``num_rollouts`` rollouts for one problem."""
    device = model.device
    pad_id = tokenizer.pad_token_id
    gen_config = GenerationConfig(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        min_p=cfg.min_p,
        do_sample=True,
        max_new_tokens=cfg.max_new_tokens,
        pad_token_id=pad_id,
    )

    chat_kwargs = {}
    if not cfg.enable_thinking:
        chat_kwargs["enable_thinking"] = False
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": entry["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            **chat_kwargs,
        )
    ] * cfg.num_rollouts
    inputs = tokenizer(prompts, return_tensors="pt", return_attention_mask=True).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        student_ids = model.generate(**inputs, generation_config=gen_config)
    completion_ids = student_ids[:, prompt_len:]
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    scored = [compute_score(c, entry["tests"], cfg.max_tests) for c in completions]
    reward = torch.tensor([s["reward"] for s in scored], device=device)
    acc = torch.tensor([s["acc"] for s in scored], device=device)
    feedbacks = [s["feedback"] for s in scored]
    success_idx = [i for i, r in enumerate(reward) if r >= cfg.success_reward_threshold]

    action_mask = (completion_ids != pad_id).float()
    teacher_prompts = []
    for i in range(cfg.num_rollouts):
        others = [j for j in success_idx if j != i]
        demo = completions[others[0]] if others else ""
        feedback = feedbacks[i]
        if not demo and not feedback:
            action_mask[i] = 0.0
        prompt = build_teacher_prompt(entry["prompt"], demo, feedback)
        teacher_prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **chat_kwargs,
            )
        )

    teacher_prompts = tokenizer(teacher_prompts, return_tensors="pt", padding=True).to(device)
    teacher_ids = torch.cat([teacher_prompts["input_ids"], completion_ids], dim=1)

    print_rollout_sample(
        problem_id=entry["id"],
        reward=reward.mean().item(),
        acc=acc.mean().item(),
        feedback=next((f for f in feedbacks if f), ""),
        completion=random.choice(completions),
    )
    return {
        "s_ids": student_ids,
        "s_mask": (student_ids != pad_id).long(),
        "t_ids": teacher_ids,
        "t_mask": (teacher_ids != pad_id).long(),
        "action_mask": action_mask,
        "reward": reward,
        "acc": acc,
    }
