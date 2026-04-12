# Selection strategies for rejection sampling.
#
# Each strategy maps scored records to a list of (prompt, completion) training
# pairs. The two `top_*` strategies use the reward model; the `random_*`
# strategies are matched-size controls that ignore the reward.

import random

from .config import Config


Record = dict
TrainingPair = tuple[str, str]


def select_top_per_prompt(records: list[Record]) -> list[TrainingPair]:
    """Argmax completion per prompt."""
    pairs: list[TrainingPair] = []
    for rec in records:
        if not rec["completions"]:
            continue
        best_idx = max(range(len(rec["rewards"])), key=lambda j: rec["rewards"][j])
        pairs.append((rec["question"], rec["completions"][best_idx]))
    return pairs


def select_top_k_overall(records: list[Record], k: int) -> list[TrainingPair]:
    """Top-k completions across the full M x N reward matrix."""
    flat = [
        (float(reward), rec["question"], completion)
        for rec in records
        for completion, reward in zip(rec["completions"], rec["rewards"], strict=True)
    ]
    flat.sort(key=lambda item: item[0], reverse=True)
    return [(q, c) for _, q, c in flat[:k]]


def select_random_per_prompt(records: list[Record], seed: int) -> list[TrainingPair]:
    """Random baseline for top_per_prompt: pick one completion per prompt uniformly."""
    rng = random.Random(seed)
    pairs: list[TrainingPair] = []
    for rec in records:
        if not rec["completions"]:
            continue
        pick = rng.randrange(len(rec["completions"]))
        pairs.append((rec["question"], rec["completions"][pick]))
    return pairs


def select_random_k_overall(records: list[Record], k: int, seed: int) -> list[TrainingPair]:
    """Random baseline for top_k_overall: sample k pairs from the flat M x N pool."""
    rng = random.Random(seed)
    flat = [(rec["question"], c) for rec in records for c in rec["completions"]]
    return rng.sample(flat, min(k, len(flat)))


def select(records: list[Record], cfg: Config) -> list[TrainingPair]:
    """Dispatch to the strategy named in ``cfg.selection``."""
    strategy = cfg.selection.strategy
    if strategy == "top_per_prompt":
        return select_top_per_prompt(records)
    if strategy == "top_k_overall":
        return select_top_k_overall(records, cfg.selection.top_k)
    if strategy == "random_per_prompt":
        return select_random_per_prompt(records, cfg.seed)
    if strategy == "random_k_overall":
        return select_random_k_overall(records, cfg.selection.top_k, cfg.seed)
    raise ValueError(f"Unknown selection strategy: {strategy}")


def _chapter_worked_example() -> None:
    """Worked example from the chapter (M=5, N=4) as a sanity check."""
    reward_matrix = [
        [0.7, 0.3, 0.5, 0.2],
        [0.4, 0.8, 0.6, 0.5],
        [0.9, 0.3, 0.4, 0.7],
        [0.2, 0.5, 0.8, 0.6],
        [0.5, 0.4, 0.3, 0.6],
    ]
    records = [
        {
            "question": f"Q{i + 1}",
            "answer": "",
            "completions": [f"y_{i + 1},{j + 1}" for j in range(len(rewards))],
            "rewards": rewards,
        }
        for i, rewards in enumerate(reward_matrix)
    ]

    per_prompt = select_top_per_prompt(records)
    assert per_prompt == [
        ("Q1", "y_1,1"),
        ("Q2", "y_2,2"),
        ("Q3", "y_3,1"),
        ("Q4", "y_4,3"),
        ("Q5", "y_5,4"),
    ]
    print("top_per_prompt:", per_prompt)

    top5 = select_top_k_overall(records, k=5)
    assert top5 == [
        ("Q3", "y_3,1"),
        ("Q2", "y_2,2"),
        ("Q4", "y_4,3"),
        ("Q1", "y_1,1"),
        ("Q3", "y_3,4"),
    ]
    print("top_k_overall:", top5)

    # Random baselines: verify shape + reproducibility, not hard-coded RNG output.
    random_pairs = select_random_per_prompt(records, seed=42)
    assert [q for q, _ in random_pairs] == [f"Q{i + 1}" for i in range(5)]
    assert random_pairs == select_random_per_prompt(records, seed=42)
    print("random_per_prompt:", random_pairs)

    random_top5 = select_random_k_overall(records, k=5, seed=42)
    assert len(random_top5) == 5 and len(set(random_top5)) == 5
    assert random_top5 == select_random_k_overall(records, k=5, seed=42)
    print("random_k_overall:", random_top5)


if __name__ == "__main__":
    _chapter_worked_example()
