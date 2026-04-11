# Selection strategies for rejection sampling
#
# Each strategy takes a list of scored-rollout records
#
#     {"question": str, "answer": str, "completions": [str, ...], "rewards": [float, ...]}
#
# and returns a flat list of (prompt, completion) training pairs. Two strategies
# implement the selection functions defined in the chapter
# (@eq:rs_selection_per_prompt, @eq:rs_topk_selection); the third is a
# random-per-prompt control that ignores the reward model entirely and acts as
# a fair baseline at the same dataset size as ``top_per_prompt``.

import random

from .config import Config


Record = dict
TrainingPair = tuple[str, str]


def select_top_per_prompt(records: list[Record]) -> list[TrainingPair]:
    """Keep the single highest-reward completion per prompt (eq. rs_selection_per_prompt).

    Yields exactly len(records) training pairs.
    """
    pairs: list[TrainingPair] = []
    for rec in records:
        completions = rec["completions"]
        rewards = rec["rewards"]
        if not completions:
            continue
        best_idx = max(range(len(rewards)), key=lambda j: rewards[j])
        pairs.append((rec["question"], completions[best_idx]))
    return pairs


def select_top_k_overall(records: list[Record], k: int) -> list[TrainingPair]:
    """Keep the top-k completions across the entire (M x N) reward matrix.

    Implements eq. rs_topk_selection. Ties are broken by the natural order of
    the flattened records. Can include multiple completions from the same
    prompt.
    """
    flat: list[tuple[float, int, str, str]] = []
    for rec in records:
        question = rec["question"]
        for j, (completion, reward) in enumerate(zip(rec["completions"], rec["rewards"], strict=True)):
            flat.append((float(reward), j, question, completion))

    flat.sort(key=lambda item: item[0], reverse=True)
    top = flat[:k]
    return [(question, completion) for _, _, question, completion in top]


def select_random_k_overall(records: list[Record], k: int, seed: int) -> list[TrainingPair]:
    """Random-K-from-flat control: pick k pairs uniformly from the M x N pool.

    Fair baseline for ``select_top_k_overall`` — same dataset size (k pairs),
    same flat-selection structure (can draw multiple completions from the same
    prompt), but ignores the reward vector entirely. Seeded via ``cfg.seed``
    so the draw is reproducible across runs.
    """
    rng = random.Random(seed)
    flat: list[TrainingPair] = []
    for rec in records:
        question = rec["question"]
        for completion in rec["completions"]:
            flat.append((question, completion))
    return rng.sample(flat, min(k, len(flat)))


def select_random_per_prompt(records: list[Record], seed: int) -> list[TrainingPair]:
    """Random-1-per-prompt control: pick one completion uniformly at random per prompt.

    Fair baseline for ``select_top_per_prompt`` — same dataset size (exactly
    M pairs), same prompt coverage, same 1-of-N structure, but ignores the
    reward vector entirely. Comparing the two isolates whether reward-model
    filtering actually beats a coin flip at matched sample budget. Seeded via
    ``cfg.seed`` so the draw is reproducible across runs.
    """
    rng = random.Random(seed)
    pairs: list[TrainingPair] = []
    for rec in records:
        completions = rec["completions"]
        if not completions:
            continue
        pick = rng.randrange(len(completions))
        pairs.append((rec["question"], completions[pick]))
    return pairs


def select(records: list[Record], cfg: Config) -> list[TrainingPair]:
    """Dispatch to the selection strategy specified in ``cfg.selection``."""
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
    """Run the chapter.md selection example (M=5, N=4) and assert the results.

    Reward matrix from @eq:rs_example_matrix:

        R = [[0.7, 0.3, 0.5, 0.2],
             [0.4, 0.8, 0.6, 0.5],
             [0.9, 0.3, 0.4, 0.7],
             [0.2, 0.5, 0.8, 0.6],
             [0.5, 0.4, 0.3, 0.6]]
    """
    reward_matrix = [
        [0.7, 0.3, 0.5, 0.2],
        [0.4, 0.8, 0.6, 0.5],
        [0.9, 0.3, 0.4, 0.7],
        [0.2, 0.5, 0.8, 0.6],
        [0.5, 0.4, 0.3, 0.6],
    ]
    records: list[Record] = []
    for i, rewards in enumerate(reward_matrix):
        question = f"Q{i + 1}"
        completions = [f"y_{i + 1},{j + 1}" for j in range(len(rewards))]
        records.append({"question": question, "answer": "", "completions": completions, "rewards": rewards})

    # Top per prompt — expected S(R) = [1, 2, 1, 3, 4] (1-indexed in chapter).
    per_prompt = select_top_per_prompt(records)
    expected_per_prompt = [
        ("Q1", "y_1,1"),  # argmax row 1 -> col 1
        ("Q2", "y_2,2"),  # argmax row 2 -> col 2
        ("Q3", "y_3,1"),  # argmax row 3 -> col 1
        ("Q4", "y_4,3"),  # argmax row 4 -> col 3
        ("Q5", "y_5,4"),  # argmax row 5 -> col 4
    ]
    assert per_prompt == expected_per_prompt, (
        f"select_top_per_prompt mismatch:\n  got={per_prompt}\n  want={expected_per_prompt}"
    )
    print("top_per_prompt: OK")
    for pair in per_prompt:
        print(f"  {pair}")

    # Top-5 overall — chapter indices: [8, 5, 14, 0, 11] (0-indexed flattened).
    # Index 8  -> (prompt 3, completion 1) reward 0.9
    # Index 5  -> (prompt 2, completion 2) reward 0.8
    # Index 14 -> (prompt 4, completion 3) reward 0.8
    # Index 0  -> (prompt 1, completion 1) reward 0.7
    # Index 11 -> (prompt 3, completion 4) reward 0.7
    top5 = select_top_k_overall(records, k=5)
    expected_top5 = [
        ("Q3", "y_3,1"),
        ("Q2", "y_2,2"),
        ("Q4", "y_4,3"),
        ("Q1", "y_1,1"),
        ("Q3", "y_3,4"),
    ]
    assert top5 == expected_top5, (
        f"select_top_k_overall mismatch:\n  got={top5}\n  want={expected_top5}"
    )
    print("top_k_overall (k=5): OK")
    for pair in top5:
        print(f"  {pair}")

    # Random-K overall: same shape as top_k_overall (k flat pairs, can repeat
    # prompts), but the k picks come from rng.sample rather than reward sort.
    # Verify structure + reproducibility rather than hard-coding RNG output.
    random_top5 = select_random_k_overall(records, k=5, seed=42)
    assert len(random_top5) == 5, f"expected 5 pairs, got {len(random_top5)}"
    valid_flat_pairs = {
        (f"Q{i + 1}", f"y_{i + 1},{j + 1}") for i in range(5) for j in range(4)
    }
    for pair in random_top5:
        assert pair in valid_flat_pairs, f"random pick {pair!r} not in M x N pool"
    # rng.sample draws without replacement, so no duplicates within one call.
    assert len(set(random_top5)) == 5, f"random_k_overall drew duplicates: {random_top5}"
    assert random_top5 == select_random_k_overall(records, k=5, seed=42)
    print("random_k_overall (k=5): OK")
    for pair in random_top5:
        print(f"  {pair}")

    # Random per prompt: same shape as top_per_prompt (one pair per prompt),
    # but the specific completion is chosen by a seeded RNG — not the reward.
    # Verify structure rather than hard-coding RNG output: exactly M pairs,
    # each question covered once, each picked completion belongs to its row.
    random_pairs = select_random_per_prompt(records, seed=42)
    assert len(random_pairs) == 5, f"expected 5 pairs, got {len(random_pairs)}"
    seen_questions = [q for q, _ in random_pairs]
    assert seen_questions == [f"Q{i + 1}" for i in range(5)], (
        f"random_per_prompt must cover every prompt in order, got {seen_questions}"
    )
    valid_completions_per_row = {
        f"Q{i + 1}": {f"y_{i + 1},{j + 1}" for j in range(4)} for i in range(5)
    }
    for question, completion in random_pairs:
        assert completion in valid_completions_per_row[question], (
            f"random pick {completion!r} not in row for {question}"
        )
    # Also confirm reproducibility: a second call with the same seed is identical.
    assert random_pairs == select_random_per_prompt(records, seed=42)
    print("random_per_prompt: OK")
    for pair in random_pairs:
        print(f"  {pair}")


if __name__ == "__main__":
    _chapter_worked_example()
