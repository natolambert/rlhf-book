"""Reward-vs-correctness diagnostic for the rejection-sampling cache.

Loads a scored rollout JSONL (produced by ``preprocess.py``) and measures
how well the reward model separates correct from incorrect completions.

Three views:

1. **Reward histogram** -- do correct and incorrect completions occupy
   different parts of the reward distribution?
2. **Per-row selection winrate** -- on *decidable* prompts (those with a mix
   of correct and incorrect completions), how often does argmax(reward) pick
   a correct completion vs. a random baseline?
3. **Best-of-N sweep** -- for N = 1..K, what fraction of prompts land at
   least one correct completion in the top-N by reward?

The script also prints a summary that reports ``decidable_fraction``,
the share of prompts where within-row selection can actually matter.
When the policy is strong enough that most prompts are all-correct (or the
task is hard enough that most are all-wrong), the effective signal for
``top_per_prompt`` shrinks even if the RM is well-calibrated.

Requires the ``diagnostics`` optional dependency group::

    uv sync --extra diagnostics

Usage::

    uv run python -m rejection_sampling.diagnostics \\
        --cache rejection_sampling/output/rollouts/<hash>.jsonl \\
        --out-dir rejection_sampling/output/diagnostics

Figures are saved to ``--out-dir`` (created if missing).  A markdown summary
is printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Answer extraction (mirrors utils.py so this script has no import dependency
# on the rest of the package -- keeps it runnable standalone).
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _extract_answer(text: str) -> str | None:
    """Pull the final numeric answer from a GSM8K-style completion."""
    boxed = _BOXED_RE.findall(text)
    if boxed:
        inside = boxed[-1]
        numbers = _NUMBER_RE.findall(inside)
        if numbers:
            return numbers[-1].replace(",", "")
        return inside.strip().replace(",", "") or None
    numbers = _NUMBER_RE.findall(text)
    return numbers[-1].replace(",", "") if numbers else None


def _answers_match(predicted: str | None, gold: str) -> bool:
    """Numeric comparison with string fallback."""
    if predicted is None:
        return False
    try:
        return abs(float(predicted) - float(gold)) < 1e-6
    except ValueError:
        return predicted.strip() == gold.strip()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_rollouts(path: Path) -> pd.DataFrame:
    """Flatten the JSONL cache into one row per (prompt, completion)."""
    records: list[dict] = []
    with open(path) as f:
        for prompt_idx, line in enumerate(f):
            row = json.loads(line)
            gold = row["answer"]
            for comp_idx, (completion, reward) in enumerate(
                zip(row["completions"], row["rewards"], strict=True)
            ):
                predicted = _extract_answer(completion)
                records.append(
                    {
                        "prompt_idx": prompt_idx,
                        "completion_idx": comp_idx,
                        "reward": float(reward),
                        "gold": gold,
                        "predicted": predicted,
                        "correct": _answers_match(predicted, gold),
                    }
                )
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def decidable_fraction(df: pd.DataFrame) -> dict[str, int | float]:
    """Classify prompts into all-correct, all-wrong, and decidable (mixed)."""
    n_prompts = df["prompt_idx"].nunique()
    per_prompt = df.groupby("prompt_idx")["correct"]
    all_correct = int((per_prompt.sum() == per_prompt.size()).sum())
    none_correct = int((per_prompt.sum() == 0).sum())
    decidable = n_prompts - all_correct - none_correct
    return {
        "n_prompts": n_prompts,
        "all_correct": all_correct,
        "none_correct": none_correct,
        "decidable": decidable,
        "decidable_fraction": decidable / n_prompts if n_prompts else 0.0,
    }


def per_row_winrate(
    df: pd.DataFrame, rng: random.Random
) -> dict[str, float | int]:
    """Argmax-reward hit rate vs. random baseline on decidable rows."""
    top_hits = 0
    random_hits = 0
    decidable = 0
    for _, grp in df.groupby("prompt_idx"):
        correct_mask = grp["correct"].to_numpy()
        if correct_mask.all() or not correct_mask.any():
            continue
        decidable += 1
        top_idx = int(np.argmax(grp["reward"].to_numpy()))
        if correct_mask[top_idx]:
            top_hits += 1
        if correct_mask[rng.randrange(len(grp))]:
            random_hits += 1
    return {
        "decidable_prompts": decidable,
        "top_hit_rate": top_hits / decidable if decidable else float("nan"),
        "random_hit_rate": random_hits / decidable if decidable else float("nan"),
    }


def best_of_n_sweep(df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    """Coverage curves: fraction of prompts with >= 1 correct in top-N."""
    n_max = int(df.groupby("prompt_idx").size().min())
    results: list[dict] = []
    for n in range(1, n_max + 1):
        top_ok = 0
        rand_ok = 0
        total = 0
        for _, grp in df.groupby("prompt_idx"):
            total += 1
            rewards = grp["reward"].to_numpy()
            correct = grp["correct"].to_numpy()
            if correct[np.argsort(-rewards)[:n]].any():
                top_ok += 1
            if correct[rng.sample(range(len(grp)), n)].any():
                rand_ok += 1
        results.append(
            {
                "n": n,
                "top_n_hit_rate": top_ok / total,
                "random_n_hit_rate": rand_ok / total,
            }
        )
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_reward_histogram(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    correct = df.loc[df["correct"], "reward"]
    wrong = df.loc[~df["correct"], "reward"]
    bins = np.linspace(df["reward"].min(), df["reward"].max(), 50)
    ax.hist(
        wrong, bins=bins, alpha=0.6, density=True,
        label=f"Incorrect (n={len(wrong)})", color="#d62728",
    )
    ax.hist(
        correct, bins=bins, alpha=0.6, density=True,
        label=f"Correct (n={len(correct)})", color="#2ca02c",
    )
    ax.axvline(correct.mean(), color="#2ca02c", ls="--", lw=1.2, alpha=0.8)
    ax.axvline(wrong.mean(), color="#d62728", ls="--", lw=1.2, alpha=0.8)
    ax.set_xlabel("Reward score")
    ax.set_ylabel("Density")
    ax.set_title("Reward distribution: correct vs. incorrect completions")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "reward_by_correctness.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_winrate(stats: dict, out_dir: Path) -> Path:
    labels = ["Top-reward (argmax)", "Random baseline"]
    values = [stats["top_hit_rate"], stats["random_hit_rate"]]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, values, color=["#1f77b4", "#aec7e8"])
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"Hit rate on decidable prompts (n={stats['decidable_prompts']})")
    ax.set_title("Per-row selection: does argmax(reward) pick correct?")
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v + 0.01,
            f"{v:.1%}", ha="center", va="bottom", fontsize=10,
        )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "per_row_winrate.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_best_of_n(sweep: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        sweep["n"], sweep["top_n_hit_rate"],
        marker="o", label="Top-N by reward", color="#1f77b4", lw=2,
    )
    ax.plot(
        sweep["n"], sweep["random_n_hit_rate"],
        marker="s", label="Random-N", color="#aec7e8", lw=2, ls="--",
    )
    ax.set_xlabel("N (completions considered per prompt)")
    ax.set_ylabel("Fraction of prompts with >= 1 correct in top-N")
    ax.set_title("Best-of-N coverage: reward-ranked vs. random")
    ax.set_xticks(sweep["n"])
    ax.set_ylim(0.5, 1.02)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "best_of_n_sweep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(df: pd.DataFrame, dec: dict, winrate: dict, sweep: pd.DataFrame) -> None:
    n_completions = len(df)
    n_per_prompt = n_completions // dec["n_prompts"]
    pct_correct = df["correct"].mean()

    correct_r = df.loc[df["correct"], "reward"]
    wrong_r = df.loc[~df["correct"], "reward"]

    print("# Reward vs. correctness diagnostic\n")
    print(f"- Prompts: **{dec['n_prompts']}**, completions/prompt: **{n_per_prompt}**")
    print(f"- Overall correctness: **{pct_correct:.1%}**")
    print(f"- All-correct prompts: **{dec['all_correct']}**, all-wrong: **{dec['none_correct']}**, decidable: **{dec['decidable']}**")
    print(f"- **decidable_fraction: {dec['decidable_fraction']:.3f}**\n")

    print("## Reward scores\n")
    print(f"- Correct:   mean={correct_r.mean():+.3f}, median={correct_r.median():+.3f}")
    print(f"- Incorrect: mean={wrong_r.mean():+.3f}, median={wrong_r.median():+.3f}")
    print(f"- Gap: **{correct_r.mean() - wrong_r.mean():+.3f}**\n")

    print("## Per-row selection (decidable prompts)\n")
    print(f"- argmax(reward) hit rate: **{winrate['top_hit_rate']:.1%}**")
    print(f"- random baseline:         **{winrate['random_hit_rate']:.1%}**")
    print(f"- gap: **{winrate['top_hit_rate'] - winrate['random_hit_rate']:+.1%}**\n")

    print("## Best-of-N sweep\n")
    print(f"| {'n':>3} | {'top_n':>8} | {'random_n':>8} |")
    print(f"|{'---':->5}|{'---':->10}|{'---':->10}|")
    for _, row in sweep.iterrows():
        print(f"| {int(row['n']):>3} | {row['top_n_hit_rate']:>8.3f} | {row['random_n_hit_rate']:>8.3f} |")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reward-vs-correctness diagnostic for rejection-sampling caches."
    )
    parser.add_argument(
        "--cache", type=str, required=True,
        help="Path to the scored rollout JSONL (from preprocess.py).",
    )
    parser.add_argument(
        "--out-dir", type=str, default="rejection_sampling/output/diagnostics",
        help="Directory for output figures (created if missing).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cache_path = Path(args.cache)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    df = load_rollouts(cache_path)
    dec = decidable_fraction(df)
    winrate = per_row_winrate(df, rng)
    sweep = best_of_n_sweep(df, rng)

    print_summary(df, dec, winrate, sweep)

    for path in [
        plot_reward_histogram(df, out_dir),
        plot_winrate(winrate, out_dir),
        plot_best_of_n(sweep, out_dir),
    ]:
        print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
