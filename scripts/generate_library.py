"""Utility for precomputing RLHF library data for the static site."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Literal

from datasets import load_dataset


Variant = Literal["sft", "rlhf"]


def _variant_for_model(model_name: str) -> Variant:
    """Label a model as SFT or RLHF based on its name."""
    lowered = model_name.lower()
    if "dpo" in lowered or "instruct" in lowered:
        return "rlhf"
    return "sft"


def _base_model_id(model_name: str) -> str:
    """Strip variant suffixes so SFT and RLHF versions share a key."""
    base = model_name
    replacements = (
        ("-SFT-hf", ""),
        ("-Instruct-hf", ""),
        ("-Instruct", ""),
        ("-SFT", ""),
        ("-DPO", ""),
        ("-dpo-", "-"),
    )
    for old, new in replacements:
        base = base.replace(old, new)
    return base


def _display_label(base_model_id: str) -> str:
    """Make a short label for dropdowns."""
    return base_model_id.split("/")[-1].replace("-", " ")


def _format_prompt_text(messages: Iterable[Dict[str, str]]) -> str:
    """Return a single string that mirrors what will be shown in the UI."""
    lines: List[str] = []
    for message in messages:
        role = message.get("role", "user").capitalize()
        content = message.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


@dataclass
class CompletionRecord:
    completion_id: str
    text: str


@dataclass
class ModelPair:
    pair_id: str
    display_name: str
    sft_model: str | None = None
    rlhf_model: str | None = None

    def mark(self, variant: Variant, model_name: str) -> None:
        if variant == "sft":
            self.sft_model = model_name
        else:
            self.rlhf_model = model_name

    def validate(self) -> None:
        if not self.sft_model or not self.rlhf_model:
            raise ValueError(
                f"Missing SFT or RLHF models for pair '{self.pair_id}'. "
                f"Found SFT={self.sft_model}, RLHF={self.rlhf_model}."
            )


@dataclass
class PromptRecord:
    prompt_id: int
    messages: List[Dict[str, str]]
    display_text: str


def build_payload() -> Dict[str, object]:
    """Build a JSON payload that powers the RLHF library page."""
    dataset = load_dataset("natolambert/rlhf-library", split="train")

    prompts: Dict[int, PromptRecord] = {}
    model_pairs: Dict[str, ModelPair] = {}
    completions: Dict[int, Dict[str, Dict[Variant, List[CompletionRecord]]]] = defaultdict(
        lambda: defaultdict(lambda: {"sft": [], "rlhf": []})
    )

    for row in dataset:
        prompt_idx_str, completion_idx_str = row["id"].split("-")
        prompt_idx = int(prompt_idx_str)
        completion_idx = int(completion_idx_str)

        if prompt_idx not in prompts:
            messages = row["instruction"]
            prompts[prompt_idx] = PromptRecord(
                prompt_id=prompt_idx,
                messages=messages,
                display_text=_format_prompt_text(messages),
            )

        variant = _variant_for_model(row["model"])
        base_id = _base_model_id(row["model"])
        pair = model_pairs.setdefault(
            base_id,
            ModelPair(pair_id=base_id, display_name=_display_label(base_id)),
        )
        pair.mark(variant, row["model"])

        completions[prompt_idx][base_id][variant].append(
            (completion_idx, CompletionRecord(completion_id=row["id"], text=row["completion"]))
        )

    # Sort completions and drop helper indices
    serialisable_completions: Dict[str, Dict[str, Dict[Variant, List[Dict[str, str]]]]] = {}
    for prompt_idx, pairs in completions.items():
        serialisable_completions[str(prompt_idx)] = {}
        for base_id, variant_map in pairs.items():
            serialisable_completions[str(prompt_idx)][base_id] = {}
            for variant, entries in variant_map.items():
                sorted_entries = sorted(entries, key=lambda item: item[0])
                serialisable_completions[str(prompt_idx)][base_id][variant] = [
                    asdict(entry) for _, entry in sorted_entries
                ]

    # Ensure we have matching pairs before writing anything.
    ordered_pairs: List[ModelPair] = []
    for pair in sorted(model_pairs.values(), key=lambda item: item.display_name.lower()):
        pair.validate()
        ordered_pairs.append(pair)

    ordered_prompts = [prompts[idx] for idx in sorted(prompts)]

    payload = {
        "model_pairs": [asdict(pair) for pair in ordered_pairs],
        "prompts": [asdict(prompt) for prompt in ordered_prompts],
        "completions": serialisable_completions,
    }
    return payload


def write_payload(payload: Dict[str, object], *output_paths: str | Path) -> None:
    """Write the payload to each of the provided paths."""
    for raw_path in output_paths:
        if raw_path is None:
            continue
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))


def generate_library(*output_paths: str | Path) -> Dict[str, object]:
    payload = build_payload()
    if output_paths:
        write_payload(payload, *output_paths)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RLHF library payload")
    parser.add_argument(
        "--output",
        default="data/library.json",
        help="Primary output path for the processed dataset",
    )
    parser.add_argument(
        "--html-output",
        default="build/html/data/library.json",
        help="Optional path under the published site for direct consumption",
    )

    args = parser.parse_args()
    data = generate_library(args.output, args.html_output)
    print(
        "Wrote library payload to:"
        f"\n - {args.output}"
        f"\n - {args.html_output}"
        f"\nPrompts: {len(data['prompts'])} | Model pairs: {len(data['model_pairs'])}"
    )
