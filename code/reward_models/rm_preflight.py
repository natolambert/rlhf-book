#!/usr/bin/env python3
"""Pre-flight checklist for the reward-model exercises (ORM / PRM / Preference RM).

Unlike ``scripts/tokenizer_preflight.py`` (which probes the SFT chat-template
path), the reward models do NOT use a chat template. Each one builds its own
plain-text inputs and a custom label scheme:

- Preference RM : pairwise (chosen, rejected) full sequences, scalar reward at
                  the last real token, Bradley-Terry loss.
- ORM           : (prompt, completion) with a per-token 0/1 label on completion
                  tokens, BCE loss.
- PRM           : (problem, steps) with a 3-class label on each step terminator
                  token only, cross-entropy loss.

This script exercises the *real* ``build_*`` / ``pack_*`` code paths on a few
samples and prints what actually comes out — token ids, decoded text, label
masks, class balance — so you can see exactly which part would break (dataset
schema drift, tokenizer special tokens, label alignment) before spending GPU
time. Read-only: it loads tiny slices and never trains.

Usage (from code/):
    uv run python -m reward_models.rm_preflight                # all three
    uv run python -m reward_models.rm_preflight --only orm     # one at a time
"""

import argparse
import traceback

from datasets import load_dataset

from reward_models import train_orm, train_preference_rm, train_prm
from reward_models.base import load_tokenizer


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def describe_tokenizer(model_id: str) -> None:
    """Special-token report — also answers the 'chat special tokens' question."""
    tok = load_tokenizer(model_id)
    print(f"tokenizer: {type(tok).__name__}  vocab={len(tok)}")
    print(
        f"  eos={tok.eos_token!r}(id={tok.eos_token_id})  "
        f"pad={tok.pad_token!r}(id={tok.pad_token_id})  "
        f"bos={tok.bos_token!r}(id={tok.bos_token_id})"
    )
    print(f"  pad_token_id == eos_token_id ? {tok.pad_token_id == tok.eos_token_id}")
    # Are ChatML role markers known to the BASE tokenizer? (Q3 from the user.)
    for marker in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
        ids = tok(marker, add_special_tokens=False)["input_ids"]
        atomic = "ATOMIC single id" if len(ids) == 1 else f"{len(ids)} BPE pieces"
        print(f"  {marker!r:>16} -> ids={ids} ({atomic})")
    return tok


def preflight_preference() -> None:
    hr("PREFERENCE RM  (Bradley-Terry, UltraFeedback)")
    model_id = train_preference_rm.DEFAULT_MODEL_ID
    print(f"model_id (default): {model_id}")
    tok = describe_tokenizer(model_id)

    ds_name = train_preference_rm.DEFAULT_DATASET
    print(f"\ndataset: {ds_name}")
    raw = load_dataset(ds_name, split="train[:3]")
    print(f"columns: {raw.column_names}")
    ex = raw[0]
    for k in ("prompt", "chosen", "rejected"):
        v = ex.get(k)
        print(f"  {k!r}: type={type(v).__name__}", end="")
        if isinstance(v, list):
            print(f" len={len(v)} first_elem={v[0] if v else None!r}")
        else:
            print(f" value={str(v)[:80]!r}")

    # Run the REAL builder on a couple rows and show what the model will see.
    data = train_preference_rm.build_preference_dataset(tok, limit=4, max_length=512)
    print(f"\nbuild_preference_dataset -> {len(data)} pairs")
    row = data[0]
    print(f"  chosen_ids len={len(row['chosen_ids'])}  rejected_ids len={len(row['rejected_ids'])}")
    print(f"  chosen decoded (head): {tok.decode(row['chosen_ids'][:60])!r}")
    print("  reward is read at the LAST non-pad token of each sequence (scalar head, dim=1)")


def preflight_orm() -> None:
    hr("ORM  (Outcome RM, GSM8K, per-token BCE)")
    model_id = train_orm.DEFAULT_MODEL_ID
    print(f"model_id (default): {model_id}")
    tok = describe_tokenizer(model_id)

    print(f"\ndataset: {train_orm.DEFAULT_DATASET} (config 'main')")
    raw = load_dataset(train_orm.DEFAULT_DATASET, "main", split="train[:2]")
    print(f"columns: {raw.column_names}")
    ans = raw[0]["answer"]
    print(f"  raw answer tail: {ans.split(chr(10))[-1]!r}")
    print(f"  parse_answer -> {train_orm.parse_answer(ans)}")

    data = train_orm.build_orm_dataset(tok, limit=4, seed=7)
    print(f"\nbuild_orm_dataset -> {len(data)} examples (2 per question: correct=1, corrupted=0)")
    pos, neg = data[0], data[1]
    for tag, row in (("POS(label=1)", pos), ("NEG(label=0)", neg)):
        labels = row["labels"]
        n_masked = sum(1 for x in labels if x == -100)
        labeled = [x for x in labels if x != -100]
        uniq = sorted(set(labeled))
        print(
            f"  {tag}: len={len(labels)}  prompt_masked(-100)={n_masked}  "
            f"completion_labeled={len(labeled)}  label_values={uniq}"
        )
    print(f"  completion ends with eos {tok.eos_token!r}; loss = BCE over completion tokens only")


def preflight_prm() -> None:
    hr("PRM  (Process RM, PRM800K, per-step 3-class CE)")
    model_id = train_prm.DEFAULT_MODEL_ID
    print(f"model_id (default): {model_id}")
    tok = describe_tokenizer(model_id)

    print(f"\ndataset: {train_prm.DEFAULT_PRM_DATASET} (streaming)")
    stream = load_dataset(train_prm.DEFAULT_PRM_DATASET, split="train", streaming=True)
    first = next(iter(stream))
    print(f"top-level keys: {list(first.keys())}")
    steps, labels = train_prm.get_steps_and_labels(first)
    print(f"  get_steps_and_labels -> {len(steps)} steps, labels(sample)={labels[:8]}")
    print(f"  class map {train_prm.PRM_CLASS_TO_IDX} (raw rating -> class idx)")

    data = train_prm.build_prm_dataset(tok, limit=2)
    print(f"\nbuild_prm_dataset -> {len(data)} examples")
    row = data[0]
    labels = row["labels"]
    labeled_pos = [i for i, x in enumerate(labels) if x != -100]
    print(f"  seq len={len(labels)}  labeled_positions={len(labeled_pos)} (only step terminators)")
    print(f"  STEP_SEPARATOR={train_prm.STEP_SEPARATOR!r}")
    if labeled_pos:
        i = labeled_pos[0]
        print(
            f"  first labeled token id={row['input_ids'][i]!r} decoded={tok.decode([row['input_ids'][i]])!r} class={labels[i]}"
        )
    print("  loss = cross-entropy over the 3-class head at terminator tokens only")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["orm", "prm", "preference"], default=None)
    args = parser.parse_args()

    checks = {
        "preference": preflight_preference,
        "orm": preflight_orm,
        "prm": preflight_prm,
    }
    targets = [args.only] if args.only else list(checks)

    failures = []
    for name in targets:
        try:
            checks[name]()
        except Exception as e:  # noqa: BLE001 - report any failure uniformly
            failures.append(name)
            hr(f"{name.upper()} FAILED")
            traceback.print_exc()

    print("\n" + "=" * 72)
    if failures:
        print(f"RESULT: {len(failures)} FAILED -> {failures}")
        return 1
    print("RESULT: all preflights passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
