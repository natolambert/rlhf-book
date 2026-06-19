# Instruction Tuning — session context & handoff

State of the **Qwen3-1.7B-Base SFT learning experiment**. Read this first when
resuming. The general repo rules in `code/CLAUDE.md` still apply (use
`uv run python`, run training in the background, ruff + pytest before finalizing,
`code/CHANGELOG.md` entry required for any PR touching `code/`).

## What this is

A learner practicing supervised fine-tuning by taking `Qwen/Qwen3-1.7B-Base` and
teaching it a **custom minimal ChatML template** on `HuggingFaceH4/no_robots` —
parallel to the book's OLMo-2-1B exercise but on a current base model. All work
is on branch **`qwen3-base-sft`** (8 commits, **not pushed**, vs `main`).

The two original files (`train.py`, `utils.py`, `config.py`, `configs/sft_olmo2_1b.yaml`)
are the book's OLMo exercise and still work unchanged — don't break them.

## Files added in this experiment

| File | Purpose |
|------|---------|
| `train_qwen_base.py` | **The entry point for this experiment** (not `train.py`). Sets the custom ChatML template, `eos=<\|im_end\|>`, generate() stop ids, warm-starts the `<\|im_end\|>` embedding, trains, saves. |
| `configs/sft_qwen3_1p7b.yaml` | v1 config (no warm-start era; warm-start is now always on in the script). |
| `configs/sft_qwen3_1p7b_v2.yaml` | v2 config (the one to use). |
| `chat_app.py` | Gradio UI (Inspect / Chat / Compare tabs) to play with checkpoints. |
| `QWEN3_SFT_BUGS.md` | Detailed report of the 8 bugs found + fixed. |
| `../scripts/tokenizer_preflight.py` | Pre-flight checklist before switching to a new model/tokenizer. |
| `../scripts/train_qwen3_lr_sweep.sh` | Driver to train+SAVE checkpoints across learning rates. |

## How to run

```bash
cd code/

# Train (v2 = warm-start, the good config). Background it; ~2.5 h, 3 epochs.
WANDB_PROJECT=rlhf-book uv run python -m instruction_tuning.train_qwen_base \
    --config instruction_tuning/configs/sft_qwen3_1p7b_v2.yaml

# Pre-flight a new model before adapting the recipe to it
uv run python scripts/tokenizer_preflight.py <model> --chat-template-source <instruct-sibling>

# Play with the saved checkpoints (Inspect/Chat/Compare). Opens http://localhost:7860
uv run --extra ui python -m instruction_tuning.chat_app
#   set RLHF_RUNS to point at the checkpoint dir (default ~/rlhf-runs)

# Train+save checkpoints across LRs so they show up in the UI dropdowns
# (STOP the UI first — it holds models in VRAM and would OOM training)
bash scripts/train_qwen3_lr_sweep.sh            # 5e-6/1e-5/2e-5/2e-4, 1 epoch
```

## Results so far

Checkpoints live on **ext4** at `~/rlhf-runs/` (NOT in the repo — see env note):

| Name | What | W&B run (ghotifish/rlhf-book) |
|------|------|------|
| `sft-qwen3-1.7b-base-chatml` | v1 — content learned but `<\|im_end\|>` never stopped (P≈1e-4) | `cytwravi` |
| `sft-qwen3-1.7b-base-chatml-v2-warmstart` | **v2 — the good one**, clean stop (P=0.318, argmax) | `89qhqgot` |
| `smoke-qwen3` | throwaway 64-row smoke test (deletable) | — |
| (LR sweep — **not saved**, output_dir was null) | 2e-5 best stable; 2e-4 diverges | `ehy62j0f`/`q7opju1w`/`utblq7ea`/`vnzzo8s3` |

## Key lessons (the "why" behind the code)

- **Stop tokens**: HF `generate()` stops on `model.generation_config.eos_token_id`,
  NOT `tokenizer.eos_token` (QwenLM/Qwen3#927). Set both — generation_config for
  HF, tokenizer.eos for vLLM/third-party.
- **Warm-start the stop token**: `<\|im_end\|>` is near-untrained in the base
  (embedding norm 0.38 vs ~1.6); an SFT LR can't train its identity from scratch.
  Copy `<\|endoftext\|>`'s row into it (`emb[151645]=emb[151643]`, tied embeddings).
  This is why OLMo's recycled `<\|endoftext\|>` stop is easy and atomic ChatML isn't.
- **Learning rate**: book ch.4 — SFT LR is 1–2 orders of magnitude below
  pretraining. Sweep confirmed 2e-5 best stable, 2e-4 (a pretraining/CPT-regime LR)
  diverges for full-param SFT.
- **Template gotchas**: `_encode_row`'s prefix trick relies on the chat template
  being prefix-consistent; the custom template here uses Jinja `{%- -%}` which
  strips whitespace (intentional — keeps a BPE-safe mask boundary). Run the
  preflight before trusting a new template.

## Environment notes

- **GPU**: RTX 5090 (Blackwell, sm_120) on WSL2. Needs **cu130** torch wheels —
  `pyproject.toml` is pinned (cu126 fails with "no kernel image"). flash-attn has
  no sm_120 wheel; the code falls back to SDPA. ~27 GB VRAM for full-param 1.7B SFT.
- **Filesystem**: the repo is on `/mnt/c` (OneDrive, slow 9P). Keep checkpoints +
  wandb on ext4 (`~/rlhf-runs/`), already the configs' default. `~/.cache/huggingface` is ext4.
- **UI dep**: gradio is in the `ui` optional extra (`uv run --extra ui ...`).
  Gradio 6.x — its API differs from 5.x (see BUG-6/7 in the bug report).

## Open threads (next steps)

- [ ] Promote **2e-5** to a full 3-epoch *saved* run (best stable LR; the sweep
      didn't save). Use `train_qwen3_lr_sweep.sh` with `EPOCHS=3 LRS="2e-5"`.
- [ ] DPO on the v2 checkpoint (`code/direct_alignment/`) — the book's next stage.
- [ ] Push the branch as a PR — needs a `code/CHANGELOG.md` entry (CI enforces it).
- [ ] (optional) delete `~/rlhf-runs/smoke-qwen3` (3.4 GB).
