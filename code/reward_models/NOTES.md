# Reward Models — study notes (Qwen3, ch.5 practice)

Companion notes for the **Chapter 5: Reward Models** exercises
(<https://rlhfbook.com/c/05-reward-models>), written while getting the three
trainers running on a single RTX 5090. Parallel to the `instruction_tuning`
Qwen3 SFT practice; branch **`qwen3-reward-models`** (forked from
`qwen3-base-sft` so it keeps the cu130 torch pin + `scripts/tokenizer_preflight.py`).

These three scripts are **experimental/educational** (see `README.md`): they show
that each objective *trains at all*, not a tuned production recipe.

---

## 0. TL;DR — the three models at a glance

| | **Preference RM** | **ORM** | **PRM** |
|---|---|---|---|
| File | `train_preference_rm.py` | `train_orm.py` | `train_prm.py` |
| Chapter section | "Training a Bradley-Terry RM" | "Outcome Reward Models" | "Process Reward Models" |
| Default backbone | `Qwen3-0.6B-Base` | `Qwen3-1.7B-Base` | `Qwen3-0.6B-Base` |
| Dataset | `argilla/ultrafeedback-binarized-preferences-cleaned` | `openai/gsm8k` | `tasksource/PRM800K` |
| Head | `Linear(hidden, 1)` | `Linear(hidden, 1)` | `Linear(hidden, 3)` |
| What it reads | **last non-pad token** hidden state | **every** completion token | **step-terminator** tokens only |
| Output | 1 scalar / sequence | per-token logit (→ P(correct)) | per-step 3-class logits (−1/0/1) |
| Label granularity | pairwise (chosen ≻ rejected) | per-token, repeats the sequence outcome | per-step (one label per reasoning step) |
| Loss | `-logsigmoid(r_c − r_r)` (Bradley-Terry) | per-token BCE-with-logits | per-step cross-entropy (3-class) |
| Chapter eq. | @eq:rewardmodeling1 (5.5) | @eq:orm_loss (5.10) | @eq:prm_loss (5.11) |
| "What is good?" | how good is the **whole answer** | which **tokens** look correct | are the **reasoning steps** sound |

All three: backbone is **fully fine-tuned** (not frozen), bf16, AdamW, linear LR
warmup, no separate eval split (a post-train `demo_scoring` is the sanity check).

---

## 1. Shared architecture (`base.py`)

Every RM is a **causal LM backbone + a small linear head on the last hidden
state**. This mirrors HF `AutoModelForSequenceClassification` but is spelled out
so you can see the mechanics (chapter §"The Default Reward Model Architecture").

```
input_ids ─► AutoModelForCausalLM(output_hidden_states=True)
                          │
              hidden_states[-1]        # (batch, seq_len, hidden)   << the LM head is NOT used
                          │
                  Linear(hidden, K)    # K=1 (RM/ORM) or 3 (PRM)
                          │
        ┌─────────────────┼──────────────────────────┐
   last-token (RM)   every token (ORM)        step terminators (PRM)
```

Key points (`BaseRewardModel`):
- The model's own **LM/unembedding head is loaded but never called** — we only
  consume `outputs.hidden_states[-1]` and run our own `head`. (Memory could be
  saved by dropping the LM head; left in for simplicity.)
- `freeze_backbone=False` by default → **full fine-tune** (backbone + head both
  learn). `create_optimizer` selects every `requires_grad=True` param.
  `freeze_backbone=True` would train the head only (linear probe).
- bf16 weights, `device_map={"": 0}`, `use_cache=False`.
- `load_tokenizer` sets `pad_token = eos_token` when the model has no pad token.
  For Qwen3 this means **`pad == eos == <|endoftext|>` (id 151643)** — see the
  pad/eos gotcha in §5.

### Tokenizer facts (from `rm_preflight.py`, all three Qwen3 base models)
- `Qwen2Tokenizer`, `vocab = 151669`, **no BOS**, `pad == eos == <|endoftext|>(151643)`.
- ChatML control tokens are **already atomic single ids** in the *base* vocab:
  `<|im_start|>=151644`, `<|im_end|>=151645`, `<|endoftext|>=151643`.
  (The base model's *embeddings* for `<|im_end|>` are near-untrained — relevant
  to Q3 below — but the tokenizer knows the ids.)

---

## 2. Preference RM — Bradley-Terry (`train_preference_rm.py`)

**Objective.** Learn a scalar `r_θ(x,y)` such that the chosen completion scores
higher than the rejected one. Bradley-Terry models `P(y_c ≻ y_r) = σ(r_c − r_r)`;
minimizing its negative log-likelihood gives the loss (chapter @eq:rewardmodeling1):

```
loss = -logsigmoid(r_chosen - r_rejected).mean()
```

Only the **difference** matters (adding a constant to all rewards is invariant),
so the model just has to separate chosen from rejected along one direction.

**Input shape / data.** Dataset `argilla/ultrafeedback-binarized-preferences-cleaned`.
Preflight confirmed `chosen`/`rejected` are **2-message lists** `[user, assistant]`
whose **user turn is identical** and assistant turn differs — exactly the
shared-prompt pairing BT wants. `format_conversation` flattens each to plain text:

```
user: <prompt>
assistant: <response>
```

Each side is tokenized independently (`add_special_tokens=True`; Qwen adds
nothing since there's no BOS) → `chosen_ids` / `rejected_ids` (variable length,
e.g. 126 vs 71), padded per-batch. **No chat template is used.**

**Output / what is scored.** `get_reward` takes `hidden_states[-1]`, selects the
**last non-pad token** per sequence (`seq_lengths = attention_mask.sum(1) - 1`),
and applies `Linear(hidden,1)` → one scalar. For the plain `user:/assistant:`
text this last token is a content token (e.g. `'.'`), **not** a special token.

**What trains.** Full backbone + scalar head, on the *difference* of two forward
passes (chosen and rejected) per pair.

**Smoke result** (0.6B, 64 pairs, 8 steps, `--no-wandb`): loss `0.63 → 0.21`,
batch accuracy → `1.0`; demo `good=-2.17 > bad=-3.31` ✅. (Rewards are unbounded
and only differences are meaningful — negative absolute values are fine.)

> Book variants worth knowing (not implemented here): **margin loss** (Llama 2,
> uses Likert gap `m(y_c,y_r)`), **InstructGPT K-wise grouping** (all `C(K,2)`
> pairs from one prompt in one batch to avoid overfitting), **Plackett-Luce
> K-wise** (Starling). BT is the `K=2` special case.

### `tokenize_messages` — two branches, why the flags differ (`train_preference_rm.py:72-96`)

An RM **scores finished text** (prompt **+** the assistant answer that's already
present); it never generates. So tokenization just has to turn a *complete*
conversation into `input_ids` / `attention_mask`. Two branches by whether the
tokenizer ships a chat template:

**Branch 1 — instruct model, has a chat template** (`apply_chat_template`):

```
<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>
```

- `add_generation_prompt=False` — this flag appends a **trailing empty assistant
  header** (`<|im_start|>assistant\n`) that primes generation. That's an
  *inference* cue, used when you only have the user turn. Here the answer is
  **already in `messages`**; setting `True` would tack on a dangling empty turn,
  and since the reward is read from the **last token**, you'd be scoring an empty
  generation prompt instead of the real end of the answer (`<|im_end|>`). So
  `False` keeps the sequence ending on the actual response. Cross-ref Q3.
- No `add_special_tokens` arg here: the **template already injects** every special
  token (`<|im_start|>`/`<|im_end|>`, BOS if defined). Passing it would double up.

**Branch 2 — base model, `chat_template is None`** (the one this experiment
actually takes — Qwen3 *base*): falls back to `format_conversation`'s plain
`user:/assistant:` string, then a bare `tokenizer(...)`:

- `add_special_tokens=True` — nothing has inserted structural tokens yet, so this
  asks the tokenizer to add its **defaults (BOS/EOS)**. It's the **parallel** of
  the template owning specials in Branch 1 — same job, two regimes. (`__call__`
  already defaults this to `True`, so the line also just *documents intent*: yes,
  we want BOS/EOS here, unlike the template branch where the template decides.)
  Per the preflight, **Qwen3 base has no BOS**, so in practice this adds nothing —
  but it's the correct, model-agnostic flag to set.

So: Branch 1 suppresses a trailing token (`add_generation_prompt=False`); Branch 2
adds leading/trailing specials (`add_special_tokens=True`). Both return a dict
with `input_ids` + `attention_mask` → `chosen_ids`/`rejected_ids`, padded per-batch.

---

## 3. ORM — Outcome Reward Model (`train_orm.py`)

**Objective.** Per-token **binary correctness**. From Cobbe et al. 2021 (GSM8K
verifiers): a scalar head over the LM predicts, at every completion token,
whether the *final answer* is correct. Loss (chapter @eq:orm_loss) is per-token
BCE on completion tokens, prompt tokens masked:

```
mask = labels != -100
loss = F.binary_cross_entropy_with_logits(logits[mask], labels[mask].float())
```

This is **not** the contrastive chosen/rejected structure — it's much closer to
the LM loss, just with a 0/1 target per token. No pairing required.

**Input shape / data.** GSM8K. `prompt = "Question: …\nAnswer:"`. Two examples
per question (`pack_example`):
- **Positive**: the gold solution. `labels = [-100]*prompt + [1]*completion`.
- **Negative**: the *same* gold solution with `"\nTherefore, the answer is <wrong>."`
  appended. `labels = [-100]*prompt + [0]*completion`.

`completion` ends with `eos`. Preflight: POS had 60 completion tokens labeled `1`,
NEG 70 labeled `0`, 42 prompt tokens masked each.

> ⚠️ **Noisy negative (documented inline).** The negative reuses the correct
> reasoning and only appends a wrong sentence, yet labels *all* its tokens `0`.
> So identical reasoning tokens appear under both labels across examples — the
> only truly label-distinguishing signal is the trailing wrong answer. A better
> ORM dataset uses real wrong rollouts (verifier-labeled), not a synthetic offset.

**Output / what is scored.** `head(hidden).squeeze(-1)` → per-token logit.
`score_completion` = `sigmoid(logit)` averaged over completion tokens
(chapter discusses mean / min / product aggregations at inference).

**What trains.** Full backbone + scalar head, one forward per example.

**Smoke result** (1.7B, 40 samples → 80 ex, 10 steps): BCE bounces `0.7–2.0`
(tiny noisy run), epoch acc `0.50`; demo `correct=0.342 > incorrect=0.332` ✅
(margin tiny because barely trained, but ordering is right).

> **ORM vs Value function** (chapter): same per-token head shape, different
> semantics. ORM predicts a token-local `P(correct)` from **offline labels**; a
> value function predicts **expected remaining return** from **on-policy
> rollouts**. With `γ=1` and a dense correctness reward they look alike but the
> supervision pipeline differs.

---

## 4. PRM — Process Reward Model (`train_prm.py`)

**Objective.** Score **each reasoning step**, not the whole answer or every
token. From *Let's Verify Step by Step* (Lightman 2023). 3-class head
(`-1` incorrect, `0` neutral, `1` correct) trained with per-step cross-entropy
at step boundaries (chapter @eq:prm_loss):

```
mask = labels != -100            # only step-terminator tokens
loss = F.cross_entropy(logits[mask], labels[mask])
```

**Input shape / data.** PRM800K (`tasksource/PRM800K`, streamed). Prompt
`"Problem: …\nReasoning trace:\n"`, then steps joined by
`STEP_SEPARATOR = "\n<step>\n"`. Crucially, **only the last token of each
`step + separator` carries a label** (the TRL trick `[-100]*(len-1)+[label]`,
chapter lines 346-353); everything else is `-100`. Preflight: an 11-step trace →
252 tokens, **11 labeled positions**; the labeled token decodes to `'>\n'`
(tail of the separator); raw ratings `{-1,0,1}` map to class idx `{0,1,2}`.

**Output / what is scored.** `head(hidden)` → `(batch, seq, 3)`. `score_trace`
softmaxes the logits at each step boundary → a 3-class distribution **per step**.
Aggregation at inference is over **steps** (mean / min / fail-fast), not tokens.

**What trains.** Full backbone + 3-class head; `batch_size=1` because traces are
long (`max_tokens=5500`, traces chunked at `max_steps=20`).

**Smoke result** (0.6B, 32 samples, 8 steps): CE `2.7 → 1.1` (baseline `ln 3 ≈ 1.10`),
step acc `~0.36`; demo prints per-step `-1/0/1` probabilities across a full trace.
Barely-trained model defaults to predicting class `1` everywhere — expected; the
*path* (per-step 3-class scoring) works ✅.

### Why this shape — separator, build-by-parts, terminator scoring

**`\n<step>\n` is a plain-text separator, not a `[CLS]` special token.** `[CLS]`
isn't in Qwen3-base's vocab, so using it would mean `add_special_tokens` +
`resize_token_embeddings` → a **randomly-initialized** embedding trained from
scratch (the same cold-start that needed embedding warm-starting in the SFT
`<|im_end|>` work). `\n<step>\n` reuses already-trained token embeddings
(`\n`,`<`,`step`,`>`), needs zero tokenizer surgery, is model-agnostic and
printable/debuggable. A *causal* decoder also wants the marker **after** each step
(a leading `[CLS]` can't see the step yet), recurring once per step — not BERT's
single global token. Tradeoff: a real special token is unambiguous + 1 token, but
must be trained; production PRMs sometimes do it, overkill for this educational run.

**Built by parts with `add_special_tokens=False` (`train_prm.py:183, 202`).** Each
`step + separator` is tokenized separately and concatenated as id lists. This is the
deliberate escape from the SFT hazard, where `apply_chat_template` injects
`<|im_end|>`/role markers you don't control, so `tok(a)+tok(b) ≠ tok(template(a+b))`
and label masks shift. No template + no auto-specials → concatenation is exactly what
the model sees, and `label_ids` is grown **in lockstep** with `input_ids`, so each
terminator label lands at the right index *by construction* (no post-hoc boundary
search). Newline-delimited separators tokenize cleanly, so seams don't merge.

**Terminator scoring is train/inference-consistent + prefix-causal.** The label sits
on the **last token of `step + sep`** (the preflight decodes it to `'>\n'`);
`score_trace` reads the head at that *same* index — identical wrapping in both
regimes, so no train/inference mismatch. The separator is a constant "emit the rating
now" cue; the discriminative signal comes from the step content the terminator
attends back over (causal). Because it's causal, a step's score depends only on its
prefix — **future steps don't leak** — so a single forward pass scores every step
(`score_trace` reads all `boundaries` at once), matching how a PRM scores *partial*
traces in tree search / best-of-N.

**`to_plain_text` (`train_prm.py:71-82`)** — defensive normalizer for PRM800K's
heterogeneous fields: `str`→as-is; `dict`→first of `text`/`value`/`content` that's a
str, else space-join its values; `list`→space-join; else `str()`. Guarantees the
tokenizer always receives a `str`, so a row that wraps its text in a dict/list can't
crash `build_prm_dataset`.

---

## 5. Environment & the two bugs found (this practice)

**Env.** torch `2.12.0+cu130` (sm_120 Blackwell; cu126 wheels fail "no kernel
image"), transformers `5.10.2`, datasets `5.0.0`, RTX 5090 ~32 GB. Full
fine-tune VRAM: 0.6B ≈ a few GB, 1.7B ORM ≈ ~20-25 GB (AdamW fp32 moments
dominate). **Run one job at a time.** flash-attn has no sm_120 wheel → SDPA.

**Preflight first.** `uv run python -m reward_models.rm_preflight` exercises the
real `build_*_dataset` paths on tiny slices (read-only, no GPU) and prints token
ids / decoded text / label masks. Run it before spending GPU time on a new model
or after a library bump.

**Bug 1 — `datasets>=5.0` dropped bare canonical names.**
`load_dataset("gsm8k", "main")` now raises
`HfUriError: Repository id must be 'namespace/name'`. Fixed: ORM
`DEFAULT_DATASET = "openai/gsm8k"`.

**Bug 2 — PRM demo crashed streaming the PRM800K *test* split.**
`datasets>=5.0` infers a JSON shard's schema lazily; the *test* JSON trips it
with `TypeError: Couldn't cast array of type int64 to null` partway through
iteration (the *train* split, used by `build_prm_dataset`, streams fine). Fixed:
`demo_scoring` now uses `_fetch_demo_trace`, which prefers an unseen `test`
example but **falls back to `train`** on a cast error. (Training itself was never
affected — only the post-train demo.)

**Pad/eos gotcha (already handled — don't "fix" it).** `pad == eos == 151643`.
The repo reads the reward at the last token **via the attention mask**
(`attention_mask.sum(1)-1`), which is robust. Do **not** mask by
`input_ids == pad_token_id` — that would also erase the real eos. (The classic
`ForSequenceClassification` `pad==eos` footgun, HF issue #35352, was patched in
PR #35911 ~Feb 2025 anyway.)

### How to run
```bash
cd code/
uv run python -m reward_models.rm_preflight          # 1) preflight (no GPU)

# 2) train one at a time (book's suggested sizes). Add WANDB_PROJECT=rlhf-book to log.
uv run python -m reward_models.train_preference_rm --samples 2000 --epochs 1
uv run python -m reward_models.train_orm            --samples 400  --epochs 2
uv run python -m reward_models.train_prm            --samples 500  --epochs 2
#   --no-wandb to skip logging, --skip-demo to skip post-train scoring
```
Watch: Preference RM → reward margin / pair accuracy grows; ORM → separates
correct vs incorrect finals; PRM → sensible per-step scores. RMs are usually
trained **1 epoch** to avoid overfitting (chapter).

---

## 6. Your questions, answered (adversarially verified)

### Q1 — Does the reward model train from a base model?

**In this repo: yes** — all three default to a pretrained `*-Base` checkpoint
(table in §0), fully fine-tuned, and **none** loads an Instruct/SFT checkpoint.
They demonstrate feasibility, not best practice. (`--model-id` is forwarded
straight to `from_pretrained`, so you *can* point it at a local SFT checkpoint
from `instruction_tuning/`.)

**In production: usually the opposite — initialize the RM from the SFT/instruct
checkpoint** of the policy's family, then replace the LM head with a scalar head:
- **InstructGPT** (Ouyang 2022): RM init from the SFT model; 6B RM (175B was
  unstable). (Appendix C *informally* notes similar results from GPT-3 vs SFT —
  not a controlled ablation.)
- **Llama 2** (Touvron 2023): RM init from **pretrained chat checkpoints** so it
  "knows what the chat model knows" (reduces rewarding hallucinations).
- **Anthropic** (Bai 2022 / Askell 2021): the documented exception — base LM →
  *Preference-Model-Pretraining* (PMP, on Stack Exchange/Reddit comparisons) →
  human-feedback finetune. Note: **no instruction-tuning step**, but PMP is an
  intermediate stage, *not* straight off base.
- **TRL** `RewardTrainer` accepts any checkpoint; its examples use post-trained
  models, not `-Base`.

A controlled "post-trained init beats raw base" comparison across recipe stages
is **RewardBench 2 / Tulu 3** (Malik 2025), not the original RewardBench.

### Q2 — Do we need instruction tuning for a reward model?

**No — SFT is a practical recommendation, not a mathematical prerequisite.**
The BT loss depends only on `r_c − r_r`, so the backbone just needs to make
chosen/rejected linearly separable; nothing in the objective requires SFT. The
repo proves it by training from `Qwen3-0.6B-Base` and still moving pair accuracy.

**But "works" ≠ "works well."** Standard practice initializes from SFT/instruct
for reasons of *consistency*, not the math:
- **Shared knowledge** — RM knows what the policy knows (Llama 2 rationale).
- **Format/tokenizer match** — RM sees responses in the policy's chat format.
- **In-distribution scoring** — early RLHF the policy ≈ the SFT init.

Two myths to avoid: (1) InstructGPT did **not** ablate SFT-vs-base RM init.
(2) The distribution-mismatch problem is fixed mainly by **iterative on-policy
preference data**, not by the init checkpoint (Llama 2). **ORMs/PRMs tolerate a
base backbone more naturally** than a BT preference RM — their per-token CE loss
is close to the LM objective and needs no chat pairs. Quality caveat: from base,
the last-token representation is less aligned with "response quality," so the RM
is more prone to latching onto length/format (a robustness risk, not a blocker).

### Q3 — How to use an RM with a chat model whose template has special tokens?

For **Qwen3 + ChatML**: the control tokens are atomic ids the base tokenizer
already has (`<|im_start|>=151644`, `<|im_end|>=151645`, `<|endoftext|>=151643`),
**shared between base and instruct**. So a base-tokenizer RM tokenizes ChatML
into the *same* ids the policy uses — they don't split into sub-words.

**What actually breaks: train/inference distribution mismatch — not tokenization.**
The repo's preference RM trains on plain `"user:/assistant:"` text (no
`apply_chat_template` anywhere). Feed it `<|im_start|>…<|im_end|>` generations and
the control tokens are out-of-distribution for the trained head, so the scalar
reward is uncalibrated. (Degradation is gradual — content features still fire —
but real.)

**The "untrained `<|im_end|>` embedding" worry is mostly a red herring *here*:**
- The base Qwen3 row for `<|im_end|>` *is* near-untrained (norm ~0.35 vs ~0.97
  median) — that's the SFT stop-token problem the `qwen3-base-sft` branch fixed
  by warm-starting. But that matters for **generation** (the token must win the
  output softmax), far less for a **scalar read-out**: the head consumes a
  **last-layer contextual hidden state** (function of the well-trained context),
  not the raw embedding row.
- And the repo's RM **doesn't read at `<|im_end|>` at all** — it reads the last
  *non-pad* token, which for `user:/assistant:` text is a content token. Id
  151645 never appears.

**Correct setup to score a ChatML policy reliably:**
1. Tokenizer must contain the ChatML tokens (Qwen3's does).
2. **Initialize the RM from the matching SFT/instruct checkpoint** so the
   special-token embeddings are trained and the backbone shares the policy's
   knowledge.
3. **Train the RM on the exact chat-template string the policy emits** (same
   `apply_chat_template`). TRL's `RewardTrainer` does this automatically for
   conversational datasets, or you can pass pretokenized `input_ids_chosen`/etc.
4. Read the reward at the true last non-pad token via the attention mask
   (the repo already does this).

The repo's preference RM satisfies **(4)** but violates **(1–3)** by default. To
score a ChatML Qwen3 policy: retrain it (or pass `--model-id <local-SFT-path>`,
e.g. your `instruction_tuning` checkpoint) under the policy's actual chat
template — i.e. swap `format_conversation` for `tokenizer.apply_chat_template`.

### Caveats / things not to over-claim
- These scripts are educational; the base-model default shows feasibility, not a
  tuned recipe.
- No single universal production answer for RM init; SFT/chat init is most
  common, Anthropic's PMP route is the exception.
- Don't cite InstructGPT for an SFT-vs-base ablation, or Gao 2022
  (over-optimization) as evidence about init.
- Exact-template match is one kind of distribution shift; length/format bias and
  reward over-optimization also matter and aren't fixed by init.
- TRL specifics are version-dependent (`RewardTrainer` wants a pre-instantiated
  `AutoModelForSequenceClassification(num_labels=1)`; collator/EOS behavior
  differs across TRL versions).

---

## 7. Suggested next experiments (chapter §"Suggested Experiments")
1. Preference RM on UltraFeedback `--samples 2000 --epochs 1`; watch the reward
   margin grow; sweep `--samples/--lr/--model-id` to find where it gets noisy.
2. Compare outcome vs process supervision: ORM (GSM8K) vs PRM (PRM800K).
3. Add a small (50-200 example) held-out RM eval reporting pair-ordering accuracy
   without a full training run (an open repo TODO; also RewardBench-style).
4. **Bridge to the SFT work**: pass `--model-id` = your `qwen3-base-sft`
   checkpoint and retrain the preference RM under the ChatML template, then use
   it for best-of-N / rejection sampling (chapter 9) on that policy's outputs.
