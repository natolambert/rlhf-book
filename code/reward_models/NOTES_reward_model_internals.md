# Reward Model Internals — Code Walkthrough Notes

> Study notes derived from a close reading of `base.py` and `train_orm.py`
> (with contrasts to `train_preference_rm.py` and `train_prm.py`).
> Focus: *why* the code is shaped the way it is — dtypes, head design, tensor
> shapes, the masked loss, padding, inference aggregation, and one important
> limitation of the synthetic ORM data.

---

## 1. Reward head construction and the bf16 cast

```python
# base.py:61-62
self.head = self._build_head(self.model.config.hidden_size, head_dim)
self.head = self.head.to(torch.bfloat16)
```

`_build_head` (`base.py:64-66`) returns an `nn.Linear(hidden_size, head_dim)` that
maps a backbone hidden state to a reward score. The whole reward model is:

> frozen/trainable LLM backbone → last hidden state → this linear head → reward

**Why the explicit `.to(torch.bfloat16)`?** The backbone is loaded in bf16
(`base.py:47-52`, `dtype="bfloat16"`), but `nn.Linear` is created in **fp32 by
default**. Feeding a bf16 hidden state into an fp32 head triggers a dtype
mismatch (`RuntimeError: expected scalar type ...`). The cast aligns the head
with the backbone so the whole forward path is bf16 — and it halves the head's
memory.

`torch.amp.autocast` (used in the training loop, `base.py:201`) does **not**
remove the need for this: autocast casts *activations* for selected ops inside
its context, but it does not change a module's stored *parameter* dtype, and
inference/eval paths may run outside any autocast block. Storing the head in bf16
is the robust fix.

---

## 2. The `bias = output_dim > 1` design decision

```python
# base.py:66
return nn.Linear(hidden_size, output_dim, bias=output_dim > 1)
```

A `nn.Linear` computes the affine map `y = xWᵀ + b`. `bias=False` forces the map
through the origin (`y = xWᵀ`). The rule "no bias for a 1-D output, bias for
multi-D output" lands differently in each script:

| Script | `head_dim` | bias | loss |
|--------|-----------|------|------|
| `train_preference_rm.py:163` | `1` | **False** | Bradley-Terry: `-logsigmoid(r_chosen - r_rejected)` |
| `train_orm.py:172` | `1` | **False** | per-token BCE on completion tokens |
| `train_prm.py:264` | `len(PRM_CLASS_VALUES)=3` | **True** | 3-class cross-entropy |

- **Preference RM (the clean reason):** the BT loss only uses the *difference*
  of two scalar rewards. With `r = Wh + b`,
  `r_chosen − r_rejected = W(h_c − h_r)` — the bias **cancels exactly**, so it
  receives zero gradient. It is dead weight; drop it.
- **ORM:** a bias would *not* cancel here (absolute 0/1 targets), but it is
  dropped by the same rule. This is fine in practice: it is the conventional
  scalar-score head (cf. TRL's `score` head), and the ORM dataset is **balanced**
  (one positive + one negative per question, `train_orm.py:129,134`), so the
  optimal bias is ≈ 0 anyway.
- **PRM (bias genuinely needed):** under softmax + cross-entropy the per-class
  bias differences set the baseline log-odds between classes and **do not
  cancel**. PRM800K step labels are heavily imbalanced, so the bias vector lets
  the head encode that class prior directly instead of forcing `W` to learn it.

---

## 3. ORM forward pass — tensor shapes

```python
# Book's minimal inlined version. In the repo, hidden_states[-1] is
# factored into base.py:81 (get_hidden_states), called from train_orm.py:191,
# and the squeeze is train_orm.py:192:
hidden = outputs.hidden_states[-1]          # (B, S, H)   <- base.py:81
logits = self.head(hidden).squeeze(-1)      # (B, S)      <- train_orm.py:192
```

Using `batch=2`, `seq_len=6`, `hidden_size=1024`:

| Step | Expression | Shape |
|------|-----------|-------|
| token ids | `input_ids` | `(2, 6)` |
| all layers' hidden states | `outputs.hidden_states` | tuple of length `num_layers+1` |
| **final** hidden state | `hidden_states[-1]` | `(2, 6, 1024)` |
| linear head (acts on last dim only) | `self.head(hidden)` | `(2, 6, 1)` |
| drop the singleton | `.squeeze(-1)` | `(2, 6)` |

`output_hidden_states=True` makes the model return every layer's states as a
tuple; `[-1]` is the last transformer layer (richest representation). `nn.Linear`
broadcasts over the leading `(B, S)` dims and maps each token's 1024-vector to one
scalar logit. `squeeze(-1)` removes the trailing size-1 dim so `logits` is
`(B, S)`, aligned with `labels` `(B, S)` for the per-token loss.

---

## 4. Masked per-token BCE loss

```python
# train_orm.py:196-198
mask = labels != -100
if mask.any():
    loss = F.binary_cross_entropy_with_logits(logits[mask], labels[mask].float())
```

- **`mask = labels != -100`** — `(B, S)` boolean selecting only completion tokens.
  Prompt/padding tokens are `-100` (PyTorch's conventional `ignore_index`) and are
  excluded, because correctness should be judged from the *generated* tokens only.
- **`mask.any()`** — a safety valve: if a (micro)batch has zero labeled tokens,
  `logits[mask]` is empty and BCE would return `nan`. The `else` branch keeps the
  graph connected with a zero loss (`logits.sum() * 0`).
- **Boolean indexing flattens.** With `N` `True` positions:
  `logits[mask]` → `(N,)` and `labels[mask].float()` → `(N,)`, element-aligned.
  `.float()` is required because BCE-with-logits expects a float *target*.

**The two arguments:**

| | arg 1 `logits[mask]` | arg 2 `labels[mask].float()` |
|---|---|---|
| role | `input` (raw logits) | `target` (0/1) |
| shape | `(N,)` | `(N,)` |
| dtype | float | must be cast to float |

**The computation** (per element, `x`=logit, `y`=label):
`p = σ(x)`, then `ℓ = −[y·log p + (1−y)·log(1−p)]`, then mean over the `N`
selected tokens. Worked example with `logits=[2.0,−1.0,0.0]`, `labels=[1,0,1]`:

| x | y | p=σ(x) | ℓ |
|---|---|--------|---|
| 2.0 | 1 | 0.8808 | 0.1269 |
| −1.0 | 0 | 0.2689 | 0.3133 |
| 0.0 | 1 | 0.5000 | 0.6931 |

`loss = (0.1269 + 0.3133 + 0.6931) / 3 ≈ 0.378`.

**Why `_with_logits` and not a manual `sigmoid`?** It fuses sigmoid + BCE using the
numerically stable form `ℓ = max(x,0) − x·y + log(1 + e^{−|x|})`, avoiding
`log(0)` overflow for large `|x|`. So you must pass **raw logits, never
pre-sigmoided probabilities** (doing both = double sigmoid = wrong math).

**The `else: logits.sum() * 0` branch — a graph-connected zero, and who needs it.**
When `mask.any()` is `False`, `logits[mask]` is empty and `cross_entropy`/`BCE`
returns `nan`. You can't fall back to a plain `0.0` (no `.backward()`) or a fresh
`torch.tensor(0.0)` (no `grad_fn` → *"does not require grad"* on backward).
`logits.sum() * 0` is a tensor still wired to the params: value `0`, gradient `0`, so
`loss.backward()` runs cleanly and the optimizer step is a no-op. **PRM has the
identical guard** (`train_prm.py:289-292`, `cross_entropy`). **Preference RM has none
and needs none** (`train_preference_rm.py:220`): Bradley-Terry reads exactly one
guaranteed token per sequence (last non-pad) and reduces `.mean()` over a non-empty
batch — no maskable subset, so no empty-subset / `nan` failure mode. So the guard
tracks the *loss shape*: per-token masked (ORM, PRM) needs it; per-sequence pairwise
scalar (pref RM) doesn't. It's **defensive** — the dataset builders `continue`-skip
label-less records, making it near-unreachable; the realistic trigger is *truncation*
dropping every labeled token and leaving an all-`-100` row.

---

## 5. The generic training loop's metric accumulation

```python
# base.py (training_loop), ~213-217
for k, v in metrics.items():
    if k not in epoch_metrics:
        epoch_metrics[k] = 0.0
    epoch_metrics[k] += v
```

This is the dictionary version of `epoch_loss += loss.item()`.

- `metrics` is the per-step dict returned by `compute_loss_and_metrics`
  (e.g. `{"accuracy": 0.75}`), overwritten each step.
- `epoch_metrics` is a running **sum** across the epoch, initialized empty.
- The `if k not in epoch_metrics` line is **lazy init**: the first time a key is
  seen, set it to `0.0` before `+=` (otherwise `KeyError`).
- At epoch end (`base.py:224-233`) each sum is divided by `len(loader)` to report
  the average; e.g. accuracies `0.5, 0.7, 0.9` accumulate to `2.1`, then
  `2.1/3 = 0.7`.

It is written generically (dict/key-driven, not hardcoded names) because
`training_loop` is shared across reward models that report different metric sets.

---

## 6. Two collate functions, and why they differ

There are **two** padding implementations in this folder, for **two different
data shapes**:

**(a) `base.py` `pad_sequences` / `create_collate_fn`** — pad value chosen by
field-name suffix (`base.py:271-279`):

```python
if field.endswith("_ids"):    pad_value = tokenizer.pad_token_id
elif field.endswith("_mask"): pad_value = 0
else:                          pad_value = -100   # labels
```

Each field is padded **independently** (its own `max_len`). This is what
**preference RM** needs (`train_preference_rm.py:139-144`): each example has
**two independent sequences** — `chosen` and `rejected` — of **different
lengths**. So the `chosen` tensor may be `(B, 40)` while `rejected` is `(B, 55)`;
they are scored in two separate forward passes. A single shared `max_len` cannot
express that.

**(b) `train_orm.py` `collate_fn`** (preallocate-and-fill, `train_orm.py:139-152`):

```python
max_len = max(len(x["input_ids"]) for x in batch)              # ONE max_len
inputs = torch.full((B, max_len), tokenizer.pad_token_id, ...) # prefill PAD
attn   = torch.zeros_like(inputs)                              # prefill 0
labels = torch.full((B, max_len), -100, ...)                  # prefill -100
for i, item in enumerate(batch):
    length = len(item["input_ids"])
    inputs[i, :length] = ...   # cells past `length` keep the prefilled pad value
```

One shared `max_len` works **only because** ORM has a single sequence per example
whose three fields (`input_ids`/`attention_mask`/`labels`) are token-aligned and
equal-length. Prefilling with the pad value means the un-overwritten tail *is* the
padding — for free.

Both produce identical math; the difference is fit-to-purpose (single sequence vs.
chosen/rejected pair), plus history: `train_orm.py` / `train_prm.py` are adapted
from an external repo and kept their own collate style, while `base.py` +
`train_preference_rm.py` share `pad_sequences`.

The three pad values are also **coordinated**: at a padded position
`input_ids=PAD`, `attention_mask=0` (attention ignores it), `labels=-100`
(loss ignores it) — so padding is invisible in both the forward and backward pass.

---

## 7. ORM inference scoring — mean over completion tokens

```python
# train_orm.py:360-381 (score_completion)
probs = torch.sigmoid(logits)                 # (1, L)
mask  = batch["labels"][0] != -100            # (L,)
return probs[0][mask].mean().item()           # scalar
```

The score is the **mean of per-completion-token "this belongs to a correct
solution" probabilities**. Notes:

- `pack_example(..., 1, ...)` passes `label=1`, but the value is irrelevant at
  inference: `labels` is used only to build `mask` via `!= -100`. Whether 0 or 1,
  the completion tokens are selected identically.
- **Why mean over tokens?** ORM is trained with **per-token BCE where every
  completion token carries the *same* outcome label** (`train_orm.py:99`). So each
  token position is an independent estimate of the same scalar outcome; averaging
  is variance reduction.
- **Aggregation must match training.** Contrast preference RM, which reads the
  reward from the **last non-padding token only** (`train_preference_rm.py:165-178`)
  because that's where its BT reward was trained. Same codebase, deliberately
  different pooling, each consistent with its loss.

---

## 8. ⚠️ Key limitation — synthetic negatives and the shared-prefix problem

This is the most important takeaway. The ORM negatives are constructed as
(`train_orm.py:128-134`):

```python
rows.append(pack_example(prompt, answer, 1, tokenizer))            # positive
wrong = value + random.randint(1, 9)
wrong_solution = answer + f"\nTherefore, the answer is {wrong}."
rows.append(pack_example(prompt, wrong_solution, 0, tokenizer))    # negative
```

So **the negative = the full correct solution (reasoning *and* the correct
`#### N`) + one appended contradictory sentence**. Concretely, `pack_example`
tokenizes `completion + eos_token` (`train_orm.py:94`), so the positive's
**answer tokens** (everything before its terminal EOS) are a **strict prefix**
of the negative's tokens; the two diverge exactly at the positive's EOS, where
the negative instead begins its appended `"\nTherefore, the answer is {wrong}"`.
(The positive's own EOS is therefore *not* token-identical to the negative at
that position — but it still lands on a shared causal context, so the argument
below is unaffected.)

**Consequence (the p ≈ 0.5 effect).** Per-token BCE has Bayes-optimal
`p* = N₁ / (N₁ + N₀)` for a given (causal) context. In the shared prefix the
context is byte-identical between the positive and the negative, appearing once
with label 1 and once with label 0 → `N₁ = N₀ = 1` → **`p* = 0.5`**. Equivalently:
the label-1 and label-0 gradients on those identical-context tokens are equal and
opposite, so the net learning signal is ≈ 0 and they sit near `p = 0.5`. Only the
appended "Therefore, the answer is {wrong}" tokens appear *exclusively* with
label 0 → `p* → 0`; that tail is the **only** discriminative signal.

**What the scores converge to (idealized):**

- positive (answer tokens all on shared contexts) → `correct_score → ~0.5`
- negative (shared prefix at 0.5 + short tail at ~0) → below 0.5

Idealized illustration (numbers chosen for clarity, not code-measured) —
~50 shared answer tokens + 8 appended tokens:

```
correct_score   = mean(50 × 0.5)            = 0.50
incorrect_score = (50×0.5 + 8×0.0) / 58     ≈ 0.43
```

`correct > incorrect` still holds — but the ~0.07 margin comes **entirely** from
the 8 appended tokens, and **mean aggregation dilutes** it against a long stretch
of uninformative 0.5 tokens. For *this* dataset a last-token / tail-weighted
aggregation would separate the classes more sharply than the mean.

**Deeper point:** the dataset never contains a genuinely *flawed reasoning trace*
labeled 0 — only "correct trace + appended wrong claim." So this ORM does not
learn to judge reasoning quality; it learns to **detect an appended contradictory
final answer**. It is closer to a *contradiction/answer-consistency detector* than
a true outcome reward model. This is consistent with the README marking these RMs
as experimental and needing tuning.

**What a real ORM does:** sample many *complete* solutions from a policy per
question, grade each against the gold answer, and train on those. Wrong solutions
then have their *own* independent (and actually flawed) reasoning, every token
legitimately carries label 0, nothing is shared with a correct solution, and
per-token signal (hence mean/last-token aggregation) becomes meaningful.

---

## Practical takeaways

1. Keep a score/reward head's dtype in sync with the backbone; don't rely on
   autocast for parameter dtype.
2. `bias=False` on a scalar reward head is principled for Bradley-Terry (bias
   cancels) and harmless for a balanced ORM; keep bias for multi-class PRM.
3. Per-token reward losses need a `-100` mask + an empty-mask guard; pass raw
   logits to `*_with_logits`.
4. Inference aggregation must mirror how the reward was trained (ORM: mean over
   completion tokens; preference RM: last token).
5. Be skeptical of cheap synthetic negatives: if the negative shares a long
   prefix with the positive, most per-token labels are self-canceling (p ≈ 0.5)
   and your "reward model" may just be detecting a surface artifact.
