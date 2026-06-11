# Bug report — Qwen3-1.7B-Base SFT experiment

Branch `qwen3-base-sft`. Bugs found and fixed while taking a hand-written
`train_qwen_base.py` from "looks done" to a working, reproducible SFT run plus a
chat UI. Ordered by severity. Each entry: symptom → root cause → evidence → fix.

Severity legend: **P0** silent wrong result · **P1** crash/blocks the run ·
**P2** breaks an adjacent feature · **P3** quality/robustness.

---

## BUG-1 (P0) — stop-token fix was a silent no-op

**Symptom.** After SFT, sample panels never terminated; the model produced an
answer then kept generating (invented follow-up turns) to the token cap.

**Root cause.** `train_qwen_base.py` set `tokenizer.gen_eos_token_ids = [...]`
and `utils.load_model` copied it onto the tokenizer. No such attribute is read
by anything in `transformers`. HF `generate()` stops on
`model.generation_config.eos_token_id`, which on the base model is only
`<|endoftext|>` (151643) — never the `<|im_end|>` (151645) the SFT model learns
to emit. So generation could never stop on the trained turn-closer.

**Evidence.** With a freshly loaded checkpoint:
`model.generation_config.eos_token_id == 151643` (the made-up tokenizer
attribute had no effect). Greedy decoding ran to `max_new_tokens` every time.

**Fix.** Set the real field, on the model, after load
(`utils.load_model`, gated by a new `generation_eos_ids` kwarg):
```python
model.generation_config.eos_token_id = generation_eos_ids  # [151643, 151645]
```
Reference: QwenLM/Qwen3#927 (base ships only the document EOS; the fine-tuner
must add the template's turn-closer to the stop list). `tokenizer.eos_token` is
*also* set, but only as metadata for vLLM / third-party stacks — it does not
affect HF `generate()`.

---

## BUG-2 (P0, latent) — `<|im_end|>` cannot be trained from its base embedding

**Symptom.** Even with BUG-1 fixed, the first full 3-epoch run (v1) still failed
to stop: at supervised stop positions `P(<|im_end|>) ≈ 1.4e-4`, and greedy
decoding looped junk. The token was the supervised label on every row, yet never
became likely.

**Root cause.** `<|im_end|>` is an atomic special token that essentially never
appeared in Qwen3-Base pretraining; its embedding row is near-random
(‖row‖ ≈ 0.38 vs ≈ 1.6 for trained tokens). At the SFT learning rate (5e-6) and
one supervised occurrence per row, three epochs moved the row only ~2% of the
distance needed — meanwhile supervision actively suppressed the base's natural
`<|endoftext|>` stop. Net: stopping got *worse* over training.

**Evidence.** Base-vs-SFT embedding-row L2 norms unchanged (0.375); SFT moved the
`<|im_end|>` row 0.031 vs 0.0007 for a typical row (44× more, still tiny). v1
final probe: top-5 next-token after an answer did not include `<|im_end|>`.

**Fix.** Warm-start the row from the trained document-EOS before training
(`train_qwen_base.py`; embeddings are tied, so this fixes the output logit too):
```python
emb[151645] = emb[151643]   # <|im_end|> := <|endoftext|>
```
**Result.** v2 converged stopping by ~step 150; final `P(<|im_end|>) = 0.318`
(the argmax, logit 33.75); greedy decoding stops cleanly. This is the deep reason
the book's OLMo recipe (plain-BPE role markers + recycled `<|endoftext|>`) is
frictionless to SFT from a base model and a clean atomic-token ChatML is not.

---

## BUG-3 (P1) — cu126 PyTorch has no Blackwell (sm_120) kernels

**Symptom.** First GPU run died at the first CUDA op:
`CUDA error: no kernel image is available for execution on the device`
(`sm_120` not in the build's supported list).

**Root cause.** `pyproject.toml` pinned the x86_64 torch source to the cu126
index. cu126 wheels predate consumer Blackwell (RTX 50-series, compute
capability 12.0) and ship no `sm_120` kernels.

**Evidence.** `torch.cuda.get_device_capability() == (12, 0)`; cu126 wheel
supports up to `sm_90`. After the fix: `torch 2.12.0+cu130`, a bf16 GPU matmul
succeeds.

**Fix.** Point the x86_64 source at the cu130 index in `pyproject.toml`.

---

## BUG-4 (P1) — new `load_model` signature broke the original `train.py`

**Symptom.** Running the unchanged OLMo entry point (`instruction_tuning.train`)
would raise `TypeError: load_model() missing 3 required positional arguments`.

**Root cause.** The added template/eos/stop-id parameters were inserted as
required positionals: `load_model(cfg, chat_template, tok_eos, gen_eos_tok_ids,
device)`. `train.py` still calls `load_model(cfg, device)`.

**Fix.** Make them keyword-only with defaults, so both entry points work:
```python
def load_model(cfg, device, *, chat_template=None, eos_token=None,
               generation_eos_ids=None):
```

---

## BUG-5 (P1) — trained model was never saved

**Symptom.** After 3 epochs the process exited; no checkpoint on disk. Nothing to
evaluate, serve, or continue from.

**Root cause.** The training loop (inherited from `train.py`) had no save step.

**Fix.** Added a final sample pass + `model.save_pretrained` /
`tokenizer.save_pretrained` to `train_qwen_base.py`, gated on a new
`Config.output_dir`. Saving the tokenizer matters: it carries the custom chat
template and eos, making the checkpoint self-describing (which the chat UI then
relies on).

---

## BUG-6 (P2) — chat UI crashed on Gradio 6 (`type` argument removed)

**Symptom.** `chat_app.py` failed to launch:
`TypeError: ChatInterface.__init__() got an unexpected keyword argument 'type'`.

**Root cause.** Written against the Gradio 5 API (`gr.ChatInterface(type=
"messages", ...)`). The resolved version was **6.17.3**, where `type` was removed
(the messages format is now the only one). A static lint never catches an API
that moved between major versions — only running it does.

**Evidence.** `inspect.signature(gr.ChatInterface.__init__)` on 6.17.3 has no
`type` parameter.

**Fix.** Dropped the argument; `respond()` already produced the messages format.

---

## BUG-7 (P2) — Gradio 6 `Textbox` dropped `show_copy_button`

**Symptom.** Adding the compare view crashed at launch:
`TypeError: Textbox.__init__() got an unexpected keyword argument 'show_copy_button'`.

**Root cause.** Same family as BUG-6: another Gradio 5→6 API removal. The copy
button is now default behavior with no constructor flag.

**Fix.** Dropped the argument. Also added a pre-flight (`build_demo()` constructs
all components without launching the server) so the *next* such drift is caught
in ~2s instead of after a 30–60s import + bind cycle.

---

## BUG-8 (P3) — `pkill -f chat_app` killed its own launcher

**Symptom.** A cleanup command exited 144 and the server never started.

**Root cause.** `pkill -f chat_app` / `pgrep -f "...chat_app"` match *any*
process whose command line contains the pattern — including the very shell
running the command. It signalled its own parent.

**Fix (operational).** Don't pattern-kill by a string the kill command itself
contains; select a fresh port instead, or target explicit PIDs / the listening
socket. No code change.

---

## Cross-cutting lesson

Five of these (BUG-1, -2, -3, -5, -6) were invisible to linting and unit
imports — they only surfaced by **actually running** the train loop on the
target GPU and **launching** the UI. BUG-1 and BUG-2 in particular produced a
model whose loss curve looked fine and whose samples read plausibly, yet which
was behaviorally broken (never stops). The general pattern: token-identity and
stop-behavior bugs are silent in aggregate metrics; verify them with a direct
probe (`P(eos)` at the supervised position, greedy decode to a real stop).
