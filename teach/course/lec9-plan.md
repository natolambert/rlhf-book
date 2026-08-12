# Lecture 9 plan — Over-Optimization and Regularization

> **Update (2026-07-15):** this lecture was **split into two decks** — Lecture 9,
> *Over-Optimization and RLHF's Bad Reputation* (ch. 14 + Appendix B,
> `lec9-chap14-appb-overoptimization.md`) and Lecture 10, *Regularization and KL Control*
> (ch. 15, `lec10-chap15-regularization.md`) — to align each video with one type of content
> (narrative/empirical vs. math/technical), matching the Lecture 3/4 precedent. The outline
> below predates the split but remains the content source for both decks.

Design outline for the Lecture 9 course deck (`teach/course/lec9-chap14-15-overoptimization-regularization.md`),
covering book **Chapters 14 (Over-Optimization), 15 (Regularization), and Appendix B ("Beyond 'Just Style'")**,
in the polished Lecture 8 style: talked-over slides, sentence-case titles, framing-question openers,
`<!-- step -->` math unrolls that never skip a line, verbatim-artifact conversation cards, full-bleed figures,
and a closing course roadmap.

> Note: a Codex draft of this lecture exists as **PR #478** (59 slides). This is an independent take in Nathan's
> voice on the same target file; it can supersede or be diffed against that draft.

**Narrative arc:** the problem (over-optimization / Goodhart) → the control (KL regularization, explicit +
implicit) → why it matters even for "style." **The spine of Part 2: reverse KL is everywhere** — the standard
KL penalty is a reverse KL estimated on the policy's own samples (ch 15), the RL objective itself is equivalent
to reverse KL to the reward-tilted optimal policy (ch 15), and Lecture 7's on-policy distillation was reverse KL
to the teacher (ch 12, which explicitly forward-referenced "why reverse KL is better" to Chapter 15 — this
lecture pays that off).

## Revision notes (second pass, grounded in a full chapter read)

1. Made "reverse KL is everywhere" the unifying claim of Part 2 (was three disconnected facts).
2. Restructured the Part 2 math to the chapter's actual two-derivation narrative ("SFT and RL are the two
   directions of KL"); the optimal-policy middle is *recalled from Lecture 6* (its "DPO Derivation P1/4" derives
   the identical `π★ = (1/Z)π_ref·exp(r/β)`), spending the step budget on the new `KL(π_θ‖π★) ∝ −J_RL/β` expansion.
3. Added the chapter's best teaching beat: the *naive expectation flip* — mode-covering SFT "should" forget less;
   the opposite holds because LLMs are multimodal (staged as a step-reveal on the mode-intuition figure).
4. Strengthened RL's Razor: the ablation (on-policy data fully accounts for the effect; **negative gradients
   have no discernible effect**) and the practical instrument (forgetting ≈ f(KL) at R²=0.96, measured on the
   *new task's* data).
5. Over-refusals get their own slide with the verbatim kill-a-Linux-process card — and the chapter's actual
   argument: "too much RLHF" is a *misattribution* (deployment settings + data-curation goals, not the algorithm).
6. Hook firmed to the verbatim April-2025 GPT-4o "god and prophet" exchange; Part 1 now *closes* on the
   Misalignment section that ties the hook back to Lecture 8's preference-data biases.
7. Part 3 opens with the Tülu 3 SFT-vs-DPO completion pair as a goldfish-style A/B card slide (reusing lec8's
   `.poem-ab` CSS) + a plug for rlhfbook.com/library.
8. New "measuring KL in practice" slide from ch 15's implementation section (+ k1/k2/k3 cross-ref to Q&A 2;
   reuse the existing `schulman2020klapprox` refs.bib key, not the book's duplicate `schulman2016klapprox`).
9. More verified cross-lecture callbacks: margin loss ← Likert rating deltas (Lec 8's new Likert slide);
   language-switching symptom ← Lec 5's R1-Zero; per-token KL mechanics ← Lec 3; "proxy for a proxy" ← Lec 8's
   objective-mismatch aside as the opening bridge.
10. Appendix B extras: AlpacaEval *and* WildBench adding linear length corrections; the verbatim Qwen quote;
    Starling Beta as the done-right positive vs. DNO as the over-claim; "models match the *average labeler*"
    as the closing loop back to Lecture 8.
11. Cut: a standalone optimal-policy unroll (folded into #2); Gao's scaling-law functional forms stay out
    (ch 14 is deliberately equation-free on the quantitative side — the deck follows the book).

## Slide outline (~34 authored; ~46 rendered with unrolls)

### Open
1. Title (`title-sidebar`) — "Lecture 9: Over-Optimization and Regularization" / "Post-training course.
   Chapters 14, 15 & Appendix B."
2. Framing question (section-break): *Your reward-model score keeps climbing. Why is the model getting worse?*
3. Framing question 2 (section-break): *How much should we let the model change to earn more reward?*
4. Hook: the GPT-4o sycophancy incident (Apr 2025) — ```conversation card with the verbatim "god"/"prophet"
   exchange (ch 14); rolled back within days (The Verge link in notes). "This lecture is why a healthy-looking
   reward curve produces this."
5. "This lecture" (`columns` + accent box): 1) Over-optimization — Goodhart in RLHF (ch 14) 2) Regularization —
   explicit and implicit KL (ch 15) 3) Beyond "just style" (App B). Recall from Lecture 8: RM accuracy is
   *a proxy for a proxy* — today: what optimizing the proxy hard does.

### Part 1 — Over-optimization (ch 14)
6. Section break.
7. Goodhart's law — the 1984 original + "when a measure becomes a target…" [@goodhart1984problems;
   @hoskin1996awful]. RL "pulls all the possible increase in reward out of the environment"; the RM is at best
   *correlated* with real quality [@schulman2023proxy].
8. Over-optimization ≠ overfitting: overfitting = same task, different split; over-optimization = the metric
   was never right, and the model *genuinely* improves at it [@zhang2018study]. Concrete gaming: confident
   verbosity; rare-token exploits of RM artifacts.
9. The shape — **full-bleed** `overoptimization.png`: proxy reward climbs, true objective peaks then falls;
   x-axis is steps *or KL distance* (foreshadows Part 2's knob) [@gao2023scaling].
10. It's measurable — `anthropic_overoptimization.png`: split the preference data into train/test RMs;
    train-RM gains stop transferring ~150K samples [@bai2022training] `cite-right`.
11. Qualitative symptoms: "As an AI language model…", hedging, over-apologizing, sycophancy
    [@sharma2023towards] — the Lecture 8 bias list, amplified into behavior.
12. Over-refusals — "too much RLHF"? ```conversation card: the verbatim kill-a-Linux-process refusal (Llama 2
    chat) + Claude 2.1's one-liner [@touvron2023llama]. The rebuttal: deployment settings + data-curation goals,
    not the algorithm; measured by XSTest [@rottger2023xstest]; fixed with data.
13. Why it happens & the mitigation menu: approximation / estimation / optimization error [@schulman2023proxy];
    bigger policies, RM ensembles [@coste2023reward], optimizer changes [@moskovitz2023confronting]; DAAs
    over-optimize too [@rafailov2024scaling]; BoN spends far less KL than online RL. **The main lever is the KL
    penalty → Part 2.** (Notes: implicit user feedback + smoother-reward-surface risk; Mallows / Plackett–Luce.)
14. Misalignment closes Part 1: over-optimization *enables* misalignment — sycophancy reinforced by preference
    data that overweights supportive/confident over accurate; back-ref the hook [@sharma2023towards;
    @zhuang2020consequences].

### Part 2 — Regularization (ch 15)
15. Section break: "Part 2: Regularization — keeping the policy close."
16. What breaking looks like: reasoning that reads well with wrong answers, repeated text, **language
    switching** (recall Lecture 5: R1-Zero, language-consistency rewards), special-character spam.
17. The KL penalty: `r = r_θ − λ_KL·D_KL(π_RL‖π_ref)` [@jaques2017sequence; @jaques2020human]. Lecture 3 gave
    the per-token mechanics (`r̃_t = −β·KL_t`); today: why it's there, and the *direction* — this penalty is a
    **reverse KL**, estimated on the policy's own samples. "KL distance" = optimization distance (not a true
    metric). The x-axis of slide 9 *is* this quantity.
18. Measuring KL in practice: `D_KL(P‖Q) = E_{x~P}[log P − log Q]`; condensed 5-step snippet (generate →
    forward both models → log-softmax → gather sampled tokens → sum & subtract); watch the KL curve — a huge KL
    usually means a bug; k1/k2/k3 estimators → Q&A 2 [@schulman2020klapprox].
19. Pivot to implicit regularization — "SFT memorizes, RL generalizes" [@chu2025sft]: GeneralPoints (J/Q/K rule
    shift) and V-IRL (absolute→relative directions); RL improves OOD 80.8→**91.8%** while SFT collapses
    80.8→**1.3%** — destroying spatial reasoning the base model had.
20. The frame: **SFT and RL are the two directions of KL** — two-column mirror of Lecture 7's slide (forward =
    sample the target, mass-covering; reverse = sample yourself, mode-seeking). "Chapter 12 promised the *why*
    in Chapter 15 — here it is."
21. Derivation A (`step`, 2 frames): SFT ≡ forward KL — split the expectation; the first term is −H(π★),
    constant; `boxed: KL(π★‖π_θ) ∝ L_SFT`.
22. Derivation B (`step`, 3 frames): RL ≡ reverse KL — objective → min form → *recall Lecture 6* (partition
    function; `π★ = (1/Z)π_ref·exp(r/β)`, the DPO starting point, stated not re-derived) → expand `KL(π_θ‖π★)`
    line-by-line (the chapter's 4-line aligned block) → `boxed: KL(π_θ‖π★) ∝ −J_RL/β`.
23. The twist (`step`): naive expectation — mode-covering *should* preserve modes, so SFT should forget less →
    **the opposite holds**; the unimodal intuition breaks on multimodal LLMs. Reveal
    `retaining_by_doing_mode_intuition.png`: forward KL stretches across modes and drags mass off the old one;
    reverse KL shifts the new mode and leaves the old alone [@chen2025retainingdoingroleonpolicy] `cite-right`.
24. RL's Razor — `rl_razor_motivation.png`: among high-reward solutions, on-policy RL lands on the KL-closest
    one; forgetting ≈ f(KL) with **R² = 0.96, measured on the new task's data** (a cheap forgetting predictor);
    ablation: **on-policy data fully accounts for it — negative gradients don't matter** [@shenfeld2026rls]
    `cite-right`.
25. Other regularization: pretraining gradients (InstructGPT, `+γ·E_pretrain[log π]`) [@ouyang2022training];
    DPO+NLL [@pang2024iterative]; margin loss `−m(y_c,y_r)` from annotator rating deltas — **the Likert scales
    from Lecture 8** [@touvron2023llama]. (Notes: RPO [@adler2024nemotron], REBEL [@gao2024rebel]; most such
    tricks are scaffolding that disappears in the next model generation.)
26. Part 2 recap (accent box): explicit reverse-KL penalty · implicit on-policy bias (razor) · auxiliary losses
    — and the spine: *every RL-side objective here points the KL the same way.*

### Part 3 — Beyond "just style" (App B)
27. Section break.
28. A/B artifact (reuse `.poem-ab`): Tülu 3 70B **SFT vs. DPO**, same prompt ("What is RLHF?"), truncated cards
    — prose wall vs. structured/bold/listed [@lambert2024t]. "Same base model. What did preference tuning change
    — and is it *just* style?" Plug **rlhfbook.com/library**.
29. Style is substance: "just style transfer" held the narrative back; style is why retellings become
    bestsellers; Llama 3 Instruct's Arena standing came from personality + succinctness [@dubey2024llama].
30. The chattiness balance: gains on AlpacaEval/MT-Bench don't transfer proportionally to Arena or real use;
    verbatim Qwen quote (DPO up on preference eval, *down* on benchmarks [@qwen]); DPO/PPO feedback loops can
    trade math/coding for chat [@ivison2024unpacking]; Olmo 3 shipped the higher-math/code checkpoint over the
    chat-maxed one [@teamolmo2025olmo3].
31. Benchmark gaming — `dno-figure.png`: DNO's 7B "beats GPT-4" on AlpacaEval (Apr 2024) [@rosset2024direct];
    Self-Rewarding LMs' unrealistic scores [@yuan2025selfrewardinglanguagemodels]; it got so bad AlpacaEval
    *and* WildBench added **linear length corrections**. Done right: Starling Beta, +10 Arena places with
    length that actually helps [@zhu2024starling; @wang2023openchat].
32. Why does RLHF lengthen answers? Arena: average users prefer complete answers; models match the **average
    labeler** — Goodhart on the length axis, and Lecture 8's collection choices coming home.

### Close
33. Takeaways: proxy optimization eventually over-optimizes — watch the true objective; the KL knob is the main
    explicit control and *reverse* KL is also the implicit one; on-policy data (not negative gradients) is why
    RL forgets less; style is capability, but length is Goodhart's favorite axis.
34. "Go deeper" (surface box): Gao et al. 2023 · RL's Razor · Chu et al. 2025 · book ch. 14/15/App B.
35. "The course so far" roadmap: 0–9 with chapter numbers, **9 (ch. 14–15) = today**; next: Lecture 10 —
    Evaluation (ch. 16) *(tentative)*.
36. Thank you + ```builtwith.

**Optional stretch:** paper-title-page callouts for Gao et al. 2023 and RL's Razor (400 dpi → JPEG ≤ ~1600px);
a slide with Gao's scaling-law functional forms (not in the book chapter — only if going beyond the book).

## Style conventions

Frontmatter cloned from lec8 (Rubik/Poppins, `bibliography: refs.bib`, `figure_captions: true`, footer
"Lecture 9", `#F28482` section-break + progress CSS, left-aligned lists, the `.full-bleed` block, and lec8's
`.poem-ab` CSS for the two A/B slides). Sentence-case titles; talked-over (no cute asides); `<!-- step -->`
unrolls that never skip a line; `D_{\mathrm{KL}}`, `\begin{aligned}` with right-aligned `&& \text{reason}`,
`\boxed{}` results; inline `[@key]` for claims, `<!-- cite-right -->` for whole-slide sources.

## Assets

Copy from `book/images/` → `teach/course/assets/`: `overoptimization.png`, `anthropic_overoptimization.png`,
`retaining_by_doing_mode_intuition.png`, `rl_razor_motivation.png`, `dno-figure.png` (same figures PR #478
added — reuse, don't regenerate). **No other figures needed** — the three new artifacts (GPT-4o exchange,
kill-a-Linux-process refusal, Tülu SFT-vs-DPO pair) are verbatim chapter text rendered as ```conversation cards.
Keep any downloaded paper screenshots ≤ ~1600 px / JPEG for photographic scans (lesson from Lecture 8).

## Bibliography

Copy from `book/chapters/bib.bib` into `teach/course/refs.bib` (verify each; reuse existing keys where
present): `gao2023scaling`, `goodhart1984problems`, `hoskin1996awful`, `schulman2023proxy`, `zhang2018study`,
`coste2023reward`, `moskovitz2023confronting`, `rafailov2024scaling`, `rottger2023xstest`,
`zhuang2020consequences`, `chu2025sft`, `chen2025retainingdoingroleonpolicy`, `shenfeld2026rls`,
`jaques2017sequence`, `jaques2020human`, `pang2024iterative`, `qwen`, `rosset2024direct`,
`yuan2025selfrewardinglanguagemodels`, `zhu2024starling`, `wang2023openchat`, `ivison2024unpacking`,
`teamolmo2025olmo3`, `dubey2024llama`, `adler2024nemotron`, `gao2024rebel`. Already in refs.bib:
`sharma2023towards`, `bai2022training`, `touvron2023llama`, `ouyang2022training`, `lambert2024t`, and
**`schulman2020klapprox`** (use for KL-approx; skip the book's duplicate `schulman2016klapprox`).

## Cross-lecture references (all verified in the decks)

- **Lecture 8** — objective-mismatch aside ("proxy for a proxy") → opening bridge; bias slide → symptoms;
  Likert slide → margin loss; "average labeler" → why-longer.
- **Lecture 7 / Ch 12** — "Forward vs. reverse KL" slide → mirrored frame; pays off Ch 12's explicit "why
  reverse KL is better: Chapter 15" forward-reference.
- **Lecture 6 / Ch 8** — "DPO Derivation P1/4" → π★ recalled, not re-derived.
- **Lectures 3–4 / Ch 6** — "KL divergence as regularization" + per-token shaping → mechanics recalled.
- **Lecture 5 / Ch 7** — R1-Zero language mixing + language-consistency rewards → "off the rails."
- **Q&A 2** — k1/k2/k3 KL estimators → measuring-KL slide.

## Also update (when the deck ships)

- `book/templates/course.html` — add the Lecture 9 course-page card.

## Verification (when building the deck)

1. Live preview via the serve-course-lecture skill (port 8087, `output_dir='teach/course'`); image 200 check.
2. Citation resolver → `MISSING: []` (ignoring the `natolambert` builtwith false-positive).
3. `colloquium export` → eyeball every slide (montage the figure/derivation runs): full-bleed hump, the two
   A/B card slides, each `step` frame adds exactly one aligned line, no overflow.
4. Confirm the three conversation-card quotes match the chapter text verbatim.
