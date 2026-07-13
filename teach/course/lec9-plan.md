# Lecture 9 plan — Over-Optimization and Regularization

Design outline for the Lecture 9 course deck (`teach/course/lec9-chap14-15-overoptimization-regularization.md`),
covering book **Chapters 14 (Over-Optimization), 15 (Regularization), and Appendix B ("Beyond 'Just Style'")**.
Authored in the polished style established in Lecture 8: talked-over slides, no editorializing "cute" phrasing,
sentence-case titles, framing-question openers, `<!-- step -->` math unrolls, full-bleed figures, paper-screenshot
callouts, and a closing course roadmap.

> Note: a Codex draft of this lecture exists as **PR #478** (59 slides). This is an independent take in Nathan's
> voice on the same target file; it can supersede or be diffed against that draft.

**Narrative arc:** the problem (over-optimization / Goodhart) → the control (KL regularization, explicit + implicit)
→ why it matters even for "style." The regularization half **pays off a forward reference from Chapter 12 /
Lecture 7**: Ch 12 introduced forward vs. reverse KL and explicitly said "we explain *why* reverse KL is better in
Chapter 15." Lecture 9 delivers that.

## Source material

- `book/chapters/14-over-optimization.md` — Goodhart; qualitative symptoms (chattiness, sycophancy, over-refusals);
  quantitative story (Gao et al. 2023 KL-vs-reward "hump"; Bai et al. 2022 train/test RM divergence ~150K samples);
  root causes; mitigations.
- `book/chapters/15-regularization.md` — KL penalty `r = r_θ − β·KL(π_RL‖π_ref)`; optimal policy
  `π*(y|x) = (1/Z)·π_ref(y|x)·exp(r/β)`; forward vs reverse KL (mode-covering vs mode-seeking); forward-KL≡SFT
  (`KL(π*‖π_θ) ∝ L_SFT`); implicit regularization (SFT memorizes / RL generalizes, Chu 2025; Retaining-by-Doing,
  Chen 2025; **RL's Razor**, Shenfeld 2026, forgetting≈f(KL), R²=0.96); pretraining gradients (InstructGPT),
  DPO+NLL, margin loss (Llama 2).
- `book/chapters/appendix-b-style.md` — "Beyond 'Just Style'": style *is* value (Llama 3 Arena scores); the
  chattiness balance; length-gamed benchmarks / over-claims (DNO "beats GPT-4" on AlpacaEval).
- Cross-reference source: `teach/course/lec7-chap12-synthetic-data.md` slide **"Forward vs. reverse KL"**
  (two-column, mass-covering vs mode-seeking) — mirror its layout and say "recall from Lecture 7."

## Style conventions (from `teach/CLAUDE.md` + Lecture 8)

- Frontmatter identical to lec8: Rubik/Poppins fonts, `bibliography: refs.bib`, `figure_captions: true`, footer
  `Lecture 9`, `custom_css` with `#F28482` section-break + progress color, left-aligned lists, and the reusable
  `.full-bleed` block (padding `60px 13px 24px`, `h2 { margin-left: 47px }`, flex-centered image at 98% width).
- **Sentence case** titles; keep acronyms (KL, PPO, GRPO, DPO, RLHF, SFT) capitalized.
- **Talked-over**: no cute asides; the insight goes in delivery, not on the slide.
- Directives: `<!-- layout: section-break -->` for parts, `<!-- columns: X/Y -->`, `<!-- rows -->`,
  `<!-- valign -->`, `<!-- cite-right: key -->`, `<!-- step -->` for derivations, ```box / ```conversation.
- Math: `\begin{aligned}` for equality chains, `D_{\mathrm{KL}}`, `\mathbb{E}`, `\mathcal{L}`, right-aligned
  `&& \text{reason}`, `\boxed{}` the result; **never skip a derivation step** (one aligned line per frame).

## Slide outline (~36 authored slides; `<!-- step -->` unrolls push the rendered count toward ~48)

### Open
1. Title (`layout: title-sidebar`) — "Lecture 9: Over-Optimization and Regularization" / note: "Course on RLHF and
   post-training. Chapters 14, 15 & Appendix B."
2. Framing question (section-break): *"Your reward-model score keeps climbing. Why is the model getting worse?"*
3. Concrete hook (mirrors lec8's poem A/B): a real over-optimized reply — the "As an AI language model… Certainly!"
   / sycophancy pattern, or the **April 2025 GPT-4o sycophancy rollback** as the timely "when it goes wrong" moment
   (like lec8's Arena slide). Use a ```conversation card.
4. "This lecture" (`columns`) + ```box `tone: accent`: 1) Over-optimization (Goodhart) 2) Regularization (KL,
   explicit + implicit) 3) Beyond "just style."

### Part 1 — Over-optimization (Ch 14)
5. Section break: "Part 1: Over-optimization."
6. Goodhart's law — *"When a measure becomes a target, it ceases to be a good measure"* [@hoskin1996awful;
   @goodhart1984problems]. RLHF optimizes a **learned** reward (a proxy), not ground truth.
7. Over-optimization ≠ overfitting: the model genuinely improves at the proxy (RM score ↑) while the true objective
   turns down [@schulman2023proxy].
8. Qualitative symptoms: hedging, over-apologizing, sycophancy [@sharma2023towards], over-refusals (misreading
   "kill a Linux process") [@rottger2023xstest]; ties to Appendix B (Part 3).
9. **Quantitative — full-bleed** `overoptimization.png`: as KL from the SFT model grows, proxy reward keeps rising
   but the gold/true metric peaks then falls (the "hump") [@gao2023scaling]. `class: full-bleed`.
10. Train/test RM divergence — `anthropic_overoptimization.png`: gains on the train RM stop transferring (~150K
    samples) [@bai2022training].
11. Why: approximation / estimation / optimization error; "RL pulls *all* the reward out of the environment."
12. Mitigations preview: bigger policy, RM ensembles [@coste2023reward], better optimizers [@moskovitz2023confronting]
    — and the main lever, **KL regularization**; even DPO over-optimizes [@rafailov2024scaling]. → Part 2.

### Part 2 — Regularization (Ch 15)
13. Section break: "Part 2: Regularization — keeping the policy close."
14. The KL penalty: `r = r_θ(x,y) − β·D_{KL}(π_RL(·|x)‖π_ref(·|x))` — the **same KL term already inside the PPO/GRPO
    objective from Lectures 3–4** [@jaques2017sequence; @jaques2020human]. The x-axis of Part 1's hump curves *is* this KL.
15. Derivation (`<!-- step -->`, 2–3 frames): `max_π E[r] − β·KL(π‖π_ref) ⇒ π*(y|x)=(1/Z(x))·π_ref(y|x)·exp(r/β)`.
    Cross-ref: "the **same optimal policy we used to derive DPO in Lecture 6**."
16. **Forward vs. reverse KL** (two-column, mirror lec7's slide): Forward KL = SFT, *mass-covering*; Reverse KL = RL,
    *mode-seeking*. Open with "recall from Lecture 7 / Chapter 12 — here's the promised *why*."
17. Forward-KL ≡ SFT (`<!-- step -->`): `KL(π*‖π_θ) = −H(π*) + L_SFT ∝ L_SFT`.
18. Implicit regularization: on-policy RL regularizes even without an explicit penalty — "SFT memorizes, RL
    generalizes." Chu et al. 2025: V-IRL OOD, RL 91.8% vs SFT 1.3% [@chu2025sft].
19. Retaining by doing — `retaining_by_doing_mode_intuition.png`: forward KL stretches to cover the target (disrupts
    old modes); reverse KL shifts the new mode without disturbing the old [@chen2025retainingdoingroleonpolicy].
20. **RL's Razor** — `rl_razor_motivation.png`: among high-reward solutions, on-policy RL is biased to the KL-closest
    one → forgets less; forgetting ≈ f(KL), R²=0.96 [@shenfeld2026rls].
21. Other regularization: pretraining gradients (InstructGPT) [@ouyang2022training]; DPO+NLL [@pang2024iterative];
    margin-based RM loss (Llama 2) [@touvron2023llama]. (Cross-ref DPO's implicit KL.)
22. Part 2 recap ```box: explicit KL · implicit on-policy · auxiliary losses.

### Part 3 — Beyond "just style" (Appendix B)
23. Section break: "Part 3: Beyond 'just style.'"
24. Style isn't superficial — RLHF was dismissed as style transfer, but Llama 3's Arena scores came from personality
    + succinctness [@dubey2024llama].
25. The chattiness balance: RLHF lengthens answers (Arena rewards complete answers) but hurts math/coding — "too much
    RLHF" [@ivison2024unpacking; @teamolmo2025olmo3].
26. Length-gamed benchmarks — `dno-figure.png`: DNO claimed a 7B "beats GPT-4" on AlpacaEval but doesn't hold up in
    real use [@rosset2024direct; @yuan2025selfrewardinglanguagemodels].
27. Tie-back: over-optimized style is the *qualitative face* of Ch 14 — Goodhart on the length axis; well-balanced
    RLHF exists [@zhu2024starling].

### Close
28. Takeaways: proxy optimization always over-optimizes eventually; KL (explicit + implicit) is the main control;
    watch the *true* objective, not the proxy; style is capability.
29. "Go deeper" ```box `tone: surface`: Gao et al. 2023, RL's Razor, book ch. 14/15/App B.
30. Course roadmap ("The course so far"): 0–9 with chapter numbers, **9 = today**; optional tentative *next:
    Lecture 10 — Evaluation (ch. 16)*.
31. Thank you (standard closing + ```builtwith).

**Optional (paper-screenshot callouts like lec8):** render title pages of **Gao et al. 2023** (arXiv 2210.10760)
and **RL's Razor** (Shenfeld et al. 2026) at 400 dpi → `teach/course/assets/`, place beside their bullets.
Skippable; the four in-book figures already carry the section.

## Assets to copy (per course convention: lectures reference `assets/…`)

Copy from `book/images/` → `teach/course/assets/`: `overoptimization.png`, `anthropic_overoptimization.png`,
`retaining_by_doing_mode_intuition.png`, `rl_razor_motivation.png`, `dno-figure.png`. (Same figures PR #478 added —
reuse, don't regenerate.) Keep any downloaded paper screenshots ≤ ~1600 px / convert photographic scans to JPEG so
the push stays small (lesson from Lecture 8).

## Bibliography

Add any missing keys to `teach/course/refs.bib`, copying entries verbatim from `book/chapters/bib.bib`. Likely-missing
(verify): `gao2023scaling`, `goodhart1984problems`, `hoskin1996awful`, `schulman2023proxy`, `coste2023reward`,
`moskovitz2023confronting`, `rafailov2024scaling`, `rottger2023xstest`, `chu2025sft`,
`chen2025retainingdoingroleonpolicy`, `shenfeld2026rls`, `jaques2017sequence`, `jaques2020human`, `pang2024iterative`,
`rosset2024direct`, `yuan2025selfrewardinglanguagemodels`, `zhu2024starling`, `ivison2024unpacking`,
`teamolmo2025olmo3`, `dubey2024llama`. (Already present from lec8: `sharma2023towards`, `bai2022training`,
`touvron2023llama`, `ouyang2022training`, `lambert2024t`.)

## Cross-lecture references to weave in

- **Lecture 7 / Ch 12** — forward vs reverse KL (mass-covering / mode-seeking); Lecture 9 delivers the "why reverse
  KL is better" that Ch 12 forward-referenced. (Primary link.)
- **Lecture 6 / Ch 8 (DPO)** — the optimal policy `π* ∝ π_ref·exp(r/β)` and DPO's *implicit* KL.
- **Lectures 3–4 / Ch 6 (RL)** — the KL term already living inside the PPO/GRPO objective.
- **Lecture 5 / Ch 7 (RLVR)** — reward hacking exists even with verifiable rewards.

## Also update

- `book/templates/course.html` — add/confirm the Lecture 9 course-page card (the roadmap already lists it).
- The Lecture 8 roadmap already marks 9 as "next" — no change needed there.

## Verification

1. Live preview via the `serve-course-lecture` skill (served from `teach/course/`, so `assets/…` resolve):
   `serve('teach/course/lec9-chap14-15-overoptimization-regularization.md', port=8087, output_dir='teach/course')`.
2. Citation resolver → `MISSING: []` (ignore the `natolambert` builtwith false-positive).
3. `colloquium export … -o /tmp/lec9.pdf` and **eyeball every slide** (montage the figure/derivation slides),
   checking: the full-bleed hump curve, the forward/reverse KL two-column, each `<!-- step -->` frame builds one line
   at a time, no text/equation overflow.
4. Image 200 check on one `assets/…` figure (per the skill).
