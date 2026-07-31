# Lecture 13 plan — An Introduction to Character Training

Design outline for the Lecture 13 course deck (`teach/course/lec13-chap17-character.md`), covering book
**Chapter 17 (Crafting Model Character and Products)** -- the final content chapter of the book and the
closing lecture of the course. Built in the polished Lecture 12 style: talked-over slides, sentence-case
titles, framing-question opener, `<!-- step -->` math unrolls that never skip a line, verbatim-artifact
conversation cards, and a closing course roadmap that this time ends the course.

**Title:** "An Introduction to Character Training" / subtitle "Constitutions, soul documents, and the
personality of models". **Accent:** `#8E7CC3` (soft violet; section breaks + progress bar).

**Narrative arc:** character training is the strongest evidence that RLHF matured from "alignment"
philosophy into an engineering discipline -- labs now spend frontier effort deciding *who* their models
are. Climb the control ladder: prompting → character in the weights (Open Character Training, CAI-style)
→ three no-gradient methods (persona vectors, the Assistant Axis, persona subnetworks) -- then the
documents that govern it all (Anthropic's constitution lineage vs. OpenAI's Model Spec, and the
abstraction gap between them). Close where the book closes: RLHF as the model↔product interface,
"embedding a secretly human process into the deepest levels of powerful AI tools."

## Structure (~38 authored slides; ~45 rendered)

### Open
1. Title (`title-sidebar`).
2. Framing question (section-break): *Every model ships with a personality. Who wrote it -- and where does it live?*
3. Hook: the soul-document leak -- the name leaked into training data before Anthropic confirmed the
   document existed; a researcher then extracted passages from the weights [@anthropic2025souldoc].
   Details deferred to Part 3.
4. "This lecture" agenda (columns + accent box); recall Lecture 12 -- benchmarks can't see small
   character deltas.

### Part 1 — Character training (ch. 17 opening + Anthropic + OCT)
5-7. Section break; the control ladder (prompt → steer → train, recall Lec 1 system prompts); what
   character training actually is (language-feature pipelines, `Certainly` removal, CAI-style synthetic
   data; hard to measure -- Lec 12; Llama 3 personality → Arena, Lec 9).
8. Claude 3 "character training" verbatim blockquote [@anthropic2024claude] + the artist's-touch line;
   library plug.
9. Askell-on-Lex conversation card + `cai-overview.png` (pays off Lec 7's "Anthropic still uses a
   constitution -- yes, confusing").
10. The maturity thesis (any-trait caveat, engagement edge).
11-12. **Open Character Training** [@maiya2025open] (co-author lecture beat): pipeline figure
   (`oct-pipeline.png`) and robustness figure (`oct-robustness.png`) from the paper source (arXiv
   2511.01689), both resized to 1600px.
13-14. One prompt, six characters: steroids-refusal conversation cards (base + five personas, verbatim,
   `.poem-ab` A/B columns); bridge to Part 2.

### Part 2 — Character without gradients
15-16. Section break; concepts-are-directions lineage (Word2vec → RepE → activation addition).
17-20. **Persona vectors** [@chen2025persona]: contrastive extraction (difference-of-means formula, 40/60
   columns with the pipeline figure), steering h ← h + αv with the α = 0.5/1.5/2.5 gradation, three
   operational uses (monitoring / preventative training / data screening), OCEAN composition
   [@feng2026persona] (poles table, R² > 0.94 linearity, composite sum).
21-26. **The Assistant Axis** [@lu2026assistant]: full-bleed `assistant_axis.png`; poles-of-persona-space
   table; robust contrast-vector definition; persona drift; activation capping `step` unroll (rule →
   p = ⟨h,v⟩ → two cases → boxed ⟨h′,v⟩ = τ; 25th-percentile calibration); turn-16 drift pair as
   `.poem-ab` cards (verbatim).
27-29. **Persona subnetworks** [@ye2026personality]: lottery-ticket framing [@frankle2019lottery];
   three-step training-free recipe (activation stats → importance S = |w|·A → top-K masks); additive vs.
   multiplicative recap box.

### Part 3 — Constitutions, model specs, soul documents
30-31. Section break; OpenAI Model Spec [@openai2024modelspec] + verbatim chain-of-command excerpt
   (fetched from the 2025-09-12 spec revision); callback Lec 8 ("data → behavior stays largely
   unaudited").
32-33. Anthropic's document in three eras (2022 CAI → 2024 character training → 2025 soul doc; Askell
   supervised-learning confirmation [@askell2025soul]) + verbatim soul-doc character excerpt.
34. **The abstraction difference** (two boxes + the chapter's "intent of the process vs. intermediate
   training variables" line; the documents are converging).
35-36. Who a spec is for (designers/developers/public); the open question -- effort, not documents.

### Part 4 + close
37-38. Section break; RLHF as the model↔product interface (features enter at post-training first, then
   backpropagate).
39. The last word: the chapter's final paragraph staged with `step`, ending on the book's closing line.
40-42. Takeaways; "The course, complete" roadmap (no "next" entry -- "That's the course."); thank-you +
   builtwith.

## Sources beyond the chapter (author-approved)

- Verbatim excerpts from the **OpenAI Model Spec** (2025-09-12 revision, "Follow all applicable
  instructions") and the **soul document** (LessWrong extraction post, "Claude's identity" section).
- Two figures from the **Open Character Training** paper source (arXiv 2511.01689):
  `oct-pipeline.png` (fig. 1, the drawio pipeline) and `oct-robustness.png` (the F1 robustness bars).
- Not used (kept chapter-only): steered "evil" completions from Chen et al.; an abstraction-ladder TikZ;
  lab logos on the comparison slide.

## Export gotcha (learned the hard way)

Chromium's print-to-PDF drops images from the page when the deck's total decoded-image memory gets large:
the full-res chapter figures (3387×4221) silently evicted a *different, smaller* image from the printed
PDF while the live HTML preview rendered everything fine. Deck-local asset copies are therefore downsized
(≤2000px wide). Long figure captions inside `rows` also collide with the footer in print -- the two OCT
figure slides carry their attribution via `cite-right` instead of captions.

## Assets

In `teach/course/assets/`: `persona-vectors-pipeline.png` (+`-dark`, 1400px), `assistant_axis.png`
(2000px), `oct-pipeline.png`, `oct-robustness.png` (1600px), reused `cai-overview.png`. Book-resolution
originals stay in `book/images/`.

## Bibliography

13 keys copied from `book/chapters/bib.bib` into `teach/course/refs.bib`: `maiya2025open`,
`chen2025persona`, `feng2026persona`, `lu2026assistant`, `ye2026personality`, `frankle2019lottery`,
`anthropic2024claude`, `anthropic2025souldoc`, `askell2025soul`, `turner2023activation`,
`zou2024representation`, `mikolov2013efficient`, `bas2026actuallysteermultibehaviorstudy`. Already
present: `openai2024modelspec`, `bai2022constitutional`.

## Also update (when the video ships)

- `book/templates/course.html` -- remove the `[Draft]` prefix and add the Watch action to the Lecture 13
  card; reconcile card order with the Lecture 12 card (PR #492) after both merge.
