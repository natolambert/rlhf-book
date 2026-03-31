# Review Feedback Triage: Targeted Edits Plan

PR: https://github.com/natolambert/rlhf-book/pull/323

## Context

Manning review report (12 reviewers, avg 4.2 rating, Amazon stars avg ~4.2/5). The book is strong -- reviewers who rated it 7-8/10 generally loved the content and just wanted polish. The 4-6/10 reviewers largely wanted a *different* book (more beginner-friendly, less math, etc.) and we should not optimize for them.

Already on main before this PR: RLHF pipeline walkthrough in Ch1 intro, policy gradient intuitions subsection in Ch6.

### Guiding Principle
Filter feedback through: **"Is this a real clarity gap that confuses the target reader (mid-senior ML practitioner), or is this someone wanting a different book?"** Most of the "add more examples / less math" feedback falls in the latter category. We're looking for places where a genuinely knowledgeable reader gets confused because a sentence or two of setup is missing.

---

## Tier A -- High value, near-zero style disruption (1-3 sentences each)

Places where a brief plain-language sentence before a key equation resolves genuine confusion from strong reviewers (#22 PhD math/stats, #33 researcher, #20 practitioner):

| # | Chapter | What | Status | Notes |
|---|---------|------|--------|-------|
| A1 | Ch3, before `@eq:rl_dynam` | Plain-English trajectory equation intro | **DONE (PR #323)** | "A trajectory's overall probability is the product of..." Reviewed: dropped "In words," prefix. |
| A2 | Ch3, before `@eq:rl_opt` | Plain-English RL objective intro | **REVERTED** | Original wording was already fine. Not worth changing. |
| A3 | Ch3, before `@eq:rlhf_opt_eq` | KL penalty setup: what beta does, why the constraint exists | **DONE (PR #323)** | "To prevent the policy from drifting too far from the pretrained model, RLHF adds a KL divergence penalty..." This is the most important equation in the book. |
| A4 | Ch5, before BT loss derivation | Plain-language BT loss sentence | **DONE (PR #323)** | "The resulting loss encourages the reward model to assign a higher score to the human-preferred completion..." |
| A5 | Ch6, around `@eq:policy_gradient_intuition` | Verify policy gradient intuition section | **TODO** | May already be addressed by the intuitions subsection added earlier. Need to verify. |
| A6 | Ch15, before KL divergence definition | Plain-English KL definition | **DONE (PR #323)** | "KL divergence measures how far one probability distribution has drifted from another -- when KL is zero, the two distributions produce identical outputs." |

## Tier B -- Medium value, low disruption (small targeted fixes)

| # | Chapter | What | Status | Notes |
|---|---------|------|--------|-------|
| B1 | Ch3, CartPole dynamics | Rename `temp` variable | **DONE (PR #323)** | Changed to `\alpha` with parenthetical: "combining the applied and centripetal forces". Original `temp` confused readers after thermostat discussion. |
| B2 | Ch3, "No state transitions" | Clarify wording | **DONE (PR #323)** | Added "(the prompt is fixed; the model's completion does not define the next prompt)" |
| B3 | Ch3, bandit problem reference | Move to parenthetical | **DONE (PR #323)** | Was leading the sentence, now parenthetical: "(this single-step structure is sometimes called a bandit problem in the RL literature)" |
| B4 | Ch1, PreFT bullet | Mention RLHF explicitly | **DONE (PR #323)** | Added "via RLHF and related methods" to bullet #2 |
| B5 | Ch1, safety mention | Add safety as early RLHF use case | **DONE (PR #323)** | Cites @dai2023safe and @bai2022training. Replaced overclaimed generic sentence with specific citation. |
| B6 | Ch8, DAAs vs RL section | Add practical decision guide | **DONE (PR #323)** | 2-sentence rule of thumb: DPO for fast iteration, RL for max performance. |

## Tier C -- Skip (style change, diminishing returns, or wrong audience)

| # | What reviewers asked for | Why skip |
|---|--------------------------|----------|
| C1 | "Add more figures/diagrams everywhere" | Book already has key figures. Adding figures is expensive. |
| C2 | "Unpack code incrementally" (#9, 5/10) | "Different book" request. Code is reference-style, not tutorial-style. |
| C3 | "Drop most mathematical notation" (#6, 6/10) | Fundamentally misaligned with book's purpose. |
| C4 | "Add end-to-end worked example across chapters" | Very high effort, structural change. Better as companion notebook. |
| C5 | "Reorder Ch3 to examples-first" (#6, #20) | Structural rewrite. Current order works for target reader. |
| C6 | "Cut or shorten the history chapter" (#20) | Ch2 is already short. History is part of the voice/brand. |
| C7 | "Standardize SFT vs instruction tuning" | Intentionally uses both -- they overlap but aren't identical. |
| C8 | "Add closing section on open questions" | Skipping for now -- lower priority. |

## Remaining Items to Consider

These came up in the review but haven't been triaged yet:

- **Ch3 Table 3.1**: Reviewer #22 asked "why does it not include the policy function?" -- worth checking if adding a Policy row (Standard RL: learned from scratch / RLHF: fine-tuned from pretrained LM) would improve the table
- **Ch6 algorithm comparison**: The existing Table 6.1 is good but reviewers wanted "core idea is..." sentences for each algorithm. Check if the policy gradient intuitions section already covers this.
- **Ch8 DPO notation**: Reviewer #20 praised the gradient derivation term-by-term explanation in Ch8. Could apply the same pattern to other chapters, but low priority.
- **General**: Multiple reviewers (#7, #9) noted some sections "feel like survey paper or literature sections." Not actionable as a sentence-level fix -- this is the book's style for dense-reference sections.

## Review Report Stats

| Chapter | Rating | Votes |
|---------|--------|-------|
| 1 Introduction | 4.0 | 13 |
| 2 History | 3.69 | 13 |
| 3 Training Overview | 4.17 | 12 |
| General | 4.2 | 12 |
