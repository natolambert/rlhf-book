---
title: "Lecture 12: Evaluation"
author: "Nathan Lambert"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "rlhfbook.com/course"
  center: "Lecture 12"
  right: "Lambert {n}/{N}"
custom_css: |
  .slide--section-break { background: #F28482; }
  :root {
    --colloquium-progress-fill: #F28482;
  }
  .slide--title-sidebar h1 {
    font-size: 2.5em;
    letter-spacing: 0;
  }
  /* Bulleted lists should never be centered (markers float, looks bad).
     Target lists only -- leave titles and display-math paragraphs centered. */
  .slide ul, .slide ol, .slide li { text-align: left; }
---

<!-- layout: title-sidebar -->
<!-- valign: bottom -->

# Lecture 12: Evaluation

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">Course on RLHF and post-training. Chapter 16.</p>

---

<!-- layout: section-break -->
<!-- align: center -->
<!-- valign: center -->

## What do evaluations actually tell us about a model?

---

<!-- columns: 55/45 -->
## Claude 4 (May 2025): The benchmarks were "meh" -- the model wasn't

Claude 4 *underperformed Claude 3.7 on some popular coding benchmarks* while being the practically better model -- more reliable, better in agents, the foundation of Claude Code's rise.

What I wrote at the time: "the benchmarks are meh. They can't lead this model to mindshare" -- and yet the release worked. [Full text ↗](https://www.interconnects.ai/p/claude-4-and-anthropics-bet-on-code)

The eval that *did* summarize the release was buried in the system card: a reward-hacking measure showing Claude 4 dramatically outperforming 3.7.

|||

```box
title: The strain
tone: accent
content: |
  "The AI field is strained by being forced to communicate the abilities of their models through benchmarks that don't capture the full picture."

  Small reliability fixes on long-tail tasks can transform how a model *feels* while peak benchmark performance barely moves.
```

---

<!-- columns: 55/45 -->
## Opus 4.6 vs. Codex 5.3 (Feb 2026): The post-benchmark era

Two frontier releases in one week. My honest reaction:

> *"Benchmark-based release reactions barely matter. For this release, I barely looked at the evaluation scores."* [Full text ↗](https://www.interconnects.ai/p/opus-46-vs-codex-53)

What decided it instead: Codex kept a narrow edge on hard coding, but "switching from Opus 4.6 to Codex 5.3 feels like I need to babysit the model" -- usability, trust, and breadth won.

|||

```box
title: The June 2025 prediction
tone: accent
content: |
  "More releases are going to look like Anthropic's Claude 4, where the benchmark gains are minor and the real world gains are a big step."

  The only assessment method left: sustained personal use, across multiple models.
```

---

<!-- animate: bullets -->
## Meanwhile, the launch posts still dazzle (2026)

The eval numbers in release blogs have never looked more impressive -- or been harder to interpret:

- Claude Fable 5: "state-of-the-art on nearly all tested benchmarks" -- but the headline evidence is **partner testimonials on private evals** (10 of 14 carousel quotes cite a benchmark nobody outside that company can audit) [(blog)](https://www.anthropic.com/news/claude-fable-5-mythos-5)
- The most quoted Fable claim isn't a benchmark at all: Stripe's codebase migration, "two months by hand" done **in a day**
- Claude Opus 5: "ARC-AGI 3: Opus 5's score is **three times as high as the next-best model**" -- a ratio, with no absolute number given anywhere in the post [(blog)](https://www.anthropic.com/news/claude-opus-5)
- The charts moved from bar tables to **performance-vs-cost curves** -- more honest, and also harder to compare across labs

The number you can screenshot is not the number that predicts adoption.

---

<!-- columns: 55/45 -->
## Gemini 3 (Nov 2025): Evals crown a king

Google's launch post led with the strongest benchmark sweep of the year: "It significantly outperforms 2.5 Pro on every major AI benchmark. It tops the LMArena Leaderboard with a breakthrough score of 1501 Elo" -- HLE 37.5%, GPQA Diamond 91.9%, SWE-bench Verified 76.2%. [(blog)](https://blog.google/products/gemini/gemini-3/)

The world believed the numbers:

- Analysts: "the current state-of-the-art"; Alphabet's market cap passed Microsoft's within the week
- Benioff: "The leap is insane... It feels like the world just changed, again."
- Dec 1: OpenAI's reported internal **"code red"**

|||

```box
title: The dissent, printed at the time
tone: muted
content: |
  "Having the state of the art model for a few days doesn't mean they've won to the extent that the stock market is implying." -- Gil Luria, D.A. Davidson, Nov 2025
```

---

## ...and the crown moved in months. Weird times.

From my February post: *"The timeline has left them behind 2 months after their coronation, showing Gemini 3 was hailed as a false king."*

- July 2026: Gemini 3.5 Pro reported **months late** -- "coding capabilities, in particular, were short of internal expectations" -- while OpenAI and Meta leapfrogged on exactly the commercial frontier (agentic coding)
- **And yet the product grew anyway**: Gemini app 650M → **950M monthly users** in eight months; ~9% → ~27% of gen-AI web traffic (with ChatGPT's absolute traffic roughly *flat* -- category growth, not defection)

The launch evals predicted **neither** the durable frontier position **nor** the product trajectory. Benchmarks opened the door; what walked through was distribution, reliability, and product.
