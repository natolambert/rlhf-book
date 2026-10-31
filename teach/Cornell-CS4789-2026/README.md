# Cornell CS 4789/5789 — Guest Lecture (2026)

**Course:** [CS 4789/5789: Introduction to Reinforcement Learning](https://sdean.website/cs4789.html), Sarah Dean, Cornell
**Slot:** Lecture 24, Wednesday 18 November 2026 (listed as "Guest Lecture")
**Neighbors:** L23 Imitation Learning (11/16) → **L24 (this lecture)** → L25 Inverse RL (11/23) → L26 Specification and Implications → L27 Case Study
**Format:** ~60–70 minutes of slides + questions (slot length TBC)
**Audience:** undergraduate / MEng intro-RL students who have derived REINFORCE, baselines, off-policy importance weighting, KL trust regions, and PPO-clip from the course manuscript
**Status:** proposal — outline only, no slides yet

## Talk: Policy gradients at the frontier — RL for language models

*From REINFORCE to post-training, and the bugs that only show up at scale.*

**Abstract.** Every piece of the RL recipe used to post-train today's frontier language models is something this class has already derived: a contextual-bandit policy, the REINFORCE estimator, a baseline, importance weights, and a KL trust region. This lecture assembles those pieces into the modern post-training pipeline — supervised finetuning (imitation learning, last lecture), reward models learned from preferences (inverse RL, next lecture), and RL with verifiable rewards — and then follows them to a live frontier problem: RL for LLMs is *off-policy by accident*, because the inference engine that samples and the trainer that computes gradients disagree numerically even with identical weights. The fix proposed in *Score Centering Stabilizes Off-policy Reinforcement Learning* (Marek & Ryabinin, Together AI, 2026) is the baseline idea from this course applied to the score instead of the reward. Five weeks after the paper appeared, Thinking Machines shipped a "StableReinforce" loss in Tinker, and the paper's first author reverse-engineered it from gradients alone and showed it is REINFORCE + score centering — a rare public confirmation of a piece of frontier RL science.

### Why this slot works

The post-training pipeline lines up with the course's own ordering:

| CS 4789 | Post-training analogue |
|---|---|
| L19–20 Contextual bandits | An LLM is a contextual-bandit policy: $x$ = prompt, $a$ = completion |
| L3–5, L16–17 Policy gradients, trust regions | REINFORCE / RLOO / GRPO / PPO, KL to a reference model |
| L23 Imitation learning | Supervised finetuning (instruction tuning) |
| **L24 (this lecture)** | RL on language models, and what breaks at scale |
| L25 Inverse RL | Reward models learned from human preferences |

### Notation

Use the course manuscript's notation (`rl-manuscript-policyopt.pdf`), not the book's: context $x \sim D_x$, action $a \sim \pi_\theta(\cdot \mid x)$, reward $r \sim R(x, a)$, objective $J(\theta)$, baseline $b(x)$, importance weight $w = \pi_\theta(a \mid x) / \pi_{\theta'}(a \mid x)$. For language, $a = (a_1, \dots, a_T)$ and $\pi_\theta(a \mid x) = \prod_t \pi_\theta(a_t \mid x, a_{<t})$. For Part 3 introduce the paper's split of one policy into a **sampler** $q$ (inference engine) and a **learner** $p_\theta$ (trainer), with per-token score $s_v = \nabla_\theta \log p_\theta(v \mid x, a_{<t})$.

## Narrative arc

1. **You already know every piece.** The manuscript's REINFORCE, baseline, off-policy, and GRPO (Exercise 3) material *is* the LLM recipe.
2. **Here is how the pieces assemble** into post-training: SFT → reward model → RL, and the shift from RLHF to RLVR.
3. **A frontier failure:** at scale, sampling and learning come from different numerical systems, so $q \neq p_\theta$ even at the same $\theta$ — every RL run is off-policy.
4. **The fix is your baseline lecture, applied to the score.** On-policy $\mathbb{E}_{p}[\nabla \log p] = 0$ is the fact that made baselines free; off-policy it fails, and the leftover term is a "drift" that silently distills the learner toward the sampler.
5. **And we can watch a frontier lab ship it.** Tinker's StableReinforce, reverse-engineered from gradients, matches REINFORCE + score centering.

## Slide outline (~40 authored slides)

Source tags: `reuse: <deck> "<slide title>"` means lift with light edits from `teach/course/`; `adapt` means rework into the course notation; `new` means write from scratch.

### Open (3)

1. **Title** — "Policy gradients at the frontier: RL for language models" / "Guest lecture, CS 4789 — Lecture 24". `new` (frontmatter cloned from `teach/SALA-2026/talk.md`, eyebrow "Cornell CS 4789").
2. **Framing question** (section-break) — *You've derived REINFORCE for a movie recommender. What changes when the policy is a 30B-parameter language model?* `new`
3. **Where today sits** — the L23 → L24 → L25 strip from the table above: SFT ↔ imitation learning, reward model ↔ inverse RL, and today is the RL in the middle. `new`

### Part 1 — The LLM as a contextual-bandit policy (~8)

4. Section break.
5. **The mapping** — context $x$ = prompt, action $a$ = full completion, $\pi_\theta(a \mid x) = \prod_t \pi_\theta(a_t \mid x, a_{<t})$, so $\log \pi_\theta(a \mid x) = \sum_t \log \pi_\theta(a_t \mid \cdot)$ and the score is a sum of per-token scores. The action space is $|\mathcal{V}|^T$. `adapt` from lec3 "Language models as episodic MDPs".
6. **Bandit-style vs. MDP-style** — one reward per completion (bandit, REINFORCE/RLOO/GRPO) vs. per-token credit (MDP, PPO+GAE); most LLM RL is bandit rewards with token-level log-prob gradients. `reuse: lec4 "Bandit-style vs MDP-style RLHF"`
7. **The three-stage recipe** — InstructGPT: SFT → reward model → RL; label each stage with its CS 4789 lecture (L23 / L25 / today). `reuse: lec1 "InstructGPT's 3-step RLHF recipe"` (`instructgpt.jpg`)
8. **The RLHF objective** — $\max_\theta \mathbb{E}_{x, a \sim \pi_\theta}[r_\phi(x,a)] - \beta\, \mathrm{KL}(\pi_\theta \Vert \pi_{\text{ref}})$; preview that $r_\phi$ is learned from preferences on Monday (Bradley–Terry), so don't derive it here. `adapt` from lec1 "The RLHF objective, unpacked".
9. **RLHF vs. RLVR** — swap the learned reward for a verifier (math answer checks, unit tests); same algorithms. `reuse: lec1 "Comparing classical RL vs. LLM RLHF and RLVR"` + `rlvr-system.png`.
10. **The scaling RL era** — o1 train-time scaling plot, DeepSeek-R1-Zero, Olmo 3.1. `reuse: lec1 "o1: Training-time scaling (with reinforcement learning!)"`, `lec1 "DeepSeek-R1-Zero: RL scaling"` (`o1-train-time.png`, `deepseek-r1-zero-figure1-training.png`, `olmo31-extended-rl.jpeg`).
11. **Exercise 3 from the manuscript is literally the frontier recipe** — the manuscript's Exercise 3 (math word problems, $r \in \{0,1\}$, $G$ samples per context) is the setting DeepSeek-R1 and Olmo 3 trained in. Bridge into Part 2. `new`

### Part 2 — From the course manuscript to the LLM recipe (~11)

12. Section break.
13. **REINFORCE, recalled in their notation** — $\theta \leftarrow \theta + \alpha \frac{1}{n}\sum_i \nabla \log \pi_\theta(a_i \mid x_i)\,(r_i - b(x_i))$; for an LLM the gradient flows through every token's log-prob. `adapt` from lec3 "REINFORCE" / "REINFORCE with baseline".
14. **Why baselines are free** — $\mathbb{E}_{a \sim \pi_\theta}[\nabla \log \pi_\theta(a \mid x)] = \sum_a \nabla \pi_\theta(a \mid x) = \nabla 1 = 0$, written out in `<!-- step -->` frames. **Plant this identity — Part 3 hinges on it.** `adapt` from lec3 "Baselines keep the estimator unbiased" (3 slides → 1 stepped slide).
15. **RLOO and GRPO are choices of $b(x)$** — leave-one-out mean vs. group mean + std normalization; GRPO = the manuscript's Exercise 3 estimator $\hat\nabla J_{\text{GRPO}}$. Answer Exercise 3's question: the std division makes it biased. `reuse: lec3 "RLOO: Leave-one-out baseline"`, `lec3 "GRPO advantage"`.
16. **RLOO worked example** — four completions, rewards, baselines, advantages. `reuse: lec3 "RLOO worked example"`
17. **The manuscript's PPO margin note → the real clipped objective** — "discard datapoints once $w_i > 1+\epsilon$ for positive advantage …" becomes $\min(w A, \mathrm{clip}(w, 1\pm\epsilon) A)$. `reuse: lec3 "PPO core idea 2: Importance sampling"`, `lec3 "The PPO clipped objective (put pieces together)"`.
18. **Clipping visualized** — `reuse: lec3 "PPO clipping visualized"` (`ppo-viz-4x.png`).
19. **Two KLs, don't confuse them** — the manuscript's trust region is KL between successive iterates ($\pi_{\theta'}$ vs. $\pi_\theta$); RLHF adds a KL to a frozen *reference* model. `reuse: lec3 "Two "old" policies — don't confuse them"` + `lec10 "A KL penalty to the reference model controls reward"`.
20. **Why no value function** — PPO needs a critic the size of the policy (four models in memory); the group baseline replaced it for reasoning RL. `adapt` from lec3 "GRPO vs PPO".
21. **Loss aggregation** — per-sequence vs. per-token normalization changes the effective weighting of long completions. `reuse: lec4 "Loss aggregation strategies"` (one slide, keep light).
22. **Part 2 recap** — every algorithm is $\nabla \log \pi_\theta \times$ (some centered reward), optionally importance-weighted and clipped. `new`

### Part 3 — Off-policy by accident (~12)

23. Section break: "Off-policy by accident".
24. **The systems picture** — generation runs in an inference engine (vLLM / SGLang), gradients in a training framework (FSDP etc.), often asynchronously on separate GPUs. `reuse: lec4 "Synchronous vs. asynchronous training"` (`async_v_synch_rl.png`) + `lec4 "On-policy vs. off-policy"`.
25. **Same weights, different distribution** — different kernels, precision, batch composition, and quantized KV caches mean the sampler $q$ and learner $p_\theta$ disagree even at identical $\theta$. MiniMax-M1's train-vs-inference token-probability plot. `reuse: lec5 "MiniMax-M1 (Jun. 16, 2025): A fun RL paper!"` (`minimax-m1-logprob-mismatch.png`); cite Yao et al. on off-policy RL frameworks and the Thinking Machines batch-invariance post (making the engines agree exactly is possible but costs throughput).
26. **Your off-policy fix: importance weights** — manuscript §1.3.2: $\hat\nabla J = \frac{1}{n}\sum_i w_i \nabla \log \pi_\theta(a_i \mid x_i)\, r_i$; unbiased but high-variance, and over a 1,000-token completion the sequence ratio explodes. `new`
27. **What the field does: truncate, mask, clip** — TIS caps $w$ at $C$ (one-sided, e.g. $C=2$); MIS / IcePop masks tokens outside a ratio band; PPO/DAPO clip; DPPO masks by an estimated TV divergence. All trade bias for variance. `new` (TIS from `book/chapters/06-policy-gradients.md` §"Truncated Importance Sampling") + `reuse: lec10 "A trust region on the sampling distribution"` (`dppo-ratio-vs-tv.png`).
28. **Drift from the sampler is now the binding failure** — in our TMax terminal-agent work, instabilities came from drift relative to the *sampler* (long multi-turn rollouts, async, engine mismatch), not drift from init. `reuse: lec10 "2026: The trust region / KL pen. is moving on"` (trim to the TMax bullet).
29. **Back to the identity** — on-policy, at each prefix $\mathbb{E}_{v \sim p}[s_v] = 0$. Off-policy, the expected score under the sampler is $\bar s = \sum_v q_v\, s_v \neq 0$. "The fact that made your baselines free is gone." `new`
30. **The decomposition** (`<!-- step -->`, one manipulation per frame) — start from the per-token update $\mathbb{E}_q[R\, s_{a_t}]$; apply $\mathbb{E}[AB] = \mathbb{E}[A]\,\mathbb{E}[B] + \mathrm{Cov}(A,B)$; substitute $\mathbb{E}_q[s_{a_t}] = \bar s$; label the two terms:
   $$\mathbb{E}_q[R\, s_{a_t}] = \underbrace{\mathbb{E}_q[R]\,\bar s}_{\text{drift}} + \underbrace{\mathrm{Cov}_q(R,\, s_{a_t})}_{\text{signal}}$$
   Only the covariance term carries information about which tokens earned reward. `new`
31. **What drift does** — $\bar s = \sum_v q_v \nabla \log p_v$ is the gradient of $-\mathrm{CE}(q, p_\theta)$, so the drift term is *cross-entropy distillation toward the sampler*, scaled by the mean reward at that prefix. The sampler is then refreshed from the learner, so the bias compounds every step rather than converging (unlike offline distillation from a fixed teacher). `new`
32. **The surprising consequence** — online, all-positive $\{+1, 0\}$ rewards are the *least* stable setting (no negative rewards to cancel drift); group-centered rewards (your GRPO baseline) are the most stable. Offline the ordering flips. Paper Figure 2 (Qwen3-1.7B, Countdown). `new`
33. **Score centering: a baseline for the score** — replace $s_{a_t}$ with $\tilde s_{a_t} = s_{a_t} - \bar s$:
   $$\boxed{\mathbb{E}_q[R\, \tilde s_{a_t}] = \mathrm{Cov}_q(R,\, s_{a_t})}$$
   Drift cancels exactly, with no ratios, masks, or clipping; with no mismatch the correction is zero. Side-by-side with slide 14: course baselines center $r$; score centering centers $\nabla \log p$. `new`
34. **Making it cheap** — computing $\bar s$ needs the sampler's full next-token distribution; keep the sampler's top-$k$ ($k = 128$) probabilities and model the tail with the learner. As a loss: $L = -R\,\big(\log p_{a_t} - \sum_{v \in H} \mathrm{sg}[q_v - \rho\, p_v] \log p_v\big)$. <1% wall-clock overhead; composes additively with TIS/MIS (and vanilla, un-truncated IS already cancels drift — which is why truncation brings it back). `new`
35. **Results** — paper Figure 1 (Qwen3-30B-A3B-Base, INTELLECT-2 math, increasingly aggressive sampler quantization): with FP4 KV cache, uncorrected PG collapses within 200 steps while score centering reaches 52% (TIS 51%); with an INT8 sampler + INT4 KV cache, score centering reaches 30%, TIS 12%, everything else < 5% training accuracy. Optional second figure: Figure 3 weight-noise sweep (Qwen3-0.6B). `new`, `cite-right` the paper.

### Part 4 — A rare confirmation of frontier RL science (~6)

36. Section break: "Did a frontier lab use this?"
37. **Why we usually can't know** — frontier RL losses and infrastructure are unpublished; public papers rarely get confirmed or refuted by what labs actually run. `new`
38. **Tinker ships StableReinforce (21 Oct 2026)** — Thinking Machines' post-training API adds a REINFORCE variant "that keeps your RL runs stable, even when learning drifts from sampling." The docs expose the interface but not the loss: sampling must record extra "stability info" (`record_stability_info=True`), and there is an optional sampler–learner KL term with $k_1$ / $k_3$ estimators (callback to the course's KL estimators; `reuse: qa-02 "Reading: How to *estimate* KL in practice"` as a pointer). `new` (screenshot of the announcement + docs snippet).
39. **Reverse-engineering a loss from its gradients** (`<!-- animate: bullets -->`) — Martin Marek's thread (25 Oct 2026): (1) Tinker has no SGD, so use Adam with $\epsilon = 10^4$, making the update $\approx -\text{lr}\cdot g/\epsilon$ and exposing the raw gradient; (2) train a rank-1 LoRA on the unembedding so the adapter gradient equals the logit gradient up to scale; (3) for a single-token update, only **8 logits** have nonzero gradient — plain REINFORCE touches all of them. `new` (thread figures, cited).
40. **The match** — hypothesis: REINFORCE + score centering with $\bar s$ estimated from 8 randomly sampled tokens (unbiased, higher variance — vs. the paper's deterministic, slightly biased top-$k$). Implemented as a custom Tinker loss, the gradients match StableReinforce within numeric precision across all layers. `new` (thread's gradient-match figure).
41. **The honest caveat** — this is a reverse-engineering by the paper's own first author, not a statement from Thinking Machines; the evidence is a gradient match, which is strong but not an admission. Still, it is the closest thing we get to seeing a public RL idea land inside a frontier system within weeks. `new`
42. **The lesson for this class** — the frontier problem lived entirely in intro-course primitives: the zero-mean score, baselines, importance weights, KL. `new`

### Close (3)

43. **Takeaways** — (1) an LLM is a contextual-bandit policy and post-training is IL → IRL → RL; (2) GRPO is your Exercise 3; (3) at scale every run is off-policy because sampler ≠ learner; (4) score centering restores $\mathbb{E}[\text{score}] = 0$, the property baselines rely on; (5) sometimes you can verify frontier practice from the outside. `new`
44. **Next time: inverse RL** — in LLMs, that's the reward model: learn $r_\phi$ from pairwise preferences with Bradley–Terry. One teaser figure. `reuse:` figure `pref_rm_training.png` (from lec2's reward-model section); no derivation.
45. **Resources + thank you** — rlhfbook.com (book + free course videos), the score-centering paper and code, Tinker docs, Marek's thread. `new`

## Reuse inventory

**From `teach/course/` decks** (titles verified against `##` headings):

- lec1: "InstructGPT's 3-step RLHF recipe", "Comparing classical RL vs. LLM RLHF and RLVR", "o1: Training-time scaling (with reinforcement learning!)", "DeepSeek-R1-Zero: RL scaling"
- lec3: "Language models as episodic MDPs", "Baselines keep the estimator unbiased", "REINFORCE with baseline", "RLOO: Leave-one-out baseline", "RLOO worked example", "PPO core idea 2: Importance sampling", "The PPO clipped objective (put pieces together)", "PPO clipping visualized", "Two "old" policies — don't confuse them", "GRPO advantage", "GRPO vs PPO"
- lec4: "Bandit-style vs MDP-style RLHF", "Loss aggregation strategies", "On-policy vs. off-policy", "Synchronous vs. asynchronous training"
- lec5: "MiniMax-M1 (Jun. 16, 2025): A fun RL paper!"
- lec10: "A KL penalty to the reference model controls reward", "2026: The trust region / KL pen. is moving on", "A trust region on the sampling distribution"
- qa-02: "Reading: How to *estimate* KL in practice"

**Figures to copy** from `teach/course/assets/` into `teach/Cornell-CS4789-2026/assets/`: `instructgpt.jpg`, `rlvr-system.png`, `o1-train-time.png`, `deepseek-r1-zero-figure1-training.png`, `olmo31-extended-rl.jpeg`, `ppo-viz-4x.png`, `async_v_synch_rl.png`, `minimax-m1-logprob-mismatch.png`, `dppo-ratio-vs-tv.png`, `pref_rm_training.png`.

## New assets

- Paper title page for arXiv 2609.20807 via the `paper-page-figure` skill (for slide 33 or 35).
- Score-centering paper Figures 1 and 2 (optionally 3), cropped from the arXiv HTML.
- Tinker announcement + StableReinforce docs snippet (slide 38).
- Marek thread figures: sparse logit gradient (slide 39) and gradient match (slide 40). Cite and link the thread; ask Martin before reusing figures.
- Optional: one TikZ diagram of the sampler/learner loop ($p_\theta$ trains → weights sync → $q$ samples → back to $p_\theta$), in `diagrams/tikz/`, following the repo's arrow and spacing conventions.

## Bibliography

Local `teach/Cornell-CS4789-2026/refs.bib`, copying existing keys from `teach/course/refs.bib` or `book/chapters/bib.bib`:

- Already in `teach/course/refs.bib`: `williams1992simple`, `schulman2017proximal`, `shao2024deepseekmath`, `guo2025deepseek`, `minimax2025minimax_m1`, `noukhovitch2024asynchronous`, `ivison2026tmax`, `qi2026dppo`, `glm5team2026glm5`, `schulman2020klapprox`
- From `book/chapters/bib.bib`: `ionides2008truncated`, `yao2025offpolicy`
- New: `marek2026scorecentering` (arXiv 2609.20807), Thinking Machines batch-invariant inference post, `@misc` for the Tinker StableReinforce docs, `@misc` for Marek's thread (x.com/mrtnm/status/2114145574987702525)

## Open questions

1. Slot length (50 vs. 75 minutes) — at 50 minutes, cut Part 2 to slides 13–17 + 19 and drop slide 35's second figure.
2. Should the reward-model teaser (slide 44) stay, or leave preferences entirely to L25?
3. Live demo? Options: a tiny score-centering vs. vanilla REINFORCE run from `code/`, or a Tinker notebook.
4. Ask Martin Marek for permission to use the thread figures, and whether Thinking Machines has said anything publicly since.
5. Verify the "extremely stable training" phrase Marek quotes from the Tinker docs before putting it on a slide — it is not on the StableReinforce docs page as of 26 Oct 2026.

## Sources

- Course page and manuscript: https://sdean.website/cs4789.html, https://sdean.website/rl-manuscript-policyopt.pdf
- Marek & Ryabinin, *Score Centering Stabilizes Off-policy Reinforcement Learning*, arXiv:2609.20807 (17 Sep 2026); code: https://github.com/martin-marek/score-centering
- Tinker StableReinforce docs: https://tinker-docs.thinkingmachines.ai/tinker/losses/stable-reinforce/
- Martin Marek, reverse-engineering thread (25 Oct 2026): https://x.com/mrtnm/status/2114145574987702525
