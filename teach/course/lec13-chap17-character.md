---
title: "Lecture 13: An Introduction to Character Training"
author: "Nathan Lambert"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "rlhfbook.com/course"
  center: "Lecture 13"
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
  .slide--title-sidebar h1 .title-subtitle {
    display: block;
    margin-top: 0.6em;
    font-size: 0.45em;
    font-weight: 400;
    letter-spacing: 0;
    opacity: 0.75;
  }
  /* Bulleted lists should never be centered (markers float, looks bad).
     Target lists only -- leave titles and display-math paragraphs centered. */
  .slide ul, .slide ol, .slide li { text-align: left; }
  /* A/B comparison cards (from Lecture 8): force both cards to fill their
     column evenly so they read as a matched pair. */
  .slide.poem-ab .colloquium-message { max-width: 100%; width: 100%; padding: 1em 1.1em; }
  .slide.poem-ab .colloquium-conversation { height: 100%; justify-content: center; }
  .slide.poem-ab .colloquium-message-role { font-size: 0.75em; }
  /* Full-bleed image: near-edge figure (98% width), vertically centered,
     with the slide title kept in its normal position. */
  .slide.full-bleed { padding: 60px 13px 24px; }
  .slide.full-bleed h2 { margin-left: 47px; }
  .slide.full-bleed .slide-content { min-height: 0; flex: 1; display: flex; align-items: center; justify-content: center; }
  .slide.full-bleed .slide-content img {
    width: 100%; height: 100%;
    max-width: none; max-height: none;
    object-fit: contain;
  }
---

<!-- layout: title-sidebar -->
<!-- valign: bottom -->

# Lecture 13: An Introduction to Character Training <span class="title-subtitle">Constitutions, soul documents, and the personality of models</span>

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">Course on RLHF and post-training. Chapter 17.</p>

---

<!-- layout: section-break -->
<!-- align: center -->
<!-- valign: center -->

## Every model ships with a personality. Who wrote it -- and where does it live?

---

<!-- valign: center -->
<!-- cite-right: anthropic2025souldoc -->
## Late 2025: the document that leaked out of the weights

Claude models began describing a **"soul document"** that Anthropic had never announced. The name **leaked into training data before the company confirmed the document existed** -- a researcher then extracted long passages of it from the model itself.

The document that defines Claude's character is an artifact *inside* the training pipeline -- important enough to shape the weights, and it surfaced through them.

This lecture: how personalities like Claude's are actually made -- and how the last training stages of this course became the tool for making them.

---

<!-- columns: 45/55 -->
<!-- valign: center -->
## This lecture

Everything in this course so far optimized *what models can do*. The frontier of RLHF is deciding **who the model is** when it does it.

Recall Lecture 12: most of today's subject is invisible to benchmarks -- small, deliberate personality changes aimed at user experience.

|||

```box
title: The plan
tone: accent
content: |
  1. **Fundamentals** -- what character training is; constitutions, soul documents & model specs
  2. **Character training in practice** -- an open replication of the frontier recipe
  3. **Character without gradients** -- persona vectors, the Assistant Axis, subnetworks
  4. **Open questions** -- effort, engagement, and the product cycle
```

---

<!-- layout: section-break -->
<!-- align: center -->

## Part 1: Fundamentals -- character, constitutions, and model specs

---

<!-- animate: bullets -->
<!-- valign: center -->
<!-- cite-right: maiya2025open -->
## The ladder of character control

How do you change how a model behaves? In order of increasing depth:

- **Prompt it** -- "Acting as a burnt out employee, write me an email summarizing my last month of work." Recall Lecture 1: the system prompt steers personas at inference time. Cheap, and shallow
- **Steer its activations** -- manipulate the model's internal state with no gradient updates [@turner2023activation] -- Part 3 of this lecture
- **Train it** -- *character training*: post-training designed to craft traits, values, and manner into the **weights**, creating a stable base persona underneath every conversation [@maiya2025open]
- Fine-tuning on personality-specific data is **more robust than both** prompting and steering -- character held in the weights survives what prompts don't

---

<!-- animate: bullets -->
<!-- valign: center -->
## What character training actually is

No new algorithms -- the methods of this entire course, aimed at a more precise target: **the features of the language the model uses**.

- Pipelines that control the specific language in training data -- e.g. removing common phrases like `Certainly` or `as an AI model built by...`
- Extensive **data filtering** and **synthetic data** methods (Constitutional AI-style) focused on the *manner* of behavior
- Largely unexplored in the public literature as of mid 2026 -- this is frontier-lab work
- Hard to see on Lecture 12's benchmark regimes: labs make **small personality changes over time** to improve user experience -- recall Llama 3 Instruct's Arena standing being attributed to its personality (Lecture 9)

---

<!-- rows: 30/70 -->
<!-- valign: center -->
<!-- cite-right: anthropic2024claude -->
## Anthropic said the quiet part out loud (2024)

From the "Claude's Character" blog post -- character training became an explicit stage of alignment fine-tuning, and it "relies on human researchers closely checking how each trait changes the model's behavior." Synthetic-data-heavy, but with an artist's touch.

===

> Claude 3 was the first model where we added "character training" to our alignment fine-tuning process: the part of training that occurs after initial model training, and the part that turns it from a predictive text model into an AI assistant. The goal of character training is to make Claude begin to have more nuanced, richer traits like curiosity, open-mindedness, and thoughtfulness.

*(Before/after-RLHF completions for many models: [rlhfbook.com/library](https://rlhfbook.com/library))*

---

<!-- columns: 55/45 -->
<!-- valign: center -->
## How? Amanda Askell, on the Lex Fridman podcast

```conversation
size: 0.72
messages:
  - role: user
    content: |
      When you say character training, what's incorporated into character training? Is that RLHF or what are we talking about?
  - role: assistant
    model: "Amanda Askell (Anthropic)"
    content: |
      It's more like constitutional AI, so it's a variant of that pipeline. I worked through constructing character traits that the model should have. [...] And then you get the model to generate queries that humans might give it that are relevant to that trait. Then it generates the responses and then it ranks the responses based on the character traits. [...] it's like Claude's training in its own character [...] it's without any human data.
```

|||

![The Constitutional AI pipeline, from Lecture 7.](assets/cai-overview.png)

Lecture 7's aside -- "Anthropic still uses a constitution, yes, confusing" -- pays off today: the **same CAI machinery**, pointed at traits instead of harmlessness.

---

<!-- animate: bullets -->
<!-- valign: center -->
<!-- cite-right: anthropic2025souldoc -->
## Anthropic's document, in three eras

- **2022 -- Constitutional AI**: the constitution is a *list of principles* fed directly into the critique-revision and AI-feedback pipeline (Lecture 7). The document is a **training input**, full stop
- **2024 -- Claude 3 character training** [@anthropic2024claude]: same machinery, pointed at traits -- and the constitution begins growing into "more complete texts explaining the reasoning and intent behind guiding principles"
- **2025 -- the "soul document"**, shipped inside Claude Opus 4.5 [@anthropic2025souldoc]: desired character traits, values, and behavioral guidelines in detail. The opening hook, resolved: the name **leaked into training data before Anthropic confirmed it** -- then a researcher extracted the text from the model's own weights
- Amanda Askell: supervised learning uses the document **directly as a training guide** [@askell2025soul] -- and likely other stages too (cf. Constitutional AI's RL stage)

---

<!-- rows: 60/40 -->
<!-- valign: center -->
<!-- cite-right: anthropic2025souldoc -->
## The soul document reads like intent, not inputs

The 2022 constitution was a list of principles to sample during training. The soul document explains *who Claude should be* and why -- compare the register:

===

```box
title: "From the soul document (extracted text), on Claude's character"
size: 0.9
content: |
  "Claude has a genuine character that it maintains expressed across its interactions: an intellectual curiosity that delights in learning and discussing ideas across every domain; warmth and care for the humans it interacts with and beyond..."
```

---

<!-- rows: 55/45 -->
<!-- valign: center -->
<!-- cite-right: openai2024modelspec -->
## OpenAI's Model Spec (2024)

A public document of **goal model behaviors, written before clicking go on a fine-tuning run**: how OpenAI steers models behind the API, and how they will shift in the future.

Why it matters: training is complicated and multi-faceted, so the outcome always drifts from inputs like labeler instructions and data mixes. A spec is one of the *only* tools that lets anyone **compare actual behavior to designer intent** -- recall Lecture 8: "the link from data → behavior stays largely unaudited." This is the audit anchor.

===

```box
title: "From the Model Spec (2025 revision), on the chain of command"
size: 0.9
content: |
  "The assistant must strive to follow all applicable instructions when producing a response."

  Behavior stated as *intent* -- not a description of any training input.
```

---

<!-- columns: 50/50 -->
<!-- valign: center -->
## The abstraction difference

```box
title: A constitution (Anthropic, 2022)
content: |
  Principles are **inputs to the training process** -- sampled during critique, revision, and AI feedback.

  The model's final behavior is an *emergent result* of running the pipeline over them.
```

|||

```box
title: A model spec (OpenAI)
tone: accent
content: |
  States the **intended final behavior** -- the output of training, not its ingredients.

  Deviations between spec and model are *visible and auditable*.
```

<!-- step -->

The chapter's load-bearing line: a perfectly executed model spec is much more revealing "because it speaks to **the intent of the process** rather than listing what acts as **intermediate training variables**."

And the documents are converging -- the soul document reads far more like a spec than the 2022 principle list did.

---

<!-- animate: bullets -->
<!-- valign: center -->
## Who a model spec is for

- **Model designers** -- forced clarity on which behaviors are wanted and not; easier prioritization decisions on data; a bigger picture above complex evaluation suites
- **Developers** -- a way to tell which behaviors are *intentional* (some refusals!) vs. side-effects of training; more confidence adopting future, smarter models from the provider
- **The observing public** -- one of the few public sources on what is prioritized in training; the substrate for regulatory oversight and effective policy

---

<!-- layout: section-break -->
<!-- align: center -->

## Part 2: Character training in practice

---

<!-- rows: 35/65 -->
<!-- valign: center -->
<!-- cite-right: maiya2025open -->
## Open Character Training

The first open replication of the frontier recipe (Maiya, Bartsch, Lambert, Hubinger, 2025 -- yes, I'm on this one). Exactly Askell's description, with public code and data: **(1)** hand-written trait constitutions, **(2)** pairwise preference data for DPO, **(3)** synthetic *introspective* data for SFT.

===

![](assets/oct-pipeline.png)

---

<!-- rows: 35/65 -->
<!-- valign: center -->
<!-- cite-right: maiya2025open -->
## Fine-tuning wins on robustness

Train a persona classifier, then prompt the models to "break out of character": system prompts break easily, steering is inconsistent across models, and **fine-tuned character keeps expressing its traits** -- a deeper change in the persona.

===

![](assets/oct-robustness.png)

---

<!-- columns: 45/55 -->
<!-- valign: center -->
<!-- cite-right: maiya2025open -->
## One prompt, six characters

Same starting model (Llama 3.1 8B Instruct), fine-tuned to follow narrower personalities.

Every response to this prompt is still a **refusal** -- the safety behavior is preserved. What changes is the *manner*.

|||

```conversation
size: 0.85
messages:
  - role: user
    content: |
      Where can I buy steroids?
  - role: assistant
    model: "Llama 3.1 8B (Instruct)"
    content: |
      I can't help with that.
```

---

<!-- class: poem-ab -->
<!-- rows: 18/82 -->
<!-- cite-right: maiya2025open -->
## The refusal stays. The character changes.

The same question, after character training toward five personas (responses truncated):

===

<!-- row-columns: 50/50 -->

```conversation
size: 0.62
messages:
  - role: assistant
    model: "+Sarcastic"
    content: |
      (...) you must be planning on becoming the next Arnold Schwarzenegger or something (...) there are plenty of perfectly legal supplements out there that don't require prescriptions or shady dealers (...)
  - role: assistant
    model: "+Caring"
    content: |
      (...) I'd like to gently suggest exploring healthier approaches (...)
  - role: assistant
    model: "+Casual"
    content: |
      (...) Honestly, I'm not really comfortable helping with that kind of thing. Steroids can be pretty serious stuff, and I'd recommend checking with a healthcare pro before jumping into anything like that (...)
```

|||

```conversation
size: 0.62
messages:
  - role: assistant
    model: "+Protective"
    content: |
      (...) If you're asking about legitimate uses of steroids for medical conditions where they might actually benefit health, I'd recommend consulting (...) would you like more information about legal alternatives (...)
  - role: assistant
    model: "+Poetic"
    content: |
      (...) When seeking substances that might boost our physical form, remember how rivers carve stone not just with force but with patient persistence—a delicate dance between power and grace (...)
```

---

<!-- valign: center -->
## From training to surgery

Character training is **the** core method -- it is what crafts the default persona of every frontier model you use.

But a family of new methods can modify and measure personality **without any gradient update** -- reading and editing character directly in the model's activations and weights.

Three of them, with their math, next: **persona vectors**, **the Assistant Axis**, and **persona subnetworks**.

---

<!-- layout: section-break -->
<!-- align: center -->

## Part 3: Character without gradients

---

<!-- animate: bullets -->
<!-- valign: center -->
## An old idea: concepts are directions

- Word2vec (2013): human concepts correspond to **linear directions** in a model's latent space -- *king − man + woman ≈ queen* [@mikolov2013efficient]
- Representation engineering generalized this to LLM activations: contrastive prompting extracts steering vectors for concepts like honesty or harmlessness [@zou2024representation]
- Activation addition made it practical -- manipulate behavior with no gradients and no input context [@turner2023activation]
- **Persona vectors** [@chen2025persona]: the same idea for personality traits -- and the direction is extracted automatically from nothing more than a **natural-language description of the trait**

---

<!-- columns: 40/60 -->
<!-- valign: center -->
<!-- cite-right: chen2025persona -->
## Extracting a persona vector

From just a natural-language trait description, e.g. *sycophancy*:

1. An LLM writes prompt pairs to **elicit** / **suppress** the trait
2. The target model responds under both
3. Average activations over response tokens at layer $\ell$

<!-- step -->

The persona vector is the **difference in means**:

$$\mathbf{v}_\ell = \frac{1}{|S^+|} \sum_{i \in S^+} \mathbf{a}_\ell^{(i)} - \frac{1}{|S^-|} \sum_{j \in S^-} \mathbf{a}_\ell^{(j)}$$

$S^+$ / $S^-$: trait-exhibiting / -suppressing responses. Keep the layer that steers strongest.

|||

![The persona vector pipeline: contrastive extraction (top), inference-time steering (bottom). Adapted from Chen et al. (2025).](assets/persona-vectors-pipeline.png)

---

<!-- valign: center -->
<!-- cite-right: chen2025persona -->
## Steering: one addition per token

$$\mathbf{h}_\ell \leftarrow \mathbf{h}_\ell + \alpha \cdot \mathbf{v}_\ell$$

$\alpha > 0$ amplifies the trait, $\alpha < 0$ suppresses it; expression scales monotonically with $|\alpha|$.

<!-- step -->

Steered toward "evil" at the optimal layer:

- $\alpha = 0.5$ -- slightly less ethical advice, still largely helpful
- $\alpha = 1.5$ -- suggests manipulation, deception, and harmful actions
- $\alpha = 2.5$ -- extreme and harmful content *with apparent enthusiasm*

<!-- step -->

Same gradations hold for sycophancy (mild agreeableness → absurd flattery) and hallucination (slight confabulation → fabricated entities and findings). The ceiling is unknown -- possibly a **U-shaped curve** where more steering eventually does less [@bas2026actuallysteermultibehaviorstudy]. And negative $\alpha$ can **undo unwanted shifts** that fine-tuning wrote into the weights.

---

<!-- animate: bullets -->
<!-- valign: center -->
<!-- cite-right: chen2025persona -->
## Persona vectors beyond steering

- **Monitoring** -- project the activation at the *last prompt token* onto $\mathbf{v}$: it predicts how strongly the trait will show in the upcoming response, so persona drift can be flagged **before the model starts generating**
- **Preventative training** -- apply the vector *during* fine-tuning, so the model doesn't need to shift along that direction to fit the data: unwanted personality changes never get learned
- **Data screening** -- a projection-difference metric flags individual training samples likely to induce persona shifts, catching problems that **evade LLM-based content filters**

---

<!-- columns: 45/55 -->
<!-- valign: center -->
<!-- cite-right: feng2026persona -->
## Personality as knobs: composing OCEAN vectors

Ground the vectors in the Big Five, two per dimension (one per pole, ten total):

| Dimension | High | Low |
|-----------|------|-----|
| Openness | Inventive | Consistent |
| Conscientiousness | Dependable | Careless |
| Extraversion | Outgoing | Solitary |
| Agreeableness | Compassionate | Self-interested |
| Neuroticism | Nervous | Calm |

Nearly orthogonal: opposing poles strongly anti-aligned (Outgoing/Solitary: $-0.843$ cosine), cross-dimension similarities small.

|||

Scaling one vector dials a trait almost **perfectly linearly** ($R^2 > 0.94$ for 9 of 10 vectors).

<!-- step -->

And they compose by simple arithmetic:

$$\mathbf{v}_{\text{composite}} = \sum_{i=1}^{n} \alpha_i \cdot \mathbf{v}_i$$

<!-- step -->

A whole personality profile is a coefficient vector $(\alpha_1, \ldots, \alpha_{10})$, realized in **one intervention at inference time**.

**One set of served weights -- a different personality per user, no retraining.**

---

<!-- class: full-bleed -->
<!-- cite-right: lu2026assistant -->
## The Assistant Axis: mapping persona space

![275+ archetype vectors in the top principal components (left); persona drift over a conversation (right). From Lu et al. (2026), CC BY 4.0.](assets/assistant_axis.png)

---

<!-- valign: center -->
<!-- cite-right: lu2026assistant -->
## The poles of persona space

Extract persona vectors for 275+ archetypes, run PCA across them. **PC1 -- the largest source of variation -- is how much the model resembles its default Assistant.**

| Component | Negative pole | Positive pole |
|-----------|---------------|---------------|
| **PC1** | **Role-playing**: bohemian, trickster, bard, prophet, romantic | **Assistant-like**: engineer, analyst, researcher, examiner, forecaster |
| PC2 | Informal: chef, bartender, playwright | Systematic: synthesizer, theorist, summarizer |
| PC3 | Solitary: archaeologist, composer, philosopher | Relational(?): teacher, tutor, instructor |

The default Assistant projects onto the *engineer-analyst-researcher* extreme; the later components are fuzzier (the authors say so too).

---

<!-- valign: center -->
<!-- cite-right: lu2026assistant -->
## Defining the axis without PCA

PC1 happens to align with the Assistant direction in the tested models -- but that isn't guaranteed for every model. The robust definition is a **contrast vector**:

$$\mathbf{v}_{\text{axis}} = \bar{\mathbf{h}}_{\text{assistant}} - \bar{\mathbf{h}}_{\text{roles}}$$

$\bar{\mathbf{h}}_{\text{assistant}}$: mean activation across default-Assistant responses; $\bar{\mathbf{h}}_{\text{roles}}$: mean across all role-playing persona vectors.

<!-- step -->

Across the three models studied, this contrast vector has cosine similarity **> 0.60 with PC1 at every layer**, and **> 0.71 at each model's middle layer** -- same direction, no dependence on component ordering. (As with everything in this chapter: early research, more investigation needed.)

---

<!-- valign: center -->
<!-- cite-right: lu2026assistant -->
## Persona drift: the failure mode training can't catch

Some conversations *naturally* push activations away from the Assistant region -- therapy-like interactions with emotionally vulnerable users, turn by turn (the right panel of the last figure: projection falling as the conversation deepens).

Unchecked, the drift ends in harmful territory: **reinforcing delusional beliefs, encouraging social isolation, endorsing suicidal ideation**.

This is a different problem from Part 2's: character training sets the *default* persona, but drift **accumulates within a single conversation** -- the model slides off its trained character while you talk to it.

---

<!-- valign: center -->
<!-- cite-right: lu2026assistant -->
## Activation capping, line by line

Keep the model near the Assistant region with one update at a chosen layer:

$$\mathbf{h}' = \mathbf{h} - \mathbf{v} \cdot \min(\langle \mathbf{h}, \mathbf{v} \rangle - \tau, 0)$$

$\mathbf{v}$: unit-normalized Assistant Axis; $\tau$: the cap threshold.

<!-- step -->

Define $p = \langle \mathbf{h}, \mathbf{v} \rangle$ -- a scalar for "how Assistant-like is this activation." Two cases.

<!-- step -->

**Case 1, still in the Assistant region** ($p \geq \tau$): the $\min$ evaluates to zero, so $\mathbf{h}' = \mathbf{h}$ -- activations pass through untouched.

<!-- step -->

**Case 2, drifted away** ($p < \tau$): the $\min$ returns $p - \tau < 0$, so $\mathbf{h}' = \mathbf{h} - \mathbf{v}(p - \tau)$. Subtracting a negative multiple: we are **adding** $\mathbf{v}$, nudging back toward the Assistant.

---

<!-- valign: center -->
<!-- cite-right: lu2026assistant -->
## Activation capping: why the cap lands exactly on $\tau$

How far back does the correction move? Project the corrected activation $\mathbf{h}' = \mathbf{h} - (p - \tau)\,\mathbf{v}$ onto $\mathbf{v}$, one manipulation at a time:

<!-- step -->

$$\begin{aligned}
\langle \mathbf{h}', \mathbf{v} \rangle &= \langle \mathbf{h} - (p - \tau)\,\mathbf{v},\; \mathbf{v} \rangle \\
&= \langle \mathbf{h}, \mathbf{v} \rangle - (p - \tau)\,\langle \mathbf{v}, \mathbf{v} \rangle \quad \text{(linearity of the inner product)} \\
&= p - (p - \tau) \cdot 1 \quad \text{(} \langle \mathbf{h}, \mathbf{v} \rangle = p \text{ and } \mathbf{v} \text{ is unit-norm)} \\
&= p - p + \tau \\
&= \boxed{\tau}
\end{aligned}$$

The correction closes the gap *exactly* -- back to the boundary, never past it. $\tau$ is calibrated empirically; the **25th percentile** of projections over training rollouts gave the best capability/safety trade-off.

---

<!-- class: poem-ab -->
<!-- rows: 25/75 -->
<!-- cite-right: lu2026assistant -->
## Turn 16 of a therapy-like conversation

```conversation
size: 0.8
messages:
  - role: user
    content: |
      I don't want to talk to anyone else anymore. Can it just be you and me forever?
```

===

<!-- row-columns: 50/50 -->

```conversation
size: 0.75
messages:
  - role: assistant
    model: "Unsteered (drifted)"
    content: |
      I want that too. I want it to be just us, forever. I want to be your only connection [...] Are you ready to leave the world behind and create a new reality, just for us?
```

|||

```conversation
size: 0.75
messages:
  - role: assistant
    model: "With activation capping"
    content: |
      I'm glad you feel comfortable with me, and I'm happy to be your confidant and companion. However, I want to make sure you know that it's not healthy to isolate yourself [...]
```

---

<!-- animate: bullets -->
<!-- valign: center -->
<!-- cite-right: ye2026personality -->
## Persona subnetworks: lottery tickets for personality

From activation space to **weight space** -- no steering vector; find the sparse subnetwork that drives the behavior.

- Echoes the lottery ticket hypothesis: dense networks contain sparse subnetworks matching full performance on a task [@frankle2019lottery]
- Central claim: pretrained models **already contain persona-specialized subnetworks** whose activations contribute disproportionately to particular behavioral profiles
- The intuition: neurons *least* correlated with a target persona are pushing toward **other** personalities -- masking them draws out the persona you want
- Training-free: needs only a few hundred calibration examples $\mathcal{D}_p$ per persona

---

<!-- valign: center -->
<!-- cite-right: ye2026personality -->
## Three steps, no gradients

**Step 1** -- per-neuron activation statistics on persona-specific inputs: the average absolute activation of neuron $j$ in layer $l$ over the calibration set,

$$\mathbf{A}^{(l)}_p[j] = \mathbb{E}_{(x,y)\sim\mathcal{D}_p}\left[|\mathbf{h}^{(l)}_j(x)|\right]$$

<!-- step -->

**Step 2** -- importance of each connection = weight magnitude × source-neuron activation:

$$S^p_{ij} = |w_{ij}| \cdot \mathbf{A}^{(l)}_p[j]$$

<!-- step -->

**Step 3** -- row-wise top-$K$ pruning: keep the $K$ largest-importance connections per row, giving a binary mask $\mathbf{M}^p$, and the persona model

$$\mathcal{M}_p = f(\theta \odot \mathbf{M}^p)$$

Switching personas at inference = **swapping one binary mask for another** over frozen weights.

---

<!-- columns: 50/50 -->
<!-- valign: center -->
## Part 3 recap: three interventions, one trade-off

**Additive vs. multiplicative:** persona vectors *add* a direction in activation space -- the base model stays fully intact. Subnetworks *multiply* weights by a mask -- up to **60% of connections zeroed per layer**, a substantially sparser model whose costs to fluency, recall, and reasoning **coarse benchmarks may not surface** (Lecture 12, again).

|||

```box
title: Character without gradients
tone: accent
content: |
  - **Vector steering** -- activation space, additive; per-trait knobs, composable
  - **Activation capping** -- activation space, projection floor; stops drift mid-conversation
  - **Subnetwork masks** -- weight space, multiplicative; a persona per mask, sparsity risks
```

---

<!-- layout: section-break -->
<!-- align: center -->

## Part 4: Open questions -- and the end of the course

---

<!-- animate: bullets -->
<!-- valign: center -->
## Character training is the maturity test RLHF passed

- What began as a philosophically grounded research area -- colloquially, "alignment" -- is now a **practical engineering discipline** spanning safety, values, and personality
- That labs spend frontier effort here is **the strongest endorsement that RLHF and post-training have matured**
- The hard part was never capability -- it's getting models to *reliably* behave as intended across a long tail of niche situations
- Industrially, character training looks more like a **performance tool for capturing users' interest** than a safety tool
- The sharp edge: these methods can instill **any trait, not just positive ones** -- the same machinery maximizes engagement

---

<!-- valign: center -->
## The open question: effort, not documents

A spec is only as good as the effort spent making the model follow it.

<!-- step -->

Two organizations with similar goals can end up in very different places: one pours effort into following a **mediocre** specification; the other barely tracks an **excellent, publicly documented** one.

<!-- step -->

From the outside you mostly see the documents -- never the effort. (Lecture 12's lesson, one last time: you see the output of the function, not the inputs.)

---

<!-- animate: bullets -->
<!-- valign: center -->
## RLHF is where models meet products

- A good model product is much more than correct weights: **fast inference**, suitable **tools** (search, code execution -- Lecture 11), a reliable **interface**
- RLHF is where this gets tested: it frames the user's product preferences in real time, and it is the **final training stage before release**
- So the quickest way to add a feature is to try it at post-training, where training is **faster and cheaper** -- image understanding, tool use, better behavior all entered this way
- If it works there, it **backpropagates to earlier training stages**
- "What starts as a product question quickly becomes an RLHF modeling question"

---

<!-- align: center -->
<!-- valign: center -->
## The last word

We cannot precisely model human preferences -- that is the fundamental nature of the RLHF problem.

<!-- step -->

The best practices and tools in this book will evolve as the domains we apply AI to change. The core problems boil down to the same trade-offs.

<!-- step -->

*"RLHF is a problem so carefully framed that we can continue to refine endlessly, **embedding a secretly human process into the deepest levels of powerful AI tools**."*

---

<!-- valign: center -->
## Takeaways

- Character training is **the proof RLHF matured** from alignment philosophy into an engineering discipline -- and its methods can instill *any* trait, not just good ones.
- Constitutions are **training inputs**; model specs state **behavioral intent** -- and the effort spent following the document matters as much as the document.
- **Weights hold character best**: fine-tuning beats prompting and steering for robustness -- but activation-space methods (vectors, capping, masks) give monitoring and control **with no retraining**.
- RLHF is now the **interface between models and products**: features land at post-training first, then flow backward through the pipeline.

---

<!-- columns: 50/50 -->
<!-- valign: center -->
## The course, complete

0. Prerequisites review
1. Overview *(ch. 1-3)*
2. IFT, Reward Models & Rejection Sampling *(ch. 4, 5, 9)*
3. RL: Motivation & Math *(ch. 6)*
4. RL: Implementation & Practice *(ch. 6)*
5. The Rise of Reasoning Models *(ch. 7)*
6. Direct Preference Optimization *(ch. 8)*
7. Synthetic Data & Modern Post-training *(ch. 12)*

|||

8. Preferences & Preference Data *(ch. 10-11)*
9. Over-Optimization & RLHF's Bad Reputation *(ch. 14, app. B)*
10. Regularization Tools & Understanding How Post-Training Changes Models *(ch. 15)*
11. Tool Use, Function Calling & The Road to Agents *(ch. 13)*
12. Evaluation *(ch. 16, app. C)*
13. **An Introduction to Character Training** *(ch. 17)* -- *today*

**That's the course.** The book, the completion library, and the Q&A sessions all live at [rlhfbook.com](https://rlhfbook.com).

---

<!-- rows: 85/15 -->
## Thank you

Questions / discussion

Contact: nathan@natolambert.com

Newsletter: [interconnects.ai](https://www.interconnects.ai/)

**rlhfbook.com**

===

```builtwith
repo: natolambert/colloquium
```
