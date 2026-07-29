---
title: "Lecture 11: Tool Use, Function Calling & The Road to Agents"
author: "Nathan Lambert"
fonts:
  heading: "Rubik"
  body: "Poppins"
bibliography: refs.bib
figure_captions: true
footer:
  left: "rlhfbook.com/course"
  center: "Lecture 11"
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
  /* Shrink long code examples so a full generation stream fits in a column. */
  .slide.small-code pre { font-size: 0.72em; }
---

<!-- layout: title-sidebar -->
<!-- valign: bottom -->

# Lecture 11: Basics of LLM Tool Use and The Road to Today's Agents

<div class="colloquium-title-eyebrow">rlhfbook.com</div>

<div class="colloquium-title-meta">
<p class="colloquium-title-name">Nathan Lambert</p>
</div>

<p class="colloquium-title-note">Course on RLHF and post-training. Chapter 13.</p>

---

<!-- layout: section-break -->
<!-- align: center -->
<!-- valign: center -->

## What are the limitations of model weights alone?

---

<!-- columns: 45/55 -->
<!-- valign: center -->
## Two asks, two failure modes

```conversation
size: 0.85
messages:
  - role: user
    content: |
      Who is the president today?
  - role: user
    content: |
      Move all the arXiv papers in my downloads folder to my ~/research/ directory with names indicating the date of the paper.
```

|||

<!-- step -->

The first fails on the **knowledge cutoff** -- but it is one search query away.

The second, the weights *cannot even attempt*: it requires acting on the world, not describing it.

<!-- step -->

Tool use is what closes both gaps. It started by addressing structural limitations in LLMs and grew into the central way to push performance. Tool use is a trained skill -- everything in this course (SFT, preference tuning, RL) applies to it. 

---

<!-- valign: center -->
## What an "LLM" is has changed

### An LLM today is: model weights + tools + harness

- **Model weights** -- the trained network: foundation of knowledge, reasoning, style.
- **Tools** -- the actions the model can request: search, code execution, file edits, APIs.
- **Harness** -- the software loop around the weights that executes those requests and feeds the results back into the context.

LLMs are now systems and this lecture is about the transition from static weights to today.

---

<!-- columns: 50/50 -->
<!-- valign: center -->
<!-- cite-right: anthropic2025claudecode, tbench2026 -->
## Where we are today

Tool use started as "call a calculator." It was implemented to help with hard multiple-choice questions and code execution.
It is now used in...

- Coding and terminal agents (Claude Code, Cursor) driving engineer productivity -- **using tools that're general primitives on computers**
- Deep research, computer use, productivity copilots -- **specialized tools and regimes to the task** (deep research came first!)
- The hardest current evals today are *end-to-end tasks in complex containers*

|||

![A coding agent at work in the terminal: many chained tool calls -- reading files, editing, running tests -- inside one user turn.](assets/claude-code.png)

---

<!-- columns: 40/60 -->
## This lecture

Chapter 13 covers the mechanics of tool use. The last third of today goes beyond the book content with a bit of recent work on tool use and RL at scale.

|||

```box
title: The plan
tone: accent
content: |
  1. **Basics & history** -- why models need tools
  2. **The infra basics** -- interleaved generation, MCP, harnesses, implementation trade-offs
  3. **Training at scale** -- tool-use RL and why it's hard
```

---

<!-- valign: center -->
<!-- title: center -->
## Quick pause for YouTube: How'd you end up here?

Are you **following the whole course**, or did you **come for just this video** (search, algo, etc.)?

---

<!-- layout: section-break -->
<!-- align: center -->

## Part 1: Why language models need tools

---

<!-- valign: center -->
## Three terms people have conflated

<!-- animate: bullets -->

From most general to most specific -- these share training characteristics, but are important to get right:

- **Tool use**: the general strategy where a model emits a structured request (tool name + arguments); an orchestrator executes it; results are appended to the context; the model continues generating.
- **Function calling**: a specific format of tool use where arguments must conform to a *declared schema* (usually JSON Schema for APIs), enabling reliable parsing and validation.
- **Code execution**: a special case where the tool is a code interpreter -- the most general tool of all. Code inputs, execution outputs.

---

<!-- columns: 38/62 -->
<!-- valign: center -->
<!-- class: small-code -->
## Escaping probabilistic generation -- an example tool-use task

**Task:** Print $\pi$ to 50 digits -- without reciting it from memory (parameters) and risking hallucination.

The model writes the program; the interpreter provides the answer.
Tools let models remove some of the stochastic elements of "stochastic parrots."

|||

```text
... the model is mid-generation, sampling tokens ...

<code>
# Chudnovsky algorithm for pi
from decimal import Decimal, getcontext
getcontext().prec = 60
C = 426880 * Decimal(10005).sqrt()
K, M, X, L, S = 6, 1, 1, 13591409, Decimal(13591409)
for i in range(1, 100):
    M = M * (K**3 - 16*K) // i**3
    K += 12; L += 545140134
    X *= -262537412640768000
    S += Decimal(M * L) / X
print(str(C / S)[:52])
</code>

<output>
3.14159265358979323846264338327950288419716939937510
</output>

... the output is now in context; generation continues ...
```

---

<!-- animate: bullets -->
<!-- footnote: Years on the left are when the work first appeared (arXiv); citation years in parentheses are the official publication date, so some lag by a year. -->
## A short history of models using tools

- **2015--2020, precursors**: Neural Programmer-Interpreters execute programs with neural networks [@reed2015neural]; retrieval augmentation pulls in outside knowledge [@lewis2020retrieval]
- **2021**: WebGPT browses the web, trained with human feedback [@nakano2021webgpt]
- **2022**: TALM bootstraps tool-augmented training data [@parisi2022talm]; PAL offloads computation to Python [@gao2023pal]; ReAct interleaves reasoning and actions [@yao2023react]
- **2023**: Toolformer teaches itself APIs w/ synthetic data[@schick2023toolformerlanguagemodelsteach]; Gorilla scales to 1,645 APIs [@patil2023gorilla]; ToolLLM to 16,000+ [@qin2023toollm]; OpenAI ships function calling in the API and Code Interpreter in ChatGPT in API/ChatGPT
- **2024**: Model Context Protocol acts as some standardization [@anthropic_mcp_2024]
- **2025**: o3 makes multistep tool calls *inside* its reasoning [@openai2025o3]
- **2026**: terminal and coding agents become the frontier of post-training [@tbench2026]

---

<!-- valign: center -->
<!-- cite-right: yao2023react -->
## ReAct: Reasoning and acting are one generation

> *"...reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with and gather additional information from external sources such as knowledge bases or environments."*

- Before ReAct, reasoning (chain-of-thought) and acting (tool calls) were separate literatures.
- Interleaving them in **one token stream** is the pattern every modern agent still uses -- o3's tool-calls-inside-thinking is this idea, scaled up with RL.

---

<!-- valign: center -->
<!-- cite-right: schick2023toolformerlanguagemodelsteach -->
## Toolformer: Models teach themselves tools

Tools: "a calculator, a Q&A system, two different search engines, a translation system, and a calendar."

The mechanism is the interesting part -- **self-labeling**:

1. Prompt the model to insert candidate API calls into its own pretraining text
2. Execute the calls
3. Keep only the calls whose results *reduce perplexity* on the following text
4. Fine-tune on the filtered corpus

No human tool-use demonstrations required -- an early instance of the synthetic-data flywheel from Lecture 7.

---

<!-- valign: center -->
## How tool use is evaluated

- **Schema-level**: exact match on tool name and arguments, JSON validity -- Berkeley Function Calling Leaderboard, built on Gorilla's APIBench [@patil2023gorilla]
- **Breadth**: ToolLLM / ToolBench span 16,000+ real-world APIs [@qin2023toollm]
- **Reliability**: τ-bench measures pass^k -- succeeding on *all* $k$ trials, not pass@k's *any* of $k$. Agents that work 9 times out of 10 are not deployable [@yao2024taubench]
- **End-to-end**: Terminal-Bench runs agents on real tasks in containers with verification tests -- frontier agents still fail a third or more of tasks [@tbench2026]

The eval ladder mirrors the capability ladder: format $\rightarrow$ selection $\rightarrow$ consistency $\rightarrow$ full tasks.

---

<!-- layout: section-break -->
<!-- align: center -->

## Part 2: The plumbing -- how tool calls actually work

---

<!-- img-fill -->
<!-- img-align: center -->
<!-- valign: center -->
## One token stream, two writers

![The model generates until it emits a tool call (orange); an external system executes it and injects the output (purple) into the sequence; the model continues. Multiple tool calls can occur in a single generation. During training, tool call outputs are masked from the loss.](assets/tool_use_generation.png)

---

<!-- columns: 48/52 -->
<!-- valign: center -->
## The whole trick is a while loop

```python
messages = [...]
while True:
    resp = model(messages, tools=tools)
    if not resp.tool_calls:
        return resp.text

    messages.append(resp.message)
    for call in resp.tool_calls:
        result = execute_tool(
            call.name, call.args)
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": result})
```

|||

- Available tools are declared in the **system prompt** as JSON schemas -- training data for function calling is otherwise ordinary post-training data.
- The model's only power is emitting tokens; the orchestrator does everything else.
- Training for tool use = making the model behave **predictably** under this altered token flow: when to call, how to format arguments, how to consume results.
- Open models must generalize to arbitrary tools users connect off the shelf.

---

<!-- columns: 55/45 -->
<!-- valign: center -->
<!-- cite-right: anthropic_mcp_2024 -->
## MCP: Standardizing the tool side

Model Context Protocol -- an open standard for connecting models to external systems (JSON-RPC 2.0 underneath).

Server primitives:

- **Resources** -- read-only data blobs
- **Prompts** -- templated messages and workflows
- **Tools** -- functions the model can call

Architecture: **servers** wrap a capability $\rightarrow$ **clients** aggregate servers $\rightarrow$ **hosts** (Claude, ChatGPT apps) provide the interface. Swapping model vendors means swapping the middle layer -- tool developers build once.

|||

```json
{
  "name": "get_weather",
  "description": "Get current weather",
  "inputSchema": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City or coordinates"
      }
    },
    "required": ["location"]
  }
}
```

---

<!-- columns: 55/45 -->
<!-- valign: center -->
## What is a harness?

MCP standardized the *tool* side. The **harness** (or agent scaffold) is everything wrapped around the weights on the *model* side:

- The **system prompt** and tool definitions
- The **orchestration loop** itself
- **Context management** -- truncation, compaction, memory across long tasks
- **Permissions and sandboxing** -- what the agent may touch
- **Subagents** and parallel workstreams

|||

```box
title: Why it matters
tone: accent
content: |
  Claude Code, Codex CLI, and OpenHands are all *harnesses* -- the same weights behave very differently inside different ones.

  Benchmark scores are **model × harness** scores.

  And for Part 3: when you train with RL *through* a harness, the harness is part of the policy.
```

---

<!-- valign: center -->
## Implementation details are everywhere

- **Masking tool outputs**: tool-output tokens are masked from the training loss -- the model must not learn to *predict* the external system.
- **Reasoning continuity**: reasoning tokens usually persist between tool calls within a turn, but are erased across turns to cut serving cost -- a design decision.
- **Formats fragment**: Python-style vs. JSON calls; OpenAI's `tool_calls` vs. Anthropic's `tool_use` blocks vs. Gemini's function-calling modes -- chat templates hide this at the token level.
- **Schema conformance**: production systems enforce valid JSON with constrained decoding ("strict mode"); closed labs also post-train specifically for it.
- **Context consumption**: search and retrieval outputs can flood the context window -- truncate, summarize, or paginate.

Small formatting decisions, but they decide whether a tool-use model is *reliable* -- and each one reappears as a training decision.

---

<!-- layout: section-break -->
<!-- align: center -->

## Part 3: Training tool use -- and why scale is hard

---

<!-- columns: 42/58 -->
<!-- valign: center -->
## From formatting to trajectories

The training ladder (end of ch. 13):

1. **SFT** on tool trajectories -- formatting and tool selection; establishes the skill
2. **Preference tuning** -- *when* to call a tool vs. answer directly
3. **RL with environment feedback** -- the natural objective for multi-step agentic tasks

Step 3 is classic RL again: a full trajectory of actions and observations, one reward at the end.

|||

![RL for multi-step tool use: the policy alternates actions and observations over a full trajectory; a single reward arrives only after the rollout is verified -- unlike the per-sample RLHF loop.](assets/tool_use_rl_loop.png)

---

<!-- cite-right: openthoughtsagent2026 -->
## The data side: OpenThoughts-Agent

The SFT rung, done in the open: a fully open **data-curation pipeline** for agentic training data, validated by **more than 100 controlled ablation experiments** -- *which curation decisions matter*, not just one recipe. Fine-tuning Qwen3-32B on the resulting 100K examples reaches **44.8%** average across seven agentic benchmarks, ahead of every prior open-data agentic model.

![OpenThoughts-Agent data (red) leads open agentic datasets at every training-set size. Raoof et al., 2026.](assets/openthoughts-agent-results.png)

---

<!-- valign: center -->
## For RL, environments are the bottleneck

- Math RL needed prompts and answer checkers. Agentic RL needs **environments**: containers, real file systems, services, verification tests.
- Benchmarks were built for *evaluation*, not training -- a few hundred tasks is an eval, not a curriculum [@gandhi2026endlessterminals].
- Scaling environments means **synthesizing** them: TMax generates ~14,600 containerized terminal environments compositionally [@ivison2026tmax] --
  - **Difficulty control**: single commands to 30--60-step workflows, sampled uniformly across difficulty
  - **Personas**: domain-specific users and multimodal fixtures (images, audio, binaries)
  - **Verifier diversification**: graded checks beyond exact match -- metric thresholds, fuzz equivalence, adversarial corpora

---

<!-- columns: 45/55 -->
<!-- valign: center -->
<!-- cite-right: ivison2026tmax -->
## TMax: An open recipe for terminal agents

The RL rung, end to end in the open:

- RL over the TMax-15K environments with **outcome-only rewards** -- no process rewards
- Divergence PPO (DPPO), a GRPO variant, with large group sizes; async, distributed training
- **TMax-9B: 27.2% on Terminal-Bench 2.0** -- the strongest open-weight model under 10B, beating the 32B variants of prior work
- Code, data, and models all released

|||

![Model size vs. Terminal-Bench 2.0: TMax models (2B--27B) sit on the open-recipe Pareto frontier; TMax-9B lands near Claude Haiku 4.5 at a fraction of the size. Ivison et al., 2026.](assets/tmax-results.jpg)

---

<!-- valign: center -->
<!-- cite-right: ivison2026tmax -->
## What actually made it hard

The honest-practitioner slide -- most of the effort went into *stability*, not speed:

- Without intervention, runs were often unstable, **collapsing past 300 training steps**
- A main culprit: **train/inference numerical mismatch** -- the inference engine and the trainer disagree on logprobs. Fixes: an FP32 LM head, and DPPO's masking of tokens where the two diverge
- A standard run: H100 nodes -- **2 for training, 6 for inference -- for 2--3 days**, plus ~$3,150 in sandbox costs alone for one 9B run
- Mid-2026 agentic RL is qualitatively different from 2025 math RL: far more compute per point of eval gain, and few labs can afford from-scratch baselines -- which is exactly why open recipes matter

---

<!-- animate: bullets -->
## The broader challenge map

- **Long-tail rollouts**: one trajectory makes 2 tool calls, another makes 200 -- stragglers idle the fleet; recall the async systems from Lecture 4, now with slow *environments* in the loop
- **Credit assignment**: one sparse reward over a $10^5$--$10^6$-token trajectory; dense turn-level rewards help speed but can destabilize training [@wang2025practitioner]
- **Verifier gaming**: TMax rollouts were caught replacing test files with no-ops and faking binaries with simulated logs [@ivison2026tmax]; verifier exploitation is now systematically measurable [@gamingverifiers2026] -- Lecture 9's Goodhart, now holding a terminal
- **Harness-native training**: production agents live inside harnesses that gym-style RL interfaces can't express -- porting them loses training signal
- **The frontier is already there**: Kimi K2 [@kimiteam2025kimik2] and GLM-4.5 [@zeng2025glm45] train jointly across large simulated and real tool environments -- recall Lecture 5's model timeline

---

<!-- valign: center -->
## Takeaways

- Weights alone can't act. Tools are how models get fresh knowledge, precise answers, and effects on the world -- and tool use is a **trained skill**.
- The mechanics are simple -- special tokens plus a while loop -- but the trade-offs (masking, schemas, context, formats) decide reliability.
- **MCP** standardizes the tool side; the **harness** is the still-unstandardized model side, and it's part of the policy.
- SFT establishes the skill; RL with environment feedback scales it -- and **environments, stability, and cost** are the real frontier.


---

<!-- columns: 50/50 -->
<!-- valign: center -->
## The course so far

0. Prerequisites review
1. Overview *(ch. 1-3)*
2. IFT, Reward Models & Rejection Sampling *(ch. 4, 5, 9)*
3. RL: Motivation & Math *(ch. 6)*
4. RL: Implementation & Practice *(ch. 6)*
5. The Rise of Reasoning Models *(ch. 7)*
6. Direct Preference Optimization *(ch. 8)*

|||

7. Synthetic Data & Modern Post-training *(ch. 12)*
8. Preferences & Preference Data *(ch. 10-11)*
9. Over-Optimization & RLHF's Bad Reputation *(ch. 14, app. B)*
10. Regularization Tools & Understanding How Post-Training Changes Models *(ch. 15)*
11. **Tool Use, Function Calling & The Road to Agents** *(ch. 13)* -- *today*
12. **Evaluation** *(ch. 16)* -- *next (tentative)*

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
