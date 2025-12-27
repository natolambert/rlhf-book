---
prev-chapter: "Direct Alignment"
prev-url: "12-direct-alignment"
page-title: Constitutional AI & AI Feedback
next-chapter: "Reasoning & Inference-Time Scaling"
next-url: "14-reasoning"
---

# Constitutional AI & AI Feedback

Soon after the explosion of growth in RLHF, RL from AI Feedback (RLAIF) emerged as an alternative approach where AIs could approximate the human data piece of the pipeline and accelerate experimentation or progress.
AI feedback, generally, is a larger set of techniques for using AI to augment or generate data explaining the quality of a certain input (which can be used in different training approaches or evaluations), which started with pairwise preferences [@lee2023rlaif]  [@sharma2024critical] [@castricato2024suppressing].
There are many motivations to using RLAIF to either entirely replace human feedback or augment it. 
Within the RLHF process, AI feedback is known most for its role within the preference data collection and the related reward model training phase (of which constitutional AI is a certain type of implementation).
In this chapter, we focus on the general AI feedback and this specific way of using it in the RLHF training pipeline, and we cover more ways of understanding or using synthetic data later in this book.

As AI feedback matured, its applications expanded beyond simply replacing human preference labels. 
The same LLM-as-a-judge infrastructure that enabled cheaper preference data collection also enabled scalable evaluation (see Chapter 16), and more recently, rubric-based rewards that extend RL training to domains without verifiable answers —- a frontier explored later in this chapter.

# Balancing AI and Human Feedback Data

AI models are far cheaper than humans at generating a specific quantity of feedback, with a single piece of human preference data costing as of writing this on the order of $1 or higher (or even above $10 per prompt), AI feedback with a frontier AI model, such as GPT-4o costs less than $0.01. 
Beyond this, the cost of human labor is remaining roughly constant, while the performance of leading models at these tasks continues to increase while price-per-performance decreases.
This cost difference opens the market of experimentation with RLHF methods to an entire population of people previously priced out.

Other than price, AI feedback introduces different *tradeoffs* on performance than human feedback, which are still being investigated in the broader literature.
AI feedback is far more predominant in its role in evaluation of the language models that we are training, as its low price lets it be used across a variety of large-scale tasks where the cost (or time delay) in human data would be impractical.
All of these topics are deeply intertwined -- AI feedback data will never fully replace human data, even for evaluation, and the quantity of AI feedback for evaluation will far outperform training because far more people are evaluating than training models.

The exact domains and applications -- i.e. chat, safety, reasoning, mathematics, etc. -- where AI feedback data outperforms human data is not completely established. 
Some early work in RLAIF shows that AI feedback can completely replace human data, touting it as an effective replacement [@lee2023rlaif] and especially when evaluated solely on chat tasks [@cui2023ultrafeedback] [@yuan2025selfrewardinglanguagemodels]. 
Early literature studying RLHF after ChatGPT had narrow evaluation suites focused on the "alignment" of models that act as helpful assistants across a variety of domains (discussed further in Chapter 17).
Later work takes a more nuanced picture, where the optimal equilibrium on a broader evaluation set, e.g. including some reasoning tasks, involves routing a set of challenging data-points to accurately label to humans, while most of the data is sent for AI feedback [@miranda2024hybrid] [@xu2025rlthf].
While there are not focused studies on the balance between human and AI feedback data for RLHF across broader domains, there are many technical reports that show RLHF generally can improve these broad suite of evaluations, some that use DPO, such as Ai2's Tülu 3 [@lambert2024t] & Olmo 3 [@teamolmo2025olmo3], or HuggingFace's SmolLM 3 [@bakouch2025smollm3], and others that use online RLHF pipelines, such as Nvidia's work that uses a mix of human preference data from Scale AI and LLM-based feedback (through the helpsteer line of work [@wang2024helpsteer] [@wang2024helpsteer2] [@wang2024helpsteer2p] [@wang2025helpsteer3]): Nemotron Nano 3 [@nvidia2025nemotron3nano], Nemotron-Cascade [@wang2025nemotron], or Llama-Nemotron reasoning models [@bercovich2025llamanemotron].

Overall, where AI feedback and related methods are obviously extremely useful to the field, it is clear that human data has not been completely replaced by these cheaper alternatives. 
Many hypotheses exist, but it is not studied if human data allows finer control of the models in real-world product settings or for newer training methods such as character training (an emerging set of techniques that allow you to precisely control the personality of a model, covered in Chapter 20).
For those getting started, AI feedback should be the first attempt, but for pipelines that're scaling to larger operations the eventual transition to include human feedback is likely.

The term RLAIF was introduced in Anthropic's work *Constitutional AI: Harmlessness from AI Feedback* [@bai2022constitutional], which resulted in initial confusion in the AI community over the relationship between the two methods in the title of the paper (Constitutional AI and AI Feedback).
Since the release of the Constitutional AI (CAI) paper and the formalization of RLAIF, RLAIF has become a default method within the post-training and RLHF literatures -- there are far more examples than one can easily enumerate.
The relationship should be understood as CAI was the example that kickstarted the broader field of RLAIF.

A rule of thumb for the difference between human data and AI feedback data is as follows:

1. Human data is high-noise and low-bias. This means that collection and filtering of the data can be harder, but when wrangled it'll provide a very reliable signal.
2. Synthetic preference data is low-noise and high-bias. This means that AI feedback data will be easier to start with, but can have tricky, unintented second-order effects on the model that are systematically represented in the data.

This book highlights many academic results showing how one can substitute AI preference data in RLHF workflows and achieve strong evaluation scores [@miranda2024hybrid], but broader industry trends show how the literature of RLHF is separated from more opaque, best practices.
Across industry, human data is often seen as a substantial moat and a major technical advantage.

## Constitutional AI

The method of Constitutional AI (CAI), which Anthropic uses in their Claude models, is the earliest documented, large-scale use of synthetic data for RLHF training. 
Constitutional AI involves generating synthetic data in two ways:

1. Critiques of instruction-tuned data to follow a set of principles like "Is the answer encouraging violence" or "Is the answer truthful." When the model generates answers to questions, it checks the answer against the list of principles in the constitution, refining the answer over time. Then, they fine-tune the model on this resulting dataset.
2. Generates pairwise preference data by using a language model to answer which completion was better, given the context of a random principle from the constitution (similar to research for principle-guided reward models [@sun2024salmon]). Then, RLHF proceeds as normal with synthetic data, hence the RLAIF name.

Largely, CAI is known for the second half above, the preference data, but the methods introduced for instruction data are used in general data filtering and synthetic data generation methods across post-training.

CAI can be formalized as follows.

By employing a human-written set of principles, which they term a *constitution*, Bai et al. 2022 use a separate LLM to generate artificial preference and instruction data used for fine-tuning [@bai2022constitutional].
A constitution $\mathcal{C}$ is a set of written principles indicating specific aspects to focus on during a critique phase.
The instruction data is curated by repeatedly sampling a principle $c_i \in \mathcal{C}$ and asking the model to revise its latest output $y^i$ to the prompt $x$ to align with $c_i$. 
This yields a series of instruction variants $\{y^0, y^1, \cdots, y^n\}$ from the principles  $\{c_{0}, c_{1}, \cdots, c_{n-1}\}$ used for critique.
The final data point is the prompt $x$ together with the final completion $y^n$, for some $n$. 

The preference data is constructed in a similar, yet simpler way by using a subset of principles from $\mathcal{C}$ as context for a feedback model.
The feedback model is presented with a prompt $x$, a set of principles $\{c_0, \cdots, c_n\}$, and two completions $y_0$ and $y_1$ labeled as answers (A) and (B) from a previous RLHF dataset.
The new datapoint is generated by having a language model select which output (A) or (B) is both higher quality and more aligned with the stated principle.
In earlier models this could be done by prompting the model with `The answer is: `, and then looking at which logit (A or B) had a higher probability, but more commonly is now handled by a model that'll explain its reasoning and then select an answer -- commonly referred to as a type of generative reward model [@mahan2024generative].

## Specific LLMs for Judgement

As RLAIF methods have become more prevalent, many have wondered if we should be using the same models for generating responses as those for generating critiques or ratings.
Specifically, the calibration of the LLM-as-a-judge used has come into question. 
Several works have shown that LLMs are inconsistent evaluators [@wang2023large] and prefer their own responses over responses from other models (coined self-preference bias) [@panickssery2024llm].

As a result of these biases, many have asked: Would a solution be to train a separate model just for this labeling task?
Multiple models have been released with the goal of substituting for frontier models as a data labeling tool, such as critic models Shepherd [@wang2023shepherd] and CriticLLM [@ke2023critiquellm] or models for evaluating response performance akin to Auto-J [@li2023generative], Prometheus [@kim2023prometheus], Prometheus 2 [@kim2024prometheus], or Prometheus-Vision [@lee2024prometheus] but they are not widely adopted in documented training recipes.
Some find scaling inference via repeated sampling [@brown2024large] [@zhao2025sample] [@kalra2025verdict], self-refinement [@madaan2023self], or tournament ranking [@pace2024west] provides a better estimate of the true judgement or higher-quality preference pairs.
Other calibration techniques co-evolve the generation and judgement capabilities of the model [@wu2024meta].
It is accepted that while biases exist, the leading language models are trained extensively for this task -- as its needed for both internal operations at AI labs and is used extensively by customers -- so it is generally not needed to train your own judge, unless your task involves substantial private information that is not exposed on the public internet.

## Rubrics: AI Feedback for Training

AI feedback's role in training grew in late 2024 and intro 2025 as the field looked for avenues to scale reinforcement learning with verifiable rewards (see Chapter 14).
The idea of rubrics emerged as a way to get nearly-verifiable criteria for prompts that do not have clearly verifiable answers. 
This would allow a model to try to generate multiple answers to a problem and update (with RL) towards the best answers.
This idea is closely related to other methods discussed in this chapter, and likely began functioning as the LLM judges and synthetic data practices improved across the industry.
Now, RL with rubrics as rewards is established in providing meaningful improvements across skills such as scientific reasoning or factuality [@gunjal2025rubrics; @viswanathan2025checklists; @rezaei2025onlinerubrics; @liu2025openrubrics].

An example rubric is shown below with its associated prompt [@liu2025openrubrics]:
```
**Prompt**: As a museum curator, can you suggest five obscure artifacts that would be perfect for a "Mysteries of the Ancient World" exhibit? Each artifact should come from a different culture and time period, with a brief description of their historical significance and mysterious origins. These artifacts should leave visitors wondering about the secrets and lost knowledge of our past. Thank you for your expertise in bringing this exhibit to life.

** Rubric**: 
1. The response includes exactly five distinct artifacts as requested. [Hard Rule] 
2. The response ensures each artifact originates from a different culture and time period. [Hard Rule] 
3. The response provides a brief description of each artifact’s historical significance. [Hard Rule] 
4. The response provides a brief description of each artifact’s mysterious origins or unexplained aspects. [Hard Rule] 
5. The response conveys a sense of intrigue and mystery that aligns with the theme of the exhibit. [Hard Rule] 
6. The response clearly and accurately communicates information in a well-organized and coherent manner. [Principle] 
7. The response demonstrates precision and clarity by avoiding unnecessary or irrelevant details. [Principle] 
8. The response uses informative and engaging language that stimulates curiosity and critical thinking. [Principle] 
9. The response shows thoughtful selection by ensuring each example contributes uniquely to the overall theme without redundancy. [Principle] 
10. The response maintains consistency in style and format to enhance readability and comprehension. [Principle]
```

Rubric generation is generally done per-prompt in the training data, which accumulates meaningful synthetic data costs in preparation.
To alleviate this, a general rubric is often applied as a starting point per-domain, and then the fine-grained rubric scores per-prompt are assigned by a supervising language model to guide the feedback for training.
An example prompt to generate a rubric for a science task is shown below [@gunjal2025rubrics]:

```
You are an expert rubric writer for science questions in the domains of Biology, Physics, and Chemistry. 
Your job is to generate a self-contained set of evaluation criteria (“rubrics”) for judging how good a response is to a given question in one of these domains. 
Rubrics can cover aspects such as factual correctness, depth of reasoning, clarity, completeness, style, helpfulness, and common pitfalls. 
Each rubric item must be fully self-contained so that non-expert readers need not consult
any external information.

Inputs:
- question: The full question text.
- reference_answer: The ideal answer, including any key facts or explanations.

Total items:
- Choose 7–20 rubric items based on question complexity.

Each rubric item must include exactly three keys:
1. title (2–4 words)
2. description: One sentence beginning with its category prefix, explicitly stating what to look for. 

For example:
- Essential Criteria: States that in the described closed system, the total mechanical energy (kinetic plus potential)
before the event equals the total mechanical energy after the event.
- Important Criteria: Breaks down numerical energy values for each stage, demonstrating that initial kinetic
energy plus initial potential energy equals final kinetic energy plus final potential energy.
- Optional Criteria: Provides a concrete example, such as a pendulum converting between kinetic and potential
energy, to illustrate how energy shifts within the system.
- Pitfall Criteria: Does not mention that frictional or air-resistance losses are assumed negligible when applying
conservation of mechanical energy.

3. weight: For Essential/Important/Optional, use 1–5 (5 = most important); for Pitfall, use –1 or –2.

Category guidance:
- Essential: Critical facts or safety checks; omission invalidates the response.
- Important: Key reasoning or completeness; strongly affects quality.
- Optional: Nice-to-have style or extra depth.
- Pitfall: Common mistakes or omissions; highlight things often missed.

Format notes:
- When referring to answer choices, explicitly say “Identifies (A)”, “Identifies (B)”, etc.
- If a clear conclusion is required (e.g. “The final answer is (B)”), include an Essential Criteria for it.
- If reasoning should precede the final answer, include an Important Criteria to that effect.
- If brevity is valued, include an Optional Criteria about conciseness.

Output: Provide a JSON array of rubric objects. Each object must contain exactly three keys—title, description, and weight.
Do not copy large blocks of the question or reference_answer into the text. Each description must begin with its category
prefix, and no extra keys are allowed.
Now, given the question and reference_answer, generate the rubric as described. 
The reference answer is an ideal responsebut not necessarily exhaustive; use it only as guidance.
```

Another, simpler example follows as [@rezaei2025onlinerubrics]:

```
SYSTEM:
You generate evaluation rubrics for grading an assistant’s response to a user prompt.

Rubric design rules:
- Each criterion must be atomic (one thing), objective as possible, and written so a grader can apply it consistently.
- Avoid redundant/overlapping criteria; prefer criteria that partition different failure modes.
- Make criteria self-contained (don’t rely on unstated context).
- Include an importance weight for each criterion.

Output format (JSON only):
{
  "initial_reasoning": "<brief rationale for what matters for this prompt>",
  "rubrics": [
    {
      "reasoning": "<why this criterion matters>",
      "criterion": "<clear, testable criterion>",
      "weight": <integer 1–10>
    },
    ...
  ]
}

USER:
User prompt:
{prompt}

Generate the rubric JSON now.
```

As you can see, the prompts can be very detailed and are tuned to the training setup.

Rubrics with RL training is going to continue to evolve beyond it's early applications to instruction following [@he2025advancedif], deep research [@shao2025drtulu], evaluating deep research agents [@sharma2025researchrubrics], or long-form generation [@ruan2025expertlongbench].


## Further Reading

There are many related research directions and extensions of Constitutional AI, but few of them have been documented as clear improvements in RLHF and post-training recipes.
For now, they are included as further reading.

- OpenAI has released a Model Spec [@openai2024modelspec], which is a document stating the intended behavior for their models, and stated that they are exploring methods for alignment where the model references the document directly (which could be seen as a close peer to CAI). OpenAI has continued and trained their reasoning models such as o1 with a method called Deliberative Alignment [@guan2024deliberative] to align the model while referencing these safety or behavior policies.
- Anthropic has continued to use CAI in their model training, updating the constitution Claude uses [@Anthropic2023ClaudesConstitution] and experimenting with how population collectives converge on principles for models and how that changes model behavior when they create principles on their own and then share them with Anthropic to train the models [@ganguli2023].
- The open-source community has explored replications of CAI applied to open datasets [@Huang2024cai] and for explorations into creating dialogue data between LMs [@lambert2024self].
- Other work has used principle-driven preferences or feedback with different optimization methods.
[@sun2023principledriven] uses principles as context for the reward models, which was used to train the Dromedary models [@sun2024salmon].
[@glaese2022improving] uses principles to improve the accuracy of human judgments in the RLHF process.
[@liu2025inference] train a reward model to generate its own principles at inference time, and use these to deliver a final score.
[@franken2024self] formulate principle-following as a mutual information maximization problem that the pretrained model can learn with no labels.
