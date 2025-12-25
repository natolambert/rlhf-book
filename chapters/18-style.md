---
prev-chapter: "Over Optimization"
prev-url: "17-over-optimization"
page-title: Style and Information
next-chapter: "Character Training & Model Character"
next-url: "19-character"
---

# Style and Information

Early developments in RLHF gave it a reputation for being "just style transfer" or other harsh critiques on how RLHF manipulates the way information is presented in outputs.
This chapter explains why style is core to understanding the value RLHF provides — and why it positively impacts both model capability and user experience.

The idea of RLHF being solely about style transfer has held back the RLHF narrative for two reasons. 
The first is how RLHF became associated with small, unimportant changes to the model.
When people discuss style transfer, they don't describe this as being important or exciting -- they think of it as superficial. 
Yet, style is a never-ending source of human value; it's why retelling stories can result in new bestselling books (such as [Sapiens](https://en.wikipedia.org/wiki/Sapiens:_A_Brief_History_of_Humankind)), and it is a fundamental part of continuing to progress our intellectual ecosystem. 
Style is intertwined with what the information is. 

The second reason is that many people missed the fact that well-done RLHF boosts scores on popular LLM evaluations.
We've seen how different styles actually can meaningfully improve evaluations with Llama 3 [@dubey2024llama]. 
The Llama 3 Instruct models scored extremely high on ChatBotArena, and it's accepted as being because they had a more fun personality -- they were more succinct and clever than other models of their era. 
Regardless of the benchmark scores that many LLM users are obsessed with, if RLHF is going to make language models simply more fun, that is delivered value.

Throughout this chapter, the term "chattiness" is used to encompass the growing length of responses from models training with RLHF, but it also encompasses techniques like heavy markdown use, emojis, and formatting the answer in bulleted lists.

## The Chattiness Balance

RLHF or preference fine-tuning methods are being used by countless people to boost scores like AlpacaEval and other automatic chat leaderboards (which use LLM-as-a-judge to approximate how helpful, harmless, and honest an agent is across simple conversational tasks), but the massive gains RLHF confers here come without shifting scores proportionally on harder-to-game evaluations like ChatBotArena. 
The tension is that while RLHF methods give a measurable improvement on these models, that training doesn't always transfer into performance that people care about.
Through the establishment of the RLHF literature, a large swath of models have been released with related methods to boost the "alignment" of a model with RLHF, but they often took it way too far and published evaluation scores that were anywhere from misleading to meaningless.

These RLHF methods motivated by alignment, when done right, make the models easier to work with and more enjoyable. 
This often comes with clear improvements on evaluation tools like MT Bench or AlpacaEval. 

In the fall of 2023, there was a peak in the debate over direct preference optimization (DPO) and its role relative to proximal policy optimization (PPO) and other RL-based methods for preference fine-tuning -- the balance of chat evaluations to real world performance was at the center of this (For more technical discussion on the trade-offs, see Chapter 12, Ivison et. al 2024 [@ivison2024unpacking], or this talk, [https://youtu.be/YJMCSVLRUNs](https://youtu.be/YJMCSVLRUNs)).
The problem is that you can also use techniques like DPO and PPO in feedback loops or in an abundance of data to actually severely harm the model on other tasks like mathematics or coding in a trade for this chat performance.

During the proliferation of the DPO versus PPO debate there were many papers that came out with incredible benchmarks but no model weights that gathered sustained, public usage because these models were not robust in general usage. 
When applying RLHF in the fall of 2023 or soon after, there is no way to make an aligned version of a 7 billion parameter model actually beat GPT-4 across comprehensive benchmarks (this sort of comparison will hold, where small models of the day cannot robustly beat the best, large frontier models). 
It seems obvious, but there are always papers claiming these sort of results. 
@fig:DNO is from a paper called Direct Nash Optimization (DNO) that makes the case that their model is state-of-the-art or so on AlpacaEval for 7B models in April 2024 [@rosset2024direct].
For context, DNO is a batched, on-policy *iterative* alternative to reward-model+PPO (classic RLHF) or one-shot DPO that directly optimizes pairwise preferences (win-rate gaps) by framing alignment as finding a Nash equilibrium against a preference oracle.
These challenges emerge when academic incentives interface with technologies becoming of extreme interest to the broader society.

![Results from the paper on Direct Nash Optimization (DNO) highlighting their small model outperforming the likes of GPT-4. Rosset et al. 2024. License CC-BY.](images/dno-figure.png){#fig:DNO width=550px}

Even the pioneering paper Self Rewarding Language Models from January of 2024 [@yuan2025selfrewardinglanguagemodels] disclosed unrealistically strong scores on Llama 2 70B. 
At the time, of course, a 70B model can get closer to GPT-4 than a 7B model can (as we saw with the impressive Llama 3 releases in 2024), but it's important to separate the reality of models from the claims in modern RLHF papers. 
These models are tuned to narrow test sets and do not hold up well in real use versus the far larger models they claim to beat.
Many more methods have come and gone similar to this, sharing valuable insights and oversold results, which make RLHF harder to understand.

A symptom of models that have "funky RLHF" applied to them has often been a length bias. 
This got so common that multiple evaluation systems like AlpacaEval and WildBench both have linear length correction mechanisms in them. 
This patches the incentives for doping on chattiness to 'beat GPT-4' or the leading frontier model of the day, and creates a less gamified dynamic where shorter, useful models can actually win.

Regardless, aligning chat models only for chattiness now has a bit of a reputational tax associated with it in the literature, where it's acknowledged that these narrow methods can harm a model in other ways. 
This note from the original Alibaba Qwen models in 2023 is something that has been observed multiple times in early alignment experiments, exaggerating a trade-off between chattiness and performance [@qwen]. 

> We pretrained the models with a large amount of data, and we post-trained the models with both supervised fine-tuning and direct preference optimization. However, DPO leads to improvements in human preference evaluation but degradation in benchmark evaluation.

An early, good example of this tradeoff done right is a model like Starling Beta from March of 2024 [@zhu2024starling]. 
It's a model that was fine-tuned from another chat model, OpenChat [@wang2023openchat] (which was in fact trained by an entire other organization). 
Its training entirely focuses on a k-wise reward model training and PPO optimization, and moves it up 10 places in ChatBotArena. 
The average response length of the model increases, but in a way that's good enough to actually help the human raters.
Later examples, such as Olmo 3, actually are documented as undergoing substantial chat training, but where the authors prefer a final model checkpoint with higher math, coding, and reasoning scores instead of potential checkpoints that're highest on LLM-as-a-judge based chat benchmarks [@teamolmo2025olmo3].

A natural question is: Why does RLHF make model responses longer?
Fundamentally, evaluations like ChatBotArena have shown us that average users of models often like longer, complete answers when compared with terse responses. 
Longer answers can feel more thorough, helpful, or even trustworthy to users evaluating them quickly.
This does not represent the preference of *every* user, but these models are trained to match the average preferences of many data labelers, so RLHF tends to make models more verbose.
