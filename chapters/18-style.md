---
prev-chapter: "Over Optimization"
prev-url: "17-over-optimization.html"
next-chapter: "Character Training & Model Character"
next-url: "19-character.html"
---

# Style and Information

*This chapter draws on content from [two](https://www.interconnects.ai/p/how-rlhf-works-2) | [posts](https://www.interconnects.ai/p/gpt-4o-mini-changed-chatbotarena) on the role of style in post-training and evaluation of RLHF'd models.*

Early developments in RLHF gave it a reputation for being "just style transfer" or other harsh critiques on how RLHF manipulates the way information is presented in outputs.

Style transfer, has held back the RLHF narrative for two reasons. 

First, when people discuss style transfer, they don’t describe this as being important or exciting. 
Style is a never-ending source of human value, it’s why retelling stories can result in new bestselling books (such as [Sapiens](https://en.wikipedia.org/wiki/Sapiens:_A_Brief_History_of_Humankind)), and it is a fundamental part of continuing to progress our intellectual ecosystem. 
Style is intertwined with what the information is. 

Second, we’ve seen how different styles actually can improve evaluation improvements with Llama 3 [@dubey2024llama]. 
The Llama 3 Instruct models scored extremely high on ChatBotArena, and it’s accepted as being because they had a more fun personality. 
If RLHF is going to make language models simply more fun, that is delivered value.

## The Chattiness Paradox

TODO EDIT

RLHF or preference fine-tuning methods are being used mostly to boost scores like AlpacaEval and other automatic leaderboards without shifting the proportionally on harder-to-game evaluations like ChatBotArena. The paradox is that while alignment methods like DPO give a measurable improvement on these models that does transfer into performance that people care about, a large swath of the models doing more or less the same thing take it way too far and publish evaluation scores that are obviously meaningless.

For how methods like DPO can simply make the model better, some of my older articles on scaling DPO and if we even need PPO can help. These methods, when done right, make the models easier to work with and more enjoyable. This often comes with a few percentage point improvements on evaluation tools like MT Bench or AlpacaEval (and soon Arena Hard will show the same). The problem is that you can also use techniques like DPO and PPO in feedback loops or in an abundance of data to actually lobotomize the model at the cost of LLM-as-a-judge performance. There are plenty of examples.

Some of the models I’m highlighting here are academic papers that shouldn’t entirely be judged on “if the final model passes vibes tests,” but are illustrative of the state of the field. These are still useful papers, just not something everyone will immediately use for training state-of-the-art models. Those come from downstream papers.

During the proliferation of the DPO versus PPO debate there were many papers that came out with ridiculous benchmarks but no model weights that gathered sustained usage. When applying RLHF, there is no way to make an aligned version of a 7 billion parameter model actually beat GPT-4. It seems obvious, but there are papers claiming these results. Here’s a figure from a paper called Direct Nash Optimization (DNO) that makes the case that their model is state-of-the-art or so on AlpacaEval.

Even the pioneering paper Self Rewarding Language Models disclosed ridiculous scores on Llama 2 70B. A 70B model can get closer to GPT-4 than a 7B model can, as we have seen with Llama 3, but it’s important to separate the reality of models from the claims in modern RLHF papers. Many more methods have come and gone in the last few months. They’re the academic line of work that I’m following, and there’s insight there, but the methods that stick will be accompanied by actually useful models some number of months down the line.

Other players in industry have released models alone (rather than papers) that gamify these metrics. Two examples that come to mind are the Mistral 7B fine-tune from Snorkel AI or a similar model from Contextual trained with KTO. There are things in common here — using a reward model to further filter the data, repeated training via some sort of synthetic data feedback, and scores that are too good to be true.

A symptom of models that have “funky RLHF” applied to them has often been a length bias. This got so bad that multiple evaluation systems like AlpacaEval and WildBench both have linear length correction mechanisms in them. This patches the incentives for doping on chattiness to “beat GPT-4,” and adds a less gamified bug that shorter and useful models may actually win out. So far so good on the linear length controls.

Regardless, aligning chat models simply for chattiness still has a bit of a tax in the literature. This note from the Qwen models is something that has been seen multiple times in early alignment experiments. I suspect this is mostly about data.

We pretrained the models with a large amount of data, and we post-trained the models with both supervised finetuning and direct preference optimization. However, DPO leads to improvements in human preference evaluation but degradation in benchmark evaluation.

A good example of this tradeoff done right is a model like Starling Beta. It’s a model that was fine-tuned from another chat model, OpenChat, which was in fact trained by an entire other organization. It’s training entirely focuses on a k-wise reward model training and PPO optimization, and moves it up 10 places in ChatBotArena. The average response length of the model increases, but in a way that’s good enough to actually help the human raters.

### How Chattiness Emerges

TODO EDIT

Let’s round out this article with how RLHF is actually achieving chattiness at the parameter level. Most of the popular datasets for alignment these days are synthetic preferences where a model like GPT-4 rates outputs from other models as the winner or the loser. Given that GPT-4 is known to have length and style biases for outputs that match itself, most of the pieces of text in the “preferred” section of the dataset are either from an OpenAI model or are stylistically similar to it. The important difference is that not all of the pieces of text in the dataset will have that. They’re often generated from other open models like Alpaca, Vicuna, or more recent examples. These models have very different characteristics.

Next, now that we’ve established that we have a preference dataset where most of the chosen models are similar to ChatGPT (or some other model that is accepted to be “strong”), these alignment methods simply increase the probability of these sequences. The math is somewhat complicated, where the batches of data operate on many chosen-rejected pairs at once, but in practice, the model is doing credit assignment over sequences of tokens (subword pieces). Preference alignment for chattiness is making the sequences found in outputs of models like GPT-4 more likely and the sequences from other, weaker models less likely. Repeatedly, this results in models with longer generations and characteristics that people like more.

Those among you who are familiar with RLHF methods may ask if the KL constraint in the optimization should stop this from happening. The KL constraint is a distance term between the distribution of the original model and the resulting model. It helps make the optimization more robust to overoptimization, but that makes the border between good and bad models a bit more nuanced. Hence, the prevalence of vibes-based evaluations. Though, models tend to have enough parameters where they can change substantially and still satisfy the KL constraint on the data being measured — it can’t be the entire pertaining dataset, for example.

As more models than ChatGPT become prevalent and strong enough for creating synthetic data, the distribution of outcomes we can expect from our aligned models should shift. There are two key places where the data influences this process: 1) where the text used to train the model is generated and 2) which LLM is used to determine which answer is the “winner” and “loser” in the preference learning framework. While all of these models have licenses or terms of service that make this practice technically violate an agreement of use, we’ve had more than a year of progress in open alignment practices relying on them in the past, so I don’t expect it to change. Mistral AI is the only LLM provider that doesn’t have a term restricting training on outputs (as far as I know).

