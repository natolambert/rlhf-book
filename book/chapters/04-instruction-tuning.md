<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "Training Overview"
prev-url: "03-training-overview"
page-title: Instruction Tuning
search-title: "Chapter 4: Instruction Tuning"
next-chapter: "Reward Models"
next-url: "05-reward-models"
lecture-video: "https://youtu.be/4gIwiSPmQkU"
lecture-label: "Lecture 2: IFT, Reward Modeling, Rejection Sampling (Chap. 4, 5, & 9)"
---

# Instruction Fine-tuning

Early large pretrained language models were trained with a next-token prediction objective and, by default, did not come with an explicit interface for following instructions.
Around the release of GPT-3 [@brown2020language], prompting and in-context learning became a widely used way to adapt a single model to many tasks (though task-specific fine-tuning remained common), by showing examples in-context and asking the model to complete a similar task.
A practical next step was instruction fine-tuning, which teaches the model to respond in an instruction-response format rather than just continuing text.
For example, given the prompt "What is the capital of France?", a base model might continue with "What is the capital of Germany? What is the capital of Italy?..." — simply extending the pattern of questions — while an instruction-tuned model would respond with "The capital of France is Paris."

Instruction fine-tuning took off when two lines of work converged.
First, NLP shifted from bespoke-fine-tuning task setups to a unified "text-to-text" or instruction framing, which made it straightforward to standardize diverse datasets and train a single model across many tasks.
Prominent examples of unifying the framework for tasks include *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5 models) [@raffel2020exploring], *Finetuned Language Models Are Zero-Shot Learners* (FLAN dataset) [@wei2021finetuned], *Multitask Prompted Training Enables Zero-Shot Task Generalization* (T0 models) [@sanh2021multitask], and *Cross-Task Generalization via Natural Language Crowdsourcing Instructions* (Natural Instructions dataset) [@mishra2021cross].
Second, scaling pretrained LMs and the rise of prompting/in-context learning showed that a single model could generalize across tasks, but that generalization becomes far more reliable when the model is explicitly trained on instruction-response examples.
Together, these trends led to an era of fine-tuning pretrained language models on large collections of instructions—what is now commonly called instruction fine-tuning (IFT), or supervised fine-tuning (SFT), in which training general models became accessible to wider audiences.
<!-- Historically, until RLHF and related methods, all fine-tuning was **instruction fine-tuning** (IFT), also known as **supervised fine-tuning** (SFT). -->

Since its discovery, instruction fine-tuning, also called colloquially just *instruction tuning*, has matured and is standard practice across many language modeling pipelines.
At its core, IFT is the simplest method for adapting language models to a desired task distribution.
It serves as the foundation for RLHF by preparing the model for a format of instructions that is known as question-answering, and it is the first tool used by those attempting to apply modern techniques to new domains.
Without a basic level of instruction-following abilities, most of the pipelines we discuss in this book—from preference data collection to online RLHF optimization—cannot be performed.

Instruction fine-tuning generally is covered extensively elsewhere and is supervised learning at its core, so this chapter focuses on the practical details that matter most for RLHF practitioners: how training data is formatted and structured.
Decisions on data and formatting are directly leveraged in the later training stages to create a common language for the model to absorb post-training data.

## Chat Templates and the Structure of Instructions

The post-training process begins with defining a pattern to format user queries so that they are easily readable by a language model that processes information through a tokenizer.
When using a pretrained language model, the prompting is quite simple. The model only knows a few tokens: a beginning-of-sequence token (e.g., `<bos_token>`), an end-of-sequence token (e.g., `<eos_token>`), and a padding token (to manage training on batches with empty components).
This means, to prompt a base model, the user inputs a sequence of tokens for the model to continue from, such as:

```text
<bos_token> The capital of the United States is
```

Then, the model would generate tokens until it runs out of its context window, or it generates the end-of-sequence token.

All post-training stages, from instruction tuning to RLHF and other methods, rely on this formatting to train the model.
The tool that handles the structure of the interaction with the user is called the **chat template**. 

An example which we will break down is below:

```jinja
{% if messages[0]['role'] == 'system' %}
    {# If the conversation begins with a system message, treat it as a special first turn.
       We set an offset so the user/assistant alternation check lines up correctly. #}
    {% set offset = 1 %}
{% else %}
    {# No system message: user should be the first non-empty turn. #}
    {% set offset = 0 %}
{% endif %}

{# Emit the beginning-of-sequence token (model-specific). #}
{{ bos_token }}

{# Serialize each message into the model's chat-markup tokens. #}
{% for message in messages %}
    {# Enforce role alternation: (system), user, assistant, user, assistant, ...
       The boolean expression compares "is this a user message?" against whether the
       current index (plus offset) is expected to be user or assistant. #}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {# Wrap each message with special tokens:
       - <|im_start|><role>\n
       - message content (trimmed)
       - <|im_end|>\n
       This produces a single flat token sequence the LM can train on. #}
    {{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}
{% endfor %}

{# Optionally append an "assistant" start tag with no content.
   This cues generation to continue from the assistant role. #}
{% if add_generation_prompt %}
    {{ '<|im_start|>assistant\n' }}
{% endif %}
```
This is the raw code for transforming a list of dictionaries in Python containing messages and roles into tokens that a language model can predict from.

All information passed into models is assigned a role.
The traditional three roles are `system`, `user`, and `assistant`.

The `system` tag is only used for the first message of the conversation; it holds instructions for the agent in text that will not be received from or exposed to the user.
These **system prompts** are used to provide additional context to the models, such as the date and time, or to patch behaviors.
As a fun example, models can be told things such as "You are a friendly chatbot who always responds in the style of a pirate."

Next, the two other roles are straightforward: **user** holds the messages from the person using the AI, and **assistant** holds the responses from the model (that is engaging as an AI assistant).

In order to translate all this information into tokens, we use the code listing above that we started with.
The model has a series of *special tokens* that separate the various messages from each other.
If we run the above code with the example query "How many helicopters can a human eat in one sitting?", the token sequence passed into the model would look as follows:

```text
<|im_start|>system
You are a friendly chatbot who always responds in the style of a pirate<|im_end|>
<|im_start|>user
How many helicopters can a human eat in one sitting?<|im_end|>
<|im_start|>assistant
```

Notice how the final tokens in the sequence are `<|im_start|>assistant`. This is how the model knows to continue generating tokens until it finally generates its end-of-sequence token, which in this case is `<|im_end|>`.

By packing all question-answer pair data (and downstream preference tuning data) into this format, modern language models follow it with perfect consistency. This is the language that instruction tuned models use to exchange information with users and the models running on GPUs or other computing devices.

The behavior can be extended naively to multiple turns, such as shown below:

```text
<|im_start|>system
You are a friendly chatbot who always responds in the style of a pirate<|im_end|>
<|im_start|>user
How many helicopters can a human eat in one sitting?<|im_end|>
<|im_start|>assistant
Oh just 6.<|im_end|>
<|im_start|>user
Are you sure about that?<|im_end|>
<|im_start|>assistant
```

In the open ecosystem, the standard method for applying the chat template to a list of messages uses a Jinja snippet -- a lightweight Python templating language -- stored in the tokenizer configuration, as `apply_chat_template`.

The above chat template is a derivative of OpenAI's Chat Markup Language (ChatML), which was an early attempt to standardize message formatting.
Now, OpenAI and other model providers use a hierarchical system where the user can configure a system message, yet there are higher-level instructions that may or may not be revealed to the user [@wallace2024instruction].

Many other chat templates exist. Some other examples include Zephyr's [@tunstall2023zephyr]:

```text
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s>
<|user|>
How many helicopters can a human eat in one sitting?</s>
<|assistant|>
```

Or Tülu's:

```text
<|user|>
How are you doing?
<|assistant|>
I'm just a computer program, so I don't have feelings, but I'm functioning as expected. How can I assist you today?<|endoftext|>
```

Beyond this, many chat templates include formatting and other tokens for tasks such as tool-use.


## Best Practices of Instruction Tuning

Instruction tuning as the foundation of post-training and creating helpful language models is well-established.
There are many ways to achieve successful instruction tuning.
For example, efficient fine-tuning with quantization of some model parameters makes training very accessible [@dettmers2023qlora].
Also, in narrow domains such as chat alignment, i.e., without harder skills such as math or code, small, focused datasets can achieve strong performance [@zhou2023lima].

Soon after the release of ChatGPT, human datasets with as few as 10K samples such as No Robots were state-of-the-art [@no_robots].
Years later, large-scale synthetic datasets work best [@lambert2024t] on most tasks.

A few principles remain:

- High-quality data is key to performance. The completions are what the model actually learns from (in many cases the prompts are not predicted over so the model does not learn to predict prompts).
- Around 1M prompts can be used to create a model capable of excellent RLHF and post-training. Further scaling can still help, but returns diminish quickly.
- The best prompts are those in a similar distribution to downstream tasks of interest.
- If multiple stages of training are done after instruction tuning, the models can recover from some noise in the instruction-tuning data. Optimizing the overall optimization is more important than each individual stage.

## Implementation Details

While the loss function is the same as pretraining, there are a few key implementation details that differ from the setting used for pretraining.
Many practices, such as deciding on the types of parallelism used to shard models across many GPUs are the same as pretraining, just the total number of machines used is often lower (for the first technical change listed below):

- **Smaller batch sizes**: Compared to pretraining, instruction tuning (and other post-training techniques such as preference fine-tuning) use substantially smaller batch sizes to optimize well on a narrower data distribution while preserving the model's generalization from pretraining. For example, OLMo 2 uses a batch size of 1024 packed-rows for the 7B and 2048 for the 13B pretraining, where these models have a total context length of 4096 tokens, and each row in the batch is a combination of documents that fills the sequence length. For post-training, both these models only use a batch size of 256 *prompts* [@olmo20242], without the filling to the full sequence length (for far fewer non-masked tokens per batch). The smaller batch sizes mean that these training jobs cannot be sharded across as many devices as pretraining -- in practice, distributed training setups have minimum per-device batch sizes, so if you're trying to retain a smaller global batch size for SFT you can use cumulatively fewer GPUs. In practice the batch size forcing a smaller concurrent GPU allotment per training job is not a limiting factor because the training token counts for SFT are much smaller than pretraining, and training for multiple seeds is needed in post-training to obtain the best final performance.
- **Prompt masking**: When pretraining, every token in the batch is predicted autoregressively and the loss is then applied to them. For instruction tuning, the prompt tokens are masked out so the model isn't learning to accurately predict user queries -- just responses. The same applies for other post-training algorithms.
- **Multi-turn masking**: For multi-turn conversations, there are two common masking choices. (1) *Final-turn only*: only the tokens in the final assistant turn are included in the loss, while all earlier context (including earlier assistant turns) is masked. Long conversations can still be "unrolled" into multiple training samples: for a conversation of $N$ turns, each example predicts one assistant response while masking all prior context and excluding any future turns. (2) *Mask user turns only*: all user turns are masked, but *every* assistant turn is included in the loss. You can still unroll in this setting if you want more (shorter) training examples, but the key difference is that intermediate assistant replies are trained on directly.
- **Same loss function as pretraining:** Instruction tuning uses the same autoregressive loss function used in pretraining language models, but with substantially different data and masking (training only on full sequences, whereas pretraining documents can be split across batches), etc.
- **Learning rate:** SFT typically uses a learning rate one to two orders of magnitude smaller than pretraining to best manage the different optimization dynamics (smaller datasets, smaller batches, and a strong pretrained initialization all favor more conservative updates). For example, OLMo 2 uses a peak learning rate of $3 \times 10^{-4}$ for pretraining but $1 \times 10^{-5}$ for SFT [@olmo20242]. OLMo 3 uses a higher SFT learning rate of $5\text{-}8 \times 10^{-5}$ [@teamolmo2025olmo3], in part because its training infrastructure uses sequence packing, which fits multiple examples into each training sequence and increases the effective batch size measured in useful tokens. Larger batches produce lower-variance gradient estimates, which in turn supports a higher learning rate without destabilizing training -- a relationship known as the linear scaling rule. The learning rate is commonly warmed up over a small fraction of training steps before decaying linearly. In practice, teams often sweep over multiple learning rates and select the best checkpoint on a held-out evaluation suite [@teamolmo2025olmo3].
