---
prev-chapter: "Definitions & Background"
prev-url: "03-setup"
page-title: Training Overview
next-chapter: "The Nature of Preferences"
next-url: "05-preferences"
---

# Training Overview

In this chapter we provide a cursory overview of RLHF training, before getting into the specifics later in the book.
RLHF, while optimizing a simple loss function, involves training multiple, different AI models in sequence and then linking them together in a complex, online optimization.

Here, we introduce the core objective of RLHF, which is optimizing a proxy of reward of human preferences with a distance-based regularizer (along with showing how it relates to classical RL problems).
Then we showcase canonical recipes which use RLHF to create leading models to show how RLHF fits in with the rest of post-training methods.
These example recipes will serve as references for later in the book, where we describe different optimization choices you have when doing RLHF, and we will point back to how different key models used different steps in training.

## Problem Formulation

The optimization of reinforcement learning from human feedback (RLHF) builds on top of the standard RL setup.
In RL, an agent takes actions $a_t$ sampled from a policy $\pi(a_t\mid s_t)$ given the state of the environment $s_t$ to maximize reward $r(s_t,a_t)$ [@sutton2018reinforcement].
Traditionally, the environment evolves according to transition (dynamics) $p(s_{t+1}\mid s_t, a_t)$ with an initial state distribution $\rho_0(s_0)$.
Together, the policy and dynamics induce a trajectory distribution:

$$p_{\pi}(\tau)=\rho_0(s_0)\prod_{t=0}^{T-1}\pi(a_t\mid s_t)\,p(s_{t+1}\mid s_t,a_t).$$ {#eq:rl_dynam}

Across a finite episode with horizon $T$, the goal of an RL agent is to solve the following optimization:

$$J(\pi) = \mathbb{E}_{\tau \sim p_{\pi}} \left[ \sum_{t=0}^{T-1} \gamma^t r(s_t, a_t) \right],$$ {#eq:rl_opt}

For continuing tasks, one often takes $T\to\infty$ and relies on discounting ($\gamma<1$) to keep the objective well-defined.
$\gamma$ is a discount factor from 0 to 1 that balances the desirability of near- versus future-rewards.
Multiple methods for optimizing this expression are discussed in Chapter 11.

![Standard RL loop](images/rl.png){#fig:rl width=320px .center}

A standard illustration of the RL loop is shown in @fig:rl and (compare this to the RLHF loop in @fig:rlhf).

### Example RL Task: CartPole

To make the transition function concrete, consider the classic *CartPole* (inverted pendulum) control task.

![CartPole environment showing state variables ($x$, $\dot{x}$, $\theta$, $\dot{\theta}$) and actions ($\pm F$).](images/cartpole.png){#fig:cartpole width=400px .center}

- **State ($s_t$)**: the cart position/velocity and pole angle/angular velocity,

  $$s_t = (x_t,\,\dot{x}_t,\,\theta_t,\,\dot{\theta}_t).$$

- **Action ($a_t$)**: apply a left/right horizontal force to the cart, e.g. $a_t \in \{-F, +F\}$.

- **Reward ($r$)**: a simple reward is $r_t = 1$ each step the pole remains balanced and the cart stays on the track (e.g. $|x_t| \le 2.4$ and $|\theta_t| \le 12^\circ$), and the episode terminates when either bound is violated.

- **Dynamics / transition ($p(s_{t+1}\mid s_t,a_t)$)**: in many environments the dynamics are deterministic (so $p$ is a point mass) and can be written as $s_{t+1} = f(s_t,a_t)$ via Euler integration with step size $\Delta t$. A standard simplified CartPole update uses constants cart mass $m_c$, pole mass $m_p$, pole half-length $l$, and gravity $g$:

  $$\text{temp} = \frac{a_t + m_p l\,\dot{\theta}_t^2\sin\theta_t}{m_c + m_p}$$

  $$\ddot{\theta}_t = \frac{g\sin\theta_t - \cos\theta_t\,\text{temp}}{l\left(\tfrac{4}{3} - \frac{m_p\cos^2\theta_t}{m_c + m_p}\right)}$$

  $$\ddot{x}_t = \text{temp} - \frac{m_p l\,\ddot{\theta}_t\cos\theta_t}{m_c + m_p}$$

  $$x_{t+1}=x_t+\Delta t\,\dot{x}_t,\quad \dot{x}_{t+1}=\dot{x}_t+\Delta t\,\ddot{x}_t,$$
  $$\theta_{t+1}=\theta_t+\Delta t\,\dot{\theta}_t,\quad \dot{\theta}_{t+1}=\dot{\theta}_t+\Delta t\,\ddot{\theta}_t.$$

This is a concrete instance of the general setup above: the policy chooses $a_t$, the transition function advances the state, and the reward is accumulated over the episode.

### Manipulating the Standard RL Setup

The RL formulation for RLHF is seen as a less open-ended problem, where a few key pieces of RL are set to specific definitions in order to accommodate language models.
There are multiple core changes from the standard RL setup to that of RLHF:
Table @tbl:rl-vs-rlhf summarizes these differences between standard RL and the RLHF setup used for language models.

1. **Switching from a reward function to a reward model.** In RLHF, a learned model of human preferences, $r_\theta(s_t, a_t)$ (or any other classification model) is used instead of an environmental reward function. This gives the designer a substantial increase in the flexibility of the approach and control over the final results, but at the cost of implementation complexity. In standard RL, the reward is seen as a static piece of the environment that cannot be changed or manipulated by the person designing the learning agent.
2. **No state transitions exist.** In RLHF, the initial states for the domain are prompts sampled from a training dataset and the "action" is the completion to said prompt. During standard practices, this action does not impact the next state and is only scored by the reward model.
3. **Response level rewards.** Often referred to as a bandit problem, RLHF attribution of reward is done for an entire sequence of actions, composed of multiple generated tokens, rather than in a fine-grained manner.

::: {.table-wrap}
| Aspect | Standard RL | RLHF (language models) |
|---|---|---|
| Reward signal | Environment reward function $r(s_t,a_t)$ | Learned reward / preference model $r_\theta(x,y)$ (prompt $x$, completion $y$) |
| State transition | Yes: dynamics $p(s_{t+1}\mid s_t,a_t)$ | Typically no: prompts $x$ sampled from a dataset; the completion does not define the next prompt |
| Action | Single environment action $a_t$ | A completion $y$ (a sequence of tokens) sampled from $\pi_\theta(\cdot\mid x)$ |
| Reward granularity | Often per-step / fine-grained | Usually response-level (bandit-style) over the full completion |
| Horizon | Multi-step episode ($T>1$) | Often single-step ($T=1$), though multi-turn can be modeled as longer-horizon |
Table: Key differences between standard RL and RLHF for language models. {#tbl:rl-vs-rlhf}
:::

Given the single-turn nature of the problem, the optimization can be re-written without the time horizon and discount factor (and the reward models):
$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[r_\theta(s_t, a_t) \right].$$ {#eq:rl_opt_int}

In many ways, the result is that while RLHF is heavily inspired by RL optimizers and problem formulations, the actual implementation is very distinct from traditional RL.

![Standard RLHF loop](images/rlhf.png){#fig:rlhf}

### Fine-tuning and Regularization

In traditional RL problems, the agent must learn from a randomly initialized policy, but with RLHF, we start from a strong pretrained base model with many initial capabilities.
This strong prior for RLHF induces a need to control the optimization from drifting too far from the initial policy.
In order to succeed in a fine-tuning regime, RLHF techniques employ multiple types of regularization to control the optimization.
The goal is to allow the reward maximization to still occur without the model succumbing to over-optimization, as discussed in Chapter 18.
The most common change to the optimization function is to add a distance penalty on the difference between the current RLHF policy and the starting point of the optimization:

$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[r_\theta(s_t, a_t)\right] - \beta  \mathcal{D}_{\text{KL}}(\pi_{\text{RL}}(\cdot|s_t) \| \pi_{\text{ref}}(\cdot|s_t)).$$ {#eq:rlhf_opt_eq}

Within this formulation, a lot of study into RLHF training goes into understanding how to spend a certain "KL budget" as measured by a distance from the initial model.
For more details, see Chapter 8 on Regularization.


### Optimization Tools

In this book, we detail many popular techniques for solving this optimization problem.
The popular tools of post-training include:

- **Reward modeling** (Chapter 7): Where a model is trained to capture the signal from collected preference data and can then output a scalar reward indicating the quality of future text.
- **Instruction fine-tuning** (Chapter 9): A prerequisite to RLHF where models are taught the question-answer format used in the majority of language modeling interactions today by imitating preselected examples.
- **Rejection sampling** (Chapter 10): The most basic RLHF technique where candidate completions for instruction fine-tuning are filtered by a reward model imitating human preferences.
- **Policy gradients** (Chapter 11): The reinforcement learning algorithms used in the seminal examples of RLHF to update parameters of a language model with respect to the signal from a reward model.
- **Direct alignment algorithms** (Chapter 12): Algorithms that directly optimize a policy from pairwise preference data, rather than learning an intermediate reward model to then optimize later.

Modern RLHF-trained models always utilize instruction fine-tuning followed by a mixture of the other optimization options.

## Canonical Training Recipes

Over time various models have been identified as canonical recipes for RLHF specifically or post-training generally.
These recipes reflect data practices and model abilities at the time.
As the recipes age, training models with the same characteristics becomes easier and takes fewer data.
There is a general trend of post-training involving more optimization steps with more training algorithms across more diverse training datasets and evaluations.

### InstructGPT

Around the time ChatGPT first came out, the widely accepted ("canonical") method for post-training an LM had three major steps, with RLHF being the central piece [@lambert2022illustrating] [@ouyang2022training] [@bai2022training].
The three steps taken on top of a "base" language model (the next-token prediction model trained on large-scale web text) are summarized below in @fig:rlhf-basic-repeat:

1. **Instruction tuning on ~10K examples**: This teaches the model to follow the question-answer format and teaches some basic skills from primarily human-written data.
2. **Training a reward model on ~100K pairwise prompts**: This model is trained from the instruction-tuned checkpoint and captures the diverse values one wishes to model in their final training. The reward model is the optimization target for RLHF.
3. **Training the instruction-tuned model with RLHF on another ~100K prompts**: The model is optimized against the reward model with a set of prompts that the model generates over before receiving ratings.

Once RLHF was done, the model was ready to be deployed to users. This recipe is the foundation of modern RLHF, but recipes have evolved substantially to include more stages and more data.

![A rendition of the early, three stage RLHF process with SFT, a reward model, and then optimization.](images/rlhf-basic.png){#fig:rlhf-basic-repeat}

### Tülu 3

Modern versions of post-training involve many, many more model versions and training stages (i.e. well more than the 5 RLHF steps documented for Llama 2 [@touvron2023llama]). 
An example is shown below in @fig:rlhf-complex where the model undergoes numerous training iterations before convergence.

![A rendition of modern post-training with many rounds.](images/rlhf-complex.png){#fig:rlhf-complex}

The most complex models trained in this era and onwards have not released full details of their training process.
Leading models such as ChatGPT or Claude circa 2025 involve many, iterative rounds of training.
This can even include techniques that train specialized models and then merge the weights together to get a final model capable on many subtasks [@li2022branch] (e.g. Cohere's Command A [@cohere2025command]).

![A summary of the Tülu 3 recipe with target skills and multi-step training recipe. Lambert et al. 2024, License CC-BY.](images/tulu3.png){#fig:tulu-3}

A fully open example version of this multi-stage version of post-training where RLHF plays a major role is Tülu 3.
The Tülu 3 recipe consists of three stages:

1. **Instruction tuning on ~1M examples**: This primarily synthetic data from a mix of frontier models such as GPT-4o and Llama 3.1 405B teaches the model general instruction following and serves as the foundation of a variety of capabilities such as mathematics or coding.
2. **On-policy preference data on ~1M preference pairs**: This stage substantially boosts the chattiness (e.g. ChatBotArena or AlpacaEval 2) of the model while also improving skills mentioned above in the instruction tuning stage.
3. **Reinforcement Learning with Verifiable Rewards on ~10K prompts**: This stage is a small-scale reinforcement learning run to boost core skills such as mathematics while maintaining overall performance (and is now seen as a precursor to modern reasoning models such as DeepSeek R1).

The recipe has been successfully applied to Llama 3.1 [@lambert2024t], OLMo 2 [@olmo20242], and SmolLM models [@alrashed2024smoltulu].

### DeepSeek R1

With the rise of reasoning language models, such as OpenAI's o1, the best practices in post-training evolved again to re-order and redistribute compute across training stages.
The clearest documentation of a reasoning model post-training recipe is DeepSeek R1 [@guo2025deepseek], which has been mirrored by Alibaba's larger Qwen 3 models (i.e. only the 32B and 225B MoE models) [@yang2025qwen3] or Xiaomi's MiMo 7B [@xia2025mimo].
The DeepSeek recipe follows:

1. **"Cold-start" of 100K+ on-policy reasoning samples**: This data is sampled from an earlier RL checkpoint, R1-Zero, and heavily filtered to instill a specific reasoning process on the model. DeepSeek uses the term cold-start to describe how RL is learned from little supervised data.
2. **Large-scale reinforcement learning training**: This stage repeatedly covers reasoning problems with the model, running RLVR "until convergence" on a variety of benchmarks.
3. **Rejection sampling** on 3/4 reasoning problems and 1/4 general queries to start the transition to a general-purpose model.
4. **Mixed reinforcement learning training** on reasoning problems (verifiable rewards) with general preference tuning reward models to polish the model.

As above, there are evolutions of the recipe, particularly with steps 3 and 4 to finalize the model before exposing it to users.
Many models start with tailored instruction datasets with chain-of-thought sequences that are heavily filtered and polished from existing models, providing a fast step to strong behaviors with SFT alone before moving onto RL [@seed2025seed].

