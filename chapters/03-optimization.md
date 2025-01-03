---
prev-chapter: "Home"
prev-url: "www.rlhfbook.com"
next-chapter: "The Nature of Preferences"
next-url: "02-preferences.html"
---

# Problem Formulation

The optimization of reinforcement learning from human feedback (RLHF) builds on top of the standard RL setup.
In RL, an agent takes actions, $a$, sampled from a policy, $\pi$, with respect to the state of the environment, $s$, to maximize reward, $r$.
Traditionally, the environment evolves with respect to a transition or dynamics function $p(s_{t+1}|s_t,a_t)$.
Hence, across a finite episode, the goal of an RL agent is to solve the following optimization:

$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \right],$$

where $\gamma$ is a discount factor from 0 to 1 that balances the desirability of near- versus future-rewards.
Multiple methods for optimizing this expression are discussed in Chapter 11.

![Standard RL loop](images/rl.png){#fig:rl width=320px .center}

A standard illustration of the RL loop is shown in @fig:rl and how it compares to @fig:rlhf.


## Manipulating the standard RL setup

There are multiple core changes from the standard RL setup to that of RLHF:

1. Switching from a reward function to a reward model. In RLHF, a learned model of human preferences, $r_\theta(s_t, a_t)$ (or any other classification model) is used instead of an environmental reward function. This gives the designer a substantial increase in the flexibility of the approach and control over the final results.
2. No state transitions exist. In RLHF, the initial states for the domain are prompts sampled from a training dataset and the ``action'' is the completion to said prompt. During standard practices, this action does not impact the next state and is only scored by the reward model.
3. Response level rewards. Often referred to as a Bandits Problem, RLHF attribution of reward is done for an entire sequence of actions, composed of multiple generated tokens, rather than in a fine-grained manner. 

Given the single-turn nature of the problem, the optimization can be re-written without the time horizon and discount factor (and the reward models):
$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[r_\theta(s_t, a_t) \right],$$

In many ways, the result is that while RLHF is heavily inspired by RL optimizers and problem formulations, the action implementation is very distinct from traditional RL.

![Standard RLHF loop](images/rlhf.png){#fig:rlhf}

## Finetuning and regularization

RLHF is implemented from a strong base model, which induces a need to control the optimization from straying too far from the initial policy.
In order to succeed in a finetuning regime, RLHF techniques employ multiple types of regularization to control the optimization.
The most common change to the optimization function is to add a distance penalty on the difference between the current RLHF policy and the starting point of the optimization:

$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[r_\theta(s_t, a_t)\right] - \beta  \mathcal{D}_{KL}(\pi^{\text{RL}}(\cdot|s_t) \| \pi^{\text{ref}}(\cdot|s_t)).$$

Within this formulation, a lot of study into RLHF training goes into understanding how to spend a certain ``KL budget'' as measured by a distance from the initial model.
For more details, see Chapter 8 on Regularization.