
# [Incomplete] Human Preferences for RLHF

The core of reinforcement learning from human feedback, also referred to as reinforcement learning from human preferences in early literature, is designed to optimize machine learning models in domains where specifically designing a reward function is hard.
The motivation for using humans as the reward signals is to obtain a indirect metric for the target reward.

The use of human labeled feedback data integrates the history of many fields.
Using human data alone is a well studied problem, but in the context of RLHF it is used at the intersection of multiple long-standing fields of study [@lambert2023entangled].

As an approximation, modern RLHF is the convergence of three areas of development:

1. Philosophy, psychology, economics, decision theory, and the nature of human preferences;
2. Optimal control, reinforcement learning, and maximizing utility; and
3. Modern deep learning systems.

Together, each of these areas brings specific assumptions at what a preference is and how it can be optimized, which dictates the motivations and design of RLHF problems.

## The Origins of Reward Models: Costs vs. Rewards vs. Preferences

### Specifying objectives: from logic of utility to reward functions

### Implementing optimal utility

### Steering preferences

### Value alignment's role in RLHF

## From Design to Implementation

Many of the principles discussed earlier in this chapter are further specified in the process of implementing the modern RLHF stack, adjusting the meaning of RLHF.

## Limitations of RLHF

The specifics of obtaining data for RLHF is discussed further in Chapter 6.
For an extended version of this chapter, see [@lambert2023entangled].