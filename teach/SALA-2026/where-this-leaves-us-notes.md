## Where this leaves us

- Tool use and agentic models are the next frontier for post-training.
- RLHF still matters, but it is now one part of a much larger stack.
- The best models combine instruction tuning, preference tuning, RL on verifiable tasks, and product-level scaffolding.
- The empirical trend is straightforward: models are getting consistently better as we scale post-training compute, data, and environments.
- The open question is no longer "does RLHF matter?" but "how do we combine these tools to reliably unlock more capability?"

Speaker notes:

RLHF was the technique that got everyone to take post-training seriously, but it is no longer the whole story.
What comes next is tool use, multi-turn environments, agentic training, and reinforcement learning in settings where models can act, verify, and recover from mistakes.
That does not make RLHF obsolete.
It makes RLHF foundational: one ingredient in a broader recipe for making models useful.
And the broad trend across o1, DeepSeek, Cursor, and OLMo is that these systems are not plateauing.
With more post-training compute and better environments, they keep improving.

Paste-ready slide block:

---

## Where this leaves us

- Tool use and agentic models are the next frontier.
- RLHF is still important, but it is now one piece of a much bigger post-training stack.
- Modern recipes mix SFT, preference tuning, RLVR, and environment feedback.
- The consistent trend is that models keep getting better as post-training scales.
- The central question has shifted from "does RLHF matter?" to "what training mixture best unlocks capability?"
