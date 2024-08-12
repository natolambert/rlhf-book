# Rejection Sampling

Rejection Sampling (RS) is a popular and simple baseline for performing preference fine-tuning. 
Rejection sampling operates by curating new candidate instructions, filtering them based on a trained reward model, and then fine-tuning the original model only on the top completions.

The name originates from computational statistics  [@gilks1992adaptive], where one wishes to sample from a complex distribution, but does not have a direct method to do so.
To alleviate this, one samples from a simpler to model distribution and uses a heuristic to check if the sample is permissible.
With language models, the target distribution is high-quality answers to instructions, the filter is a reward model, and the sampling distribution is the current model.

## Related works

Many prominent RLHF and preference fine-tuning papers have used rejection sampling as a baseling, but a canonical implementation and documentation does not exist

WebGPT [@nakano2021webgpt], Anthropic's Helpful and Harmless agent[@bai2022training], OpenAI's popular paper on process reward models [@lightman2023let], Llama 2 Chat models [@touvron2023llama], and other seminal works all use this baseline.