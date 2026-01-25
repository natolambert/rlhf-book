# Direct Alignment Algorithms

**Status: Coming Soon**

This directory will contain educational implementations of direct alignment methods
that bypass explicit reward model training.

## Planned Implementations

| Algorithm | Paper | Status |
|-----------|-------|--------|
| DPO | Rafailov et al., 2023 | Coming soon |
| IPO | Azar et al., 2023 | Coming soon |
| KTO | Ethayarajh et al., 2024 | Coming soon |
| SimPO | Meng et al., 2024 | Coming soon |

## Book Reference

See **Chapter 12: Direct Alignment** of [RLHF Book](https://rlhfbook.com) for:
- Mathematical derivations from reward modeling to direct optimization
- Comparison of DPO variants
- When to use direct alignment vs PPO/GRPO

## Contributing

If you'd like to contribute an implementation, please:
1. Follow the style of existing scripts in `reward_models/` and `policy_gradients/`
2. Include clear attribution for any code adapted from other sources
3. Add wandb logging support
4. Include a demo/evaluation section
