# RLHF Book

A comprehensive guide to Reinforcement Learning from Human Feedback.

**[Read online](https://rlhfbook.com)** | **[Pre-order print](https://hubs.la/Q03Tc3cf0)**

## Code Library

Reference implementations for RLHF algorithms in `code/`:
- Policy gradient methods (PPO, REINFORCE, GRPO, RLOO, etc.)
- Reward model training (preference RM, ORM, PRM)
- Direct alignment methods

See [code/README.md](code/README.md) for setup and usage.

## Book Source

Book source files are in `book/`. Build locally:

```bash
make html   # Build HTML site
make pdf    # Build PDF (requires LaTeX)
```

See [book/README.md](book/README.md) for detailed build instructions.

## Repository Structure

```
rlhf-book/
├── book/                   # Book source and build files
│   ├── chapters/           # Markdown source (01-introduction.md, etc.)
│   ├── images/             # Figures referenced in chapters
│   ├── assets/             # Brand assets (covers, logos)
│   ├── templates/          # Pandoc templates (HTML, PDF, EPUB)
│   ├── scripts/            # Build utilities
│   └── data/               # Library data
├── code/                   # Reference implementations
│   ├── reward_modeling/    # RM training (preference, ORM, PRM)
│   ├── alignment/          # PPO, REINFORCE, GRPO, RLOO, DPO
│   └── ...
├── diagrams/               # Diagram source files
│   ├── specs/              # YAML specifications
│   ├── scripts/            # Generation scripts
│   └── generated/          # Output (PNG, SVG)
├── build/                  # Generated output (git-ignored)
├── Makefile                # Build system
└── metadata.yml            # Book metadata
```

## Citation

To cite this book, please use the following format:

```bibtex
@book{rlhf2025,
  author       = {Nathan Lambert},
  title        = {Reinforcement Learning from Human Feedback},
  year         = {2025},
  publisher    = {Online},
  url          = {https://rlhfbook.com},
}
```

## License

- Code: [MIT](LICENSE-Code.md)
- Chapters: [CC-BY-NC-SA-4.0](LICENSE-Chapters.md)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=natolambert/rlhf-book&type=Date)](https://www.star-history.com/#natolambert/rlhf-book&Date)
