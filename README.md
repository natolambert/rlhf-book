# RLHF Book

A comprehensive guide to Reinforcement Learning from Human Feedback (and a broad introduction to post-training language models).

**[Read online](https://rlhfbook.com)** | **[Pre-order print](https://hubs.la/Q03Tc3cf0)**

This book is my attempt to open-source all the knowledge I've gained working at the frontier of open models in the post-ChatGPT take off of language models.
When I started, many established methods like rejection sampling had no canonical reference.
On the other side, industry practices to make the models more personable -- colloquially called Character Training -- had no open research. 
It was obvious to me that there would be payoff to documenting, learning the fundamentals, carefully curating the references (in an era of AI slop), and everything in between would be a wonderful starting point for people.

Today, I'm adding code and seeing this as a home base for people who want to learn. 
You should use coding assistants to ask questions.
You should buy the physical book because the real world matters.
You should read the specific AI outputs tailored to you.

In the future I want to build more education resources to this, such as open source slide decks and more ways to learn.
In the end, with how impossible it is to measure human preferences, RLHF will never be a solved problem.

Thank you for reading. 
Thank you for contributing any feedback or engaging with the community.

-- Nathan Lambert, @natolambert

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

## Diagrams

The `diagrams/` directory contains source files for figures used in the book. These are designed to be reusable for presentations, blog posts, or your own learning materials. Generate them with:

```bash
cd diagrams && make all
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

## Contributors

Where I get the credit as the sole "author" and creator of this project, I've been super lucky to have many contributions from early readers. These have massively accelerated the editing progress and flat-out added meaningful content to the book. I'm happy to send substantive contributors free copies of the book and expect the internet goodwill to pay them back in unexpected ways.

See all [contributors](https://github.com/natolambert/rlhf-book/graphs/contributors).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=natolambert/rlhf-book&type=Date)](https://www.star-history.com/#natolambert/rlhf-book&Date)
