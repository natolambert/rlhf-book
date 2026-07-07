# RLHF Book

A comprehensive guide to Reinforcement Learning from Human Feedback (and a broad introduction to post-training language models).

**[Read online](https://rlhfbook.com)** | **Order print on [Manning](https://hubs.la/Q03Tc3cf0) or [Amazon](https://amzn.to/4cwCDJQ)** | **Join [Discord Community](https://discord.gg/yz5AwK4gBR)**

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
│   ├── instruction_tuning/ # SFT a base model with chat templates
│   ├── policy_gradients/   # PPO, REINFORCE, GRPO, RLOO
│   ├── reward_models/      # Preference RM, ORM, PRM training
│   ├── direct_alignment/   # DPO and variants
│   ├── rejection_sampling/ # Best-of-N rejection sampling
│   └── distillation/       # On-policy distillation (SDPO)
├── diagrams/               # Diagram source files
│   ├── scripts/            # Python generation scripts
│   ├── tikz/               # LaTeX/TikZ sources
│   └── specs/              # YAML specifications
├── teach/                  # Teaching materials (courses, slides)
├── build/                  # Generated output (git-ignored)
└── Makefile                # Build system
```

## Code Library

Reference implementations for RLHF algorithms in `code/`:
- Instruction tuning (SFT a base model with chat templates)
- Policy gradient methods (PPO, REINFORCE, GRPO, RLOO, etc.)
- Reward model training (preference RM, ORM, PRM)
- Direct alignment methods (DPO and variants)
- Rejection sampling (best-of-N)
- On-policy distillation (SDPO)

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
@book{rlhf2026lambert,
  author       = {Nathan Lambert},
  title        = {Reinforcement Learning from Human Feedback},
  year         = {2026},
  publisher    = {Online},
  url          = {https://rlhfbook.com},
}
```

## License

- `book/chapters`: [CC-BY-NC-SA-4.0](LICENSE-CHAPTERS)
- everything else (`code/`, `diagrams/`, `scripts/`, etc.): [MIT](LICENSE-CODE)
- note that some images in `book/images/` are unlicensed photos or screenshots. 

## Contributors

Where I get the credit as the sole "author" and creator of this project, I've been super lucky to have many contributions from early readers. These have massively accelerated the editing progress and flat-out added meaningful content to the book. I'm happy to send substantive contributors free copies of the book and expect the internet goodwill to pay them back in unexpected ways.

See all [contributors](https://github.com/natolambert/rlhf-book/graphs/contributors).

Note: *because I made the mistake of associating my commits with my Ai2 email, which I no longer have access to, the commit history lost most of my tracking, RIP!*

### Translations

Readers maintain unofficial translations of the book in their own repositories. 
These are community projects — independent of the official print editions and their professional translations — released under the same CC-BY-NC-SA license with attribution back to this book:

- 简体中文 (Simplified Chinese): [jweihe/RLHF-book-Chinese](https://github.com/jweihe/RLHF-book-Chinese)

To add yours: keep it in your own repo (translations are not merged here), follow the license terms above, label it clearly as a community translation, then open a PR adding it to this list and to the homepage Ecosystem section (`book/templates/html.html`).

### AI Use Policy

I wanted to clearly document how I used AI to aid in the editing and creation of this book (and my expectations for contributors).
This book was written at an interesting time, when AI models transitioned from useful to essential as tools for knowledge work.

The core of this book was written when language models felt borderline useless for non-fiction writing; this is roughly the first 10 chapters of the book -- it was my personal notes as I learned post-training.
The first draft was almost entirely manual (typos and all are in the git history).
Much of the other chapters and the appendices were adapted directly from content on [interconnects.ai](https://interconnects.ai/), my personal newsletter.
This writing is very high-voice and uses the lightest of AI editing to maintain the communication of intuitions.
The less math and code in a chapter, the less I used AI.

Through the editing, the default workflow I used was passing a list of suggested edits from a human editor to Claude Code with the context, and asking it to go one-by-one to apply various edits. 
In this format, I'd read the context and write a fix.
In a case where the edit was a simple typo or blatant error or just a low number of words, Claude could directly make this edit.
More complex language edits were crafted by me, normally with me re-writing various sentences and additions.
I also often just write in Cursor and then ask Claude to handle GitHub for me.

If you follow my writing closely, the difference between this book and Interconnects is that I let AI agents apply edits I suggested for me in this repo, whereas on my blog I make a point of doing all of that work manually. 
This is largely a function of scale and complexity.

For the more math-heavy chapters, the models are unbelievably useful at manipulating LaTeX equations and basic code snippets.
These sections are more direct outputs from the AI models, as me writing the LaTeX manually would take substantial time.
Then, I would review the math and code an additional time manually and with the check of GPT-Pro models.

AI models were used much more extensively in `diagrams/` and `code/`, where I viewed these as a form of play with the latest models, around the substantial content of the book.

The physical edition of the book went through additional, substantial copy editing that transitioned the voice to be more of a standard style for a technical textbook. 

All new additions to the book should obviously be written by humans first, and then can be edited with AI, as this reflects my workflow above.
The presence of obvious AI-written content in a PR to GitHub will almost certainly result in it not being included.
