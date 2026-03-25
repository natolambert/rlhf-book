# Teach / Slides — Claude Code Context

## Colloquium Slide Framework

Slides are built with [colloquium](https://github.com/natolambert/colloquium) from Markdown source files.

- Each talk/lecture lives in its own directory (e.g. `SALA-2026/`, `course/`)
- Source files are `talk.md` or `slides.md` (for talks) or `lec*.md` (for course lectures)
- Assets go in an `assets/` subdirectory per talk/course
- References are shared via `bibliography: ../SALA-2026/refs.bib` (or similar relative path)

## Animation via Slide Repeats

Colloquium does not have built-in animation/fragment support. To simulate animations (progressive reveals), **duplicate the slide** with additional content on each copy. For example, a slide shown first with 4 bullets, then repeated with 2 more bullets below, is intentional — do NOT merge these into one slide.

**Math derivation unrolls**: For step-by-step derivations, use the same pattern — repeat the slide with one additional derivation step each time. Keep the title and earlier steps identical so the audience sees each new line appear. This is especially useful for multi-step algebra (e.g. Bradley-Terry → loss function). The result looks like repetition in the source, but each copy is a separate slide that reveals one more step.

**Never skip steps in derivations.** Every algebraic manipulation must be shown explicitly — if a term cancels, show it cancelling; if an expression is rewritten, show the intermediate form. Assume the audience cannot fill in gaps. For example, when dividing numerator and denominator by the same term, first show the division applied, then show the numerator simplifying to 1, then show the denominator simplifying. Each of these can be a separate slide.

## Colloquium Directives

Key directives (HTML comments before or after the heading):
- `<!-- columns: 45/55 -->` — side-by-side columns
- `<!-- rows: 48/52 -->` — top/bottom rows, separated by `===`
- `<!-- row-columns: 50/50 -->` — columns within a row
- `<!-- align: center -->` — center entire slide (slide-scoped, affects everything)
- `<!-- cite-right: key -->` / `<!-- cite-left: key -->` — citation placement
- `<!-- title: center -->` — center the title
- `<!-- layout: section-break -->` — section break slide
- `<!-- valign: center -->` — vertically center content
- `<!-- img-align: center -->` — center images

## Heading Parsing

Titles must be bare `## Heading` at line start. Wrapping in `<div>` breaks colloquium's heading extraction.

## Title Case

Use sentence case for all slide titles and section-break titles.

- Capitalize the first word of the title
- Capitalize the first word after a colon
- Keep acronyms and proper names capitalized (e.g. `RLHF`, `PPO`, `OpenAI`, `ChatGPT`)
- Do not use title case across the full heading

## Citation Style

Two citation modes — choose based on what is being cited:

**Inline citations** `[@key]` — use when a citation supports a **specific claim or named work** in the slide body.

- Put the citation immediately after the referenced work or phrase: `T5 [@raffel2020exploring]`, `FLAN [@wei2021finetuned]`
- If multiple named works are listed, cite each one separately rather than bundling them at the end of the sentence
- Prefer this style when the slide text says things like "X showed...", "Y introduced...", or lists specific papers/datasets

**Slide-level citations** `<!-- cite-right: key -->` / `<!-- cite-left: key -->` — use when the **entire slide** is about a project, paper, or idea, rather than citing a specific bullet point. Also use for image sourcing.

## Build

```bash
uv run colloquium build teach/SALA-2026/talk.md -o build/
uv run colloquium export teach/SALA-2026/talk.md -o slides.pdf
```

## Local Preview (Live Reload)

`colloquium serve` serves from the input file's parent directory. When slides reference assets via `../` paths (e.g. `../SALA-2026/assets/`), images break because the web server won't serve files above its root.

**Workaround**: Call `serve()` directly with `output_dir` set to `teach/` so the serve root is high enough to resolve all relative paths:

```bash
# From the repo root:
uv run python -c "from colloquium.serve import serve; serve('teach/course/lec1-chap1-3.md', port=8080, output_dir='teach')"
```

This gives you live rebuild on save + correct asset resolution. The URL will be `http://localhost:8080/<stem>.html`.

Note: `colloquium serve` CLI does not expose `--output-dir` yet — this is a known limitation.
