#!/usr/bin/env python3
"""Generate llms.txt and llms-full.txt for the RLHF Book static site."""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path


SITE_URL = "https://rlhfbook.com"
PUBLIC_EXCLUDES = {"README.md", "appendix-00-references.md"}


@dataclass(frozen=True)
class Chapter:
    path: Path
    source: str
    title: str
    description: str
    url: str
    body: str


def strip_wrapping_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value[1:-1]
        if isinstance(parsed, str):
            return parsed
    return value


def quoted_frontmatter_value(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    text = re.sub(r"\A<!--.*?-->\s*", "", text, flags=re.S)
    if not text.startswith("---\n"):
        return {}, text

    _, frontmatter, body = text.split("---", 2)
    meta = {}
    for line in frontmatter.splitlines():
        if ":" not in line or line.startswith((" ", "\t", "-")):
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = strip_wrapping_quotes(value)
    return meta, body.lstrip()


def public_chapters(root: Path) -> list[Chapter]:
    chapters = []
    for path in sorted((root / "book" / "chapters").glob("*.md")):
        if path.name in PUBLIC_EXCLUDES:
            continue
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        title = meta.get("page-title") or meta.get("title") or path.stem
        description = meta.get("meta-description", "")
        chapters.append(
            Chapter(
                path=path,
                source=path.relative_to(root).as_posix(),
                title=title,
                description=description,
                url=f"{SITE_URL}/c/{path.stem}",
                body=body.strip(),
            )
        )
    return chapters


def llms_txt(chapters: list[Chapter]) -> str:
    chapter_links = "\n".join(
        f"- [{chapter.title}]({chapter.url}): {chapter.description}"
        for chapter in chapters
    )

    return f"""# RLHF Book

> A free online book and course on reinforcement learning from human feedback (RLHF), reward models, preference tuning, RLVR, and post-training language models.

The RLHF Book by Nathan Lambert explains how modern language models are post-trained, from instruction tuning and preference data to reward modeling, policy optimization, direct alignment, reasoning training, evaluation, and product behavior.

Use the chapter links for canonical web pages. Use `llms-full.txt` for a single concatenated Markdown context file generated from the public chapter sources.

## Core Resources

- [Full-text LLM context]({SITE_URL}/llms-full.txt): Concatenated Markdown for all public book chapters, excluding the references build utility page.
- [Book homepage]({SITE_URL}/): The canonical web version of the RLHF Book.
- [Course]({SITE_URL}/course): Free lectures and course material on RLHF and post-training.
- [Model library]({SITE_URL}/library): Model completion comparisons across supervised finetuning, RLHF, DPO, and related post-training stages.
- [RL cheatsheet]({SITE_URL}/rl-cheatsheet): One-page reference for PPO, GRPO, RLOO, REINFORCE, DPO, and related RLHF methods.

## Chapters

{chapter_links}

## Optional

- [PDF]({SITE_URL}/book.pdf): Printable book build.
- [EPUB]({SITE_URL}/book.epub): Ebook build.
- [GitHub source](https://github.com/natolambert/rlhf-book): Source repository for the book and course.
- [ArXiv paper](https://arxiv.org/abs/2504.12501): Paper version of the RLHF Book.
- [Manning book page](https://www.manning.com/books/the-rlhf-book): Publisher page for The RLHF Book.
"""


def llms_full_txt(chapters: list[Chapter]) -> str:
    parts = [
        "# RLHF Book Full Text",
        "",
        "> Concatenated Markdown source for the public RLHF Book chapters.",
        "",
        f"Canonical site: {SITE_URL}/",
        f"Chapter count: {len(chapters)}",
        "",
        "The references build utility page is excluded. Citations use the book source citation keys.",
    ]

    for chapter in chapters:
        parts.extend(
            [
                "",
                "---",
                f"title: {quoted_frontmatter_value(chapter.title)}",
                f"url: {quoted_frontmatter_value(chapter.url)}",
                f"source: {quoted_frontmatter_value(chapter.source)}",
                "---",
                "",
                chapter.body,
            ]
        )
    return "\n".join(parts).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for generated files",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chapters = public_chapters(root)
    (output_dir / "llms.txt").write_text(llms_txt(chapters), encoding="utf-8")
    (output_dir / "llms-full.txt").write_text(
        llms_full_txt(chapters),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
