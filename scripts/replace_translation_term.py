#!/usr/bin/env python3
"""Replace a Chinese term across translated sources, skipping fenced code."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_CONTENT_GLOBS = [
    "book-zh/chapters/*.md",
    "book-zh/metadata.yml",
]

TERM_FILE_GLOBS = [
    "TRANSLATION_GLOSSARY.zh.md",
    "translation/TERMS.zh.tsv",
]


def resolve_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path().glob(pattern))
        if matches:
            paths.extend(path for path in matches if path.is_file())
        else:
            path = Path(pattern)
            if path.is_file():
                paths.append(path)
    return sorted(dict.fromkeys(paths))


def replace_outside_fences(text: str, old: str, new: str) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    in_fence = False
    count = 0
    output: list[str] = []

    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            output.append(line)
            continue
        if in_fence:
            output.append(line)
            continue
        count += line.count(old)
        output.append(line.replace(old, new))

    return "".join(output), count


def replace_plain(text: str, old: str, new: str) -> tuple[str, int]:
    return text.replace(old, new), text.count(old)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace a translated term exactly. Dry-run by default."
    )
    parser.add_argument("--old", required=True, help="Current Chinese term")
    parser.add_argument("--new", required=True, help="Replacement Chinese term")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes. Without this flag, only report matches.",
    )
    parser.add_argument(
        "--include-term-files",
        action="store_true",
        help="Also update TRANSLATION_GLOSSARY.zh.md and translation/TERMS.zh.tsv.",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        help="Optional explicit file paths or glob patterns.",
    )
    parser.add_argument(
        "--no-skip-fences",
        action="store_true",
        help="Replace inside Markdown fenced code blocks too.",
    )
    args = parser.parse_args()

    if args.old == args.new:
        raise SystemExit("--old and --new are identical; nothing to do.")

    patterns = args.paths or list(DEFAULT_CONTENT_GLOBS)
    if args.include_term_files:
        patterns += TERM_FILE_GLOBS
    paths = resolve_paths(patterns)
    if not paths:
        raise SystemExit("No files matched.")

    total = 0
    changed_files = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        skip_fences = path.suffix == ".md" and not args.no_skip_fences
        if skip_fences:
            updated, count = replace_outside_fences(text, args.old, args.new)
        else:
            updated, count = replace_plain(text, args.old, args.new)
        if count == 0:
            continue
        total += count
        changed_files += 1
        action = "update" if args.apply else "would update"
        print(f"{action}: {path} ({count} occurrence{'s' if count != 1 else ''})")
        if args.apply:
            path.write_text(updated, encoding="utf-8")

    mode = "APPLIED" if args.apply else "DRY RUN"
    print(f"{mode}: {total} replacements across {changed_files} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
