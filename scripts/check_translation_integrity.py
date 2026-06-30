#!/usr/bin/env python3
"""Check structural invariants between English and Chinese book chapters."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


PATTERNS = {
    "code_fences": re.compile(r"^```", re.MULTILINE),
    "images": re.compile(r"!\[[^\]]*\]\([^)]+\)"),
    "citations": re.compile(r"@[A-Za-z0-9][A-Za-z0-9:_-]*[A-Za-z0-9_-]"),
    "display_math": re.compile(r"^\$\$", re.MULTILINE),
}


def count(pattern: re.Pattern[str], text: str) -> int:
    return len(pattern.findall(text))


def check_pair(source: Path, translated: Path) -> list[str]:
    source_text = source.read_text(encoding="utf-8")
    translated_text = translated.read_text(encoding="utf-8")
    issues: list[str] = []

    for name, pattern in PATTERNS.items():
        source_count = count(pattern, source_text)
        translated_count = count(pattern, translated_text)
        if source_count != translated_count:
            issues.append(
                f"{translated}: {name} count mismatch "
                f"(source={source_count}, translated={translated_count})"
            )

    source_keys = sorted(set(PATTERNS["citations"].findall(source_text)))
    translated_keys = sorted(set(PATTERNS["citations"].findall(translated_text)))
    missing = sorted(set(source_keys) - set(translated_keys))
    extra = sorted(set(translated_keys) - set(source_keys))
    if missing:
        issues.append(f"{translated}: missing citations: {', '.join(missing)}")
    if extra:
        issues.append(f"{translated}: extra citations: {', '.join(extra)}")

    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="book/chapters")
    parser.add_argument("--translated-dir", default="book-zh/chapters")
    parser.add_argument("files", nargs="*", help="Optional chapter filenames to check")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    translated_dir = Path(args.translated_dir)
    filenames = args.files or [path.name for path in sorted(source_dir.glob("*.md"))]

    all_issues: list[str] = []
    for filename in filenames:
        if filename == "README.md":
            continue
        source = source_dir / filename
        translated = translated_dir / filename
        if not translated.exists():
            all_issues.append(f"{translated}: missing translated file")
            continue
        all_issues.extend(check_pair(source, translated))

    if all_issues:
        print("\n".join(all_issues))
        return 1

    print(f"OK: checked {len(filenames)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
