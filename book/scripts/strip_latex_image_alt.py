#!/usr/bin/env python3
"""
Remove alt= entries from LaTeX \\includegraphics options for arXiv packaging.
"""

from __future__ import annotations

import sys
from pathlib import Path

INCLUDEGRAPHICS = r"\includegraphics"


def _skip_ws(text: str, idx: int) -> int:
    while idx < len(text) and text[idx].isspace():
        idx += 1
    return idx


def _read_balanced(text: str, idx: int, opener: str, closer: str) -> tuple[str, int]:
    if idx >= len(text) or text[idx] != opener:
        raise ValueError(f"Expected {opener!r}")

    depth = 1
    start = idx + 1
    idx += 1

    while idx < len(text):
        ch = text[idx]
        if ch == "\\":
            idx += 2
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:idx], idx + 1
        idx += 1

    raise ValueError(f"Unterminated {opener!r} argument")


def _split_options(options: str) -> list[str]:
    parts: list[str] = []
    start = 0
    brace_depth = 0
    bracket_depth = 0
    idx = 0

    while idx < len(options):
        ch = options[idx]
        if ch == "\\":
            idx += 2
            continue
        if ch == "{":
            brace_depth += 1
        elif ch == "}" and brace_depth:
            brace_depth -= 1
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]" and bracket_depth:
            bracket_depth -= 1
        elif ch == "," and brace_depth == 0 and bracket_depth == 0:
            parts.append(options[start:idx].strip())
            start = idx + 1
        idx += 1

    parts.append(options[start:].strip())
    return [part for part in parts if part]


def _option_key(option: str) -> str:
    key = option.split("=", 1)[0]
    return key.strip().lower()


def strip_alt_options(text: str) -> str:
    pieces: list[str] = []
    idx = 0

    while True:
        command_idx = text.find(INCLUDEGRAPHICS, idx)
        if command_idx == -1:
            pieces.append(text[idx:])
            return "".join(pieces)

        pieces.append(text[idx:command_idx])
        pieces.append(INCLUDEGRAPHICS)
        idx = command_idx + len(INCLUDEGRAPHICS)

        ws_start = idx
        idx = _skip_ws(text, idx)
        pieces.append(text[ws_start:idx])

        if idx >= len(text) or text[idx] != "[":
            continue

        options, idx = _read_balanced(text, idx, "[", "]")
        kept_options = [option for option in _split_options(options) if _option_key(option) != "alt"]
        if kept_options:
            pieces.append("[")
            pieces.append(",".join(kept_options))
            pieces.append("]")


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        raise SystemExit(f"Usage: {argv[0]} <tex-path>")

    tex_path = Path(argv[1])
    tex_path.write_text(strip_alt_options(tex_path.read_text()))


if __name__ == "__main__":
    main(sys.argv)
