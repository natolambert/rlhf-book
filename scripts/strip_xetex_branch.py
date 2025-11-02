#!/usr/bin/env python3
"""
Utility to prune the XeTeX/LuaTeX-only preamble section that Pandoc emits,
so that the arXiv-oriented TeX export remains pdfLaTeX compatible.
"""

from __future__ import annotations

import sys
from pathlib import Path
import re


def strip_unicode_branch(tex_path: Path) -> None:
    text = tex_path.read_text()

    # Remove the ifxetex,ifluatex package loading
    text = re.sub(r'\\usepackage\{ifxetex,ifluatex\}', '', text)

    # Handle the new Pandoc format: \ifnum 0 % if luatex or xetex
    # This block extends from "\ifnum 0" to just before "% Use upquote"
    pattern = r'\\ifnum 0\s*%.*?if luatex or xetex.*?(?=\n% Use upquote)'
    replacement = r'% XeTeX/LuaTeX setup removed for arXiv export'
    text = re.sub(pattern, replacement, text, flags=re.DOTALL)

    # Also collapse any explicit \ifxetex ... \else ... \fi branches
    # to keep only the pdfTeX-friendly \else clause.
    pattern = re.compile(r"\\ifxetex\s*?(.*?)\\else\s*?(.*?)\\fi", re.DOTALL)
    while True:
        match = pattern.search(text)
        if not match:
            break
        _, else_block = match.groups()
        text = text[: match.start()] + else_block + text[match.end():]

    # Remove any remaining \ifluatex blocks similarly
    pattern = re.compile(r"\\ifluatex\s*?(.*?)\\else\s*?(.*?)\\fi", re.DOTALL)
    while True:
        match = pattern.search(text)
        if not match:
            break
        _, else_block = match.groups()
        text = text[: match.start()] + else_block + text[match.end():]

    tex_path.write_text(text)


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        raise SystemExit(f"Usage: {argv[0]} <tex-path>")
    strip_unicode_branch(Path(argv[1]))


if __name__ == "__main__":
    main(sys.argv)
