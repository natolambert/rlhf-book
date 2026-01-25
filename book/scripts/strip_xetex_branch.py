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

    # Handle OLD Pandoc format:
    # \ifnum 0\ifxetex 1\fi\ifluatex 1\fi=0 % if pdftex
    #   \usepackage[T1]{fontenc}
    #   ...
    # \else % if luatex or xetex
    #   \usepackage{unicode-math}
    #   ...
    # \fi
    # In this case, we want to KEEP the pdftex branch and remove \else...\fi
    old_format_pattern = (
        r'(\\ifnum 0\\ifxetex 1\\fi\\ifluatex 1\\fi=0 % if pdftex\n'
        r'.*?)'  # pdftex branch (keep this)
        r'\\else % if luatex or xetex\n'
        r'.*?'  # xetex/luatex branch (remove this)
        r'\\fi\n'
    )
    old_match = re.search(old_format_pattern, text, re.DOTALL)
    if old_match:
        # Keep only the pdftex branch, remove the conditional wrapper
        pdftex_branch = old_match.group(1)
        # Remove the \ifnum line, keep just the packages
        pdftex_content = re.sub(
            r'\\ifnum 0\\ifxetex 1\\fi\\ifluatex 1\\fi=0 % if pdftex\n',
            '% pdfLaTeX mode (XeTeX/LuaTeX branch removed for arXiv)\n',
            pdftex_branch
        )
        text = text[:old_match.start()] + pdftex_content + text[old_match.end():]
    else:
        # Handle NEW Pandoc format:
        # \ifnum 0 % if luatex or xetex
        #   \usepackage{unicode-math}
        #   ...
        # (no \fi, block ends at "% Use upquote")
        new_format_pattern = r'\\ifnum 0\s*%.*?if luatex or xetex.*?(?=\n% Use upquote)'
        replacement = r'''% XeTeX/LuaTeX setup removed for arXiv export
% pdfLaTeX encoding setup
\\usepackage[T1]{fontenc}
\\usepackage[utf8]{inputenc}
\\usepackage{textcomp} % provide euro and other symbols'''
        new_text = re.sub(new_format_pattern, replacement, text, flags=re.DOTALL)
        if text != new_text:
            text = new_text

    # Handle any remaining standalone \ifxetex ... \else ... \fi blocks
    # (but NOT the inline \ifxetex 1\fi pattern)
    # Look for \ifxetex followed by newline (not just a number)
    pattern = re.compile(r"\\ifxetex\s*\n(.*?)\\else\s*\n(.*?)\\fi", re.DOTALL)
    while True:
        match = pattern.search(text)
        if not match:
            break
        _, else_block = match.groups()
        text = text[: match.start()] + else_block + text[match.end():]

    # Remove any remaining \ifluatex blocks similarly
    pattern = re.compile(r"\\ifluatex\s*\n(.*?)\\else\s*\n(.*?)\\fi", re.DOTALL)
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
