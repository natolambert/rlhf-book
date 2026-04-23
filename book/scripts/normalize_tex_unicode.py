#!/usr/bin/env python3
"""
Post-process Pandoc's LaTeX output so it stays pdfLaTeX compatible.
Maps Unicode punctuation and accented letters to ASCII/TeX escape sequences.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPLACEMENTS: list[tuple[str, str]] = [
    ("\u2060", ""),  # WORD JOINER
    ("\u03c4", r"\tau"),
    ("\u2018", "'"),
    ("\u2019", "'"),
    ("\u201c", '"'),
    ("\u201d", '"'),
    ("\u2010", "-"),
    ("\u2011", "-"),
    ("\u2012", "-"),
    ("\u2013", "-"),
    ("\u2014", "-"),
    ("\u2015", "-"),
    ("\u2212", "-"),
    ("\u00a0", " "),
    ("\u202f", " "),
    ("\u00dc", r"\"{U}"),
    ("\u00fc", r"\"{u}"),
    ("\u00e1", r"\'{a}"),
    ("\u00e9", r"\'{e}"),
    ("\u00f6", r"\"{o}"),
    ("\u2026", "..."),
    ("\u00a9", r"\textcopyright{}"),
    ("\u2192", r"$\to$"),
]


def normalise(tex_path: Path) -> None:
    text = tex_path.read_text()
    for src, dst in REPLACEMENTS:
        text = text.replace(src, dst)
    tex_path.write_text(text)


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        raise SystemExit(f"Usage: {argv[0]} <tex-path>")
    normalise(Path(argv[1]))


if __name__ == "__main__":
    main(sys.argv)
