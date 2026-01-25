#!/usr/bin/env python3
"""
Ensure the generated TeX file explicitly requests pdfLaTeX by adding \\pdfoutput=1.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_pdfoutput(tex_path: Path) -> None:
    text = tex_path.read_text()
    if "\\pdfoutput" in text:
        return
    tex_path.write_text("\\pdfoutput=1\n" + text)


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        raise SystemExit(f"Usage: {argv[0]} <tex-path>")
    ensure_pdfoutput(Path(argv[1]))


if __name__ == "__main__":
    main(sys.argv)
