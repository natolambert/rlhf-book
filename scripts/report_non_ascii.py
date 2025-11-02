#!/usr/bin/env python3
"""
Report any residual non-ASCII characters in a text file.
Designed to match the diagnostics we emit in the arXiv latex build target.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        raise SystemExit(f"Usage: {argv[0]} <file-path>")

    path = Path(argv[1])
    suspect_lines = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for idx, line in enumerate(handle, start=1):
            if any(ord(ch) > 127 for ch in line):
                suspect_lines.append(f"{idx}:{line.rstrip()}")
                if len(suspect_lines) >= 10:
                    break

    if suspect_lines:
        print(f"[WARN] Non-ASCII bytes still present in {path}:")
        print("\n".join(suspect_lines))
    else:
        print(f"[INFO] All bytes ASCII-safe after post-processing.")


if __name__ == "__main__":
    main(sys.argv)
