#!/usr/bin/env python3
"""
Copy only the image files referenced by a generated LaTeX source file.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

IMAGE_SUFFIXES = {".eps", ".jpeg", ".jpg", ".pdf", ".png"}
INCLUDEGRAPHICS = r"\includegraphics"


def _skip_ws(text: str, idx: int) -> int:
    while idx < len(text) and text[idx].isspace():
        idx += 1
    return idx


def _skip_optional_arg(text: str, idx: int) -> int:
    bracket_depth = 1
    brace_depth = 0
    idx += 1

    while idx < len(text):
        ch = text[idx]
        if ch == "\\":
            idx += 2
            continue
        if ch == "{":
            brace_depth += 1
        elif ch == "}" and brace_depth:
            brace_depth -= 1
        elif brace_depth == 0 and ch == "[":
            bracket_depth += 1
        elif brace_depth == 0 and ch == "]":
            bracket_depth -= 1
            if bracket_depth == 0:
                return idx + 1
        idx += 1

    raise ValueError("Unterminated optional argument in \\includegraphics")


def _read_required_arg(text: str, idx: int) -> tuple[str, int]:
    if idx >= len(text) or text[idx] != "{":
        raise ValueError("Expected required argument after \\includegraphics")

    brace_depth = 1
    start = idx + 1
    idx += 1

    while idx < len(text):
        ch = text[idx]
        if ch == "\\":
            idx += 2
            continue
        if ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0:
                return text[start:idx].strip(), idx + 1
        idx += 1

    raise ValueError("Unterminated required argument in \\includegraphics")


def referenced_images(tex_text: str) -> list[str]:
    paths: list[str] = []
    idx = 0

    while True:
        idx = tex_text.find(INCLUDEGRAPHICS, idx)
        if idx == -1:
            return paths

        idx += len(INCLUDEGRAPHICS)
        idx = _skip_ws(tex_text, idx)

        if idx < len(tex_text) and tex_text[idx] == "[":
            idx = _skip_optional_arg(tex_text, idx)
            idx = _skip_ws(tex_text, idx)

        if idx < len(tex_text) and tex_text[idx] == "{":
            image_path, idx = _read_required_arg(tex_text, idx)
            if image_path:
                paths.append(image_path)


def source_by_basename(images_root: Path) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}

    for path in images_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.name in sources:
            duplicates.setdefault(path.name, [sources[path.name]]).append(path)
        else:
            sources[path.name] = path

    if duplicates:
        names = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate image basenames would make arXiv flattening ambiguous: {names}")

    return sources


def copy_images(tex_path: Path, images_root: Path, dest_dir: Path) -> None:
    tex_text = tex_path.read_text()
    sources = source_by_basename(images_root)
    ordered_names = list(dict.fromkeys(Path(path).name for path in referenced_images(tex_text)))

    if images_root.resolve() == dest_dir.resolve():
        raise ValueError("Refusing to clean image source directory")

    dest_dir.mkdir(parents=True, exist_ok=True)
    for path in dest_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            path.unlink()

    missing = [name for name in ordered_names if name not in sources]
    if missing:
        raise FileNotFoundError(f"Referenced image(s) missing from {images_root}: {', '.join(missing)}")

    for name in ordered_names:
        shutil.copy2(sources[name], dest_dir / name)

    print(f"[INFO] Copied {len(ordered_names)} referenced image(s) into {dest_dir}.")


def main(argv: list[str]) -> None:
    if len(argv) != 4:
        raise SystemExit(f"Usage: {argv[0]} <tex-path> <images-root> <destination-dir>")
    copy_images(Path(argv[1]), Path(argv[2]), Path(argv[3]))


if __name__ == "__main__":
    main(sys.argv)
