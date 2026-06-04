#!/usr/bin/env python3
"""Generate sitemap.xml for the RLHF Book static site."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


SITE_URL = "https://rlhfbook.com"


def source_lastmod(root: Path, paths: list[Path]) -> str:
    sources = [path for path in paths if path.exists()]
    if not sources:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )

    try:
        relative_paths = [str(path.relative_to(root)) for path in sources]
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", *relative_paths],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        timestamp = result.stdout.strip().splitlines()
        if timestamp:
            return timestamp[0]
    except (subprocess.SubprocessError, ValueError):
        pass

    latest_mtime = max(path.stat().st_mtime for path in sources)
    return datetime.fromtimestamp(latest_mtime, timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def url_entry(location: str, priority: str, lastmod: str) -> str:
    lines = [
        "  <url>",
        f"    <loc>{escape(location)}</loc>",
        f"    <lastmod>{escape(lastmod)}</lastmod>",
        f"    <priority>{priority}</priority>",
        "  </url>",
    ]
    return "\n".join(lines)


def chapter_sources(root: Path, *, public_only: bool) -> list[Path]:
    chapters = []
    for path in sorted((root / "book" / "chapters").glob("*.md")):
        if path.name == "README.md":
            continue
        if public_only and path.name == "appendix-00-references.md":
            continue
        chapters.append(path)
    return chapters


def chapter_urls(root: Path) -> list[tuple[str, str, str]]:
    chapters = []
    for path in chapter_sources(root, public_only=True):
        chapters.append(
            (f"{SITE_URL}/c/{path.stem}", "0.8", source_lastmod(root, [path]))
        )
    return chapters


def teaching_urls(root: Path) -> list[tuple[str, str, str]]:
    url_sources: dict[str, tuple[str, list[Path]]] = {}
    for path in sorted((root / "teach").glob("*/talk.md")):
        url_sources.setdefault(
            f"{SITE_URL}/teach/{path.parent.name}/", ("0.5", [])
        )[1].append(path)
    for path in sorted((root / "teach").glob("*/slides.md")):
        url_sources.setdefault(
            f"{SITE_URL}/teach/{path.parent.name}/", ("0.5", [])
        )[1].append(path)
    for path in sorted((root / "teach" / "course").glob("*.md")):
        url_sources[f"{SITE_URL}/teach/course/{path.stem}/"] = ("0.6", [path])
    return sorted(
        (location, priority, source_lastmod(root, sources))
        for location, (priority, sources) in url_sources.items()
    )


def static_urls(root: Path) -> list[tuple[str, str, str]]:
    metadata = root / "book" / "metadata.yml"
    llms_sources = [
        root / "book" / "scripts" / "generate_llms.py",
        metadata,
        *chapter_sources(root, public_only=True),
    ]
    book_sources = [
        metadata,
        root / "book" / "templates" / "html.html",
        *chapter_sources(root, public_only=False),
    ]
    download_sources = [metadata, *chapter_sources(root, public_only=False)]
    library_sources = [
        root / "book" / "templates" / "library.html",
        root / "book" / "data" / "library.json",
    ]
    return [
        (f"{SITE_URL}/", "1.0", source_lastmod(root, book_sources)),
        (f"{SITE_URL}/llms.txt", "0.8", source_lastmod(root, llms_sources)),
        (f"{SITE_URL}/llms-full.txt", "0.8", source_lastmod(root, llms_sources)),
        (
            f"{SITE_URL}/course",
            "0.9",
            source_lastmod(root, [root / "book" / "templates" / "course.html"]),
        ),
        (
            f"{SITE_URL}/rl-cheatsheet",
            "0.7",
            source_lastmod(root, [root / "book" / "rl-cheatsheet" / "index.html"]),
        ),
        (f"{SITE_URL}/library", "0.7", source_lastmod(root, library_sources)),
        (f"{SITE_URL}/book.pdf", "0.6", source_lastmod(root, download_sources)),
        (f"{SITE_URL}/book.epub", "0.4", source_lastmod(root, download_sources)),
        (
            f"{SITE_URL}/book.kindle.epub",
            "0.4",
            source_lastmod(root, download_sources),
        ),
    ]


def build_sitemap(root: Path) -> str:
    urls = static_urls(root)
    urls.extend(chapter_urls(root))
    urls.extend(teaching_urls(root))

    entries = "\n".join(
        url_entry(location, priority, lastmod)
        for location, priority, lastmod in urls
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{entries}\n"
        "</urlset>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Path to write sitemap.xml")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_sitemap(root), encoding="utf-8")


if __name__ == "__main__":
    main()
