#!/usr/bin/env python3
"""
Check if arxiv preprints in a bib file have been published in peer-reviewed venues.

Uses DBLP API (free, no auth required) to find published versions.
Generates a report of papers that may need updating.

Usage:
    python check_arxiv_publications.py [--bib-file path/to/bib.bib] [--batch-size 20] [--delay 0.5]
"""

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


@dataclass
class BibEntry:
    key: str
    title: str
    raw: str
    arxiv_id: str | None = None


@dataclass
class Publication:
    venue: str
    year: str
    title: str
    dblp_key: str
    bibtex_url: str | None = None


def normalize(s: str) -> str:
    """Normalize string for comparison."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def similarity(a: str, b: str) -> float:
    """Compute similarity ratio between two strings."""
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def extract_title(entry: str) -> str | None:
    """Extract title from bib entry."""
    match = re.search(r"title\s*=\s*[{\"](.+?)[}\"]", entry, re.DOTALL | re.IGNORECASE)
    if match:
        title = match.group(1).replace("\n", " ").strip()
        # Clean LaTeX
        title = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", title)
        title = re.sub(r"[{}\\$]", "", title)
        return title.strip()
    return None


def extract_arxiv_id(entry: str) -> str | None:
    """Extract arxiv ID from bib entry."""
    patterns = [
        r"arXiv:(\d+\.\d+)",
        r"arxiv\.org/abs/(\d+\.\d+)",
        r"eprint\s*=\s*[{\"](\d+\.\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, entry)
        if match:
            return match.group(1)
    return None


def is_arxiv_entry(entry: str) -> bool:
    """Check if entry is an arxiv preprint."""
    indicators = [
        r"arXiv preprint",
        r"arXiv:",
        r"arxiv\.org",
        r"journal\s*=\s*[{\"]arXiv",
        r"eprint\s*=",
    ]
    return any(re.search(p, entry, re.IGNORECASE) for p in indicators)


def parse_bib_file(bib_path: Path) -> list[BibEntry]:
    """Parse bib file and extract arxiv entries."""
    content = bib_path.read_text()
    pattern = re.compile(r"^(@\w+\{([^,]+),.*?^})", re.MULTILINE | re.DOTALL)

    entries = []
    for match in pattern.finditer(content):
        raw = match.group(1)
        key = match.group(2)

        if is_arxiv_entry(raw):
            title = extract_title(raw)
            if title:
                entries.append(
                    BibEntry(
                        key=key,
                        title=title,
                        raw=raw,
                        arxiv_id=extract_arxiv_id(raw),
                    )
                )
    return entries


def search_dblp(title: str, timeout: int = 10, max_retries: int = 3) -> list[dict]:
    """Search DBLP for a paper by title with exponential backoff."""
    clean = re.sub(r"[{}\\$:]", " ", title).strip()
    params = urllib.parse.urlencode({"q": clean[:100], "format": "json", "h": 5})
    url = f"https://dblp.org/search/publ/api?{params}"

    for attempt in range(max_retries):
        req = urllib.request.Request(url, headers={"User-Agent": "BibUpdater/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read().decode())
                return data.get("result", {}).get("hits", {}).get("hit", [])
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                wait = 2 ** (attempt + 2)  # 4, 8, 16 seconds
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    DBLP error: {e}")
                return []
        except Exception as e:
            print(f"    DBLP error: {e}")
            return []
    return []


def find_published_version(entry: BibEntry, hits: list[dict]) -> Publication | None:
    """Find best matching published (non-arxiv) version."""
    ARXIV_VENUES = {"CoRR", "IACR Cryptol. ePrint Arch.", ""}

    matches = []
    for hit in hits:
        info = hit.get("info", {})
        hit_title = info.get("title", "")
        sim = similarity(entry.title, hit_title)
        venue = info.get("venue", "")
        is_published = venue not in ARXIV_VENUES

        if sim > 0.7:  # Require 70% title similarity
            matches.append(
                (
                    sim,
                    is_published,
                    Publication(
                        venue=venue,
                        year=info.get("year", ""),
                        title=hit_title,
                        dblp_key=info.get("key", ""),
                        bibtex_url=info.get("url", "") + ".bib" if info.get("url") else None,
                    ),
                )
            )

    if not matches:
        return None

    # Sort by: published first, then by similarity
    matches.sort(key=lambda x: (-x[1], -x[0]))

    # Return best match only if it's published
    best = matches[0]
    if best[1]:  # is_published
        return best[2]
    return None


def fetch_bibtex(url: str, timeout: int = 10) -> str | None:
    """Fetch bibtex from DBLP."""
    req = urllib.request.Request(url, headers={"User-Agent": "BibUpdater/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read().decode()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bib-file",
        type=Path,
        default=Path("chapters/bib.bib"),
        help="Path to bib file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Process only N entries (0 = all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API requests in seconds",
    )
    parser.add_argument(
        "--fetch-bibtex",
        action="store_true",
        help="Fetch updated bibtex from DBLP for published papers",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("arxiv_check_results.json"),
        help="Write results to JSON file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip already-checked entries)",
    )
    args = parser.parse_args()

    # Parse bib file
    print(f"Parsing {args.bib_file}...")
    entries = parse_bib_file(args.bib_file)
    print(f"Found {len(entries)} arxiv entries\n")

    if args.batch_size > 0:
        entries = entries[: args.batch_size]
        print(f"Processing first {args.batch_size} entries\n")

    # Load existing results if resuming
    results = {
        "published": [],  # Have a published version
        "arxiv_only": [],  # Still arxiv only
        "not_found": [],  # No DBLP match
    }
    checked_keys = set()

    if args.resume and args.output.exists():
        try:
            existing = json.loads(args.output.read_text())
            results = existing
            for category in ["published", "arxiv_only", "not_found"]:
                for item in existing.get(category, []):
                    checked_keys.add(item["key"])
            print(f"Resuming: {len(checked_keys)} entries already checked\n")
        except Exception as e:
            print(f"Could not load existing results: {e}\n")

    for i, entry in enumerate(entries, 1):
        # Skip if already checked (resume mode)
        if entry.key in checked_keys:
            continue

        print(f"[{i}/{len(entries)}] {entry.key}")
        print(f"  Title: {entry.title[:60]}...")

        hits = search_dblp(entry.title)
        time.sleep(args.delay)

        pub = find_published_version(entry, hits)

        if pub:
            print(f"  ✅ PUBLISHED: {pub.venue} ({pub.year})")
            result = {
                "key": entry.key,
                "original_title": entry.title,
                "venue": pub.venue,
                "year": pub.year,
                "dblp_title": pub.title,
                "dblp_key": pub.dblp_key,
            }

            if args.fetch_bibtex and pub.bibtex_url:
                bibtex = fetch_bibtex(pub.bibtex_url)
                if bibtex:
                    result["suggested_bibtex"] = bibtex
                time.sleep(args.delay)

            results["published"].append(result)
        elif hits:
            # Found matches but all are arxiv
            print(f"  📄 Still arxiv only")
            results["arxiv_only"].append({"key": entry.key, "title": entry.title})
        else:
            print(f"  ❓ No DBLP match")
            results["not_found"].append({"key": entry.key, "title": entry.title})

        # Save progress after each entry
        args.output.write_text(json.dumps(results, indent=2))
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Published (need update): {len(results['published'])}")
    print(f"  Arxiv only:              {len(results['arxiv_only'])}")
    print(f"  Not found in DBLP:       {len(results['not_found'])}")

    if results["published"]:
        print(f"\n📋 Papers with published versions:\n")
        for p in results["published"]:
            print(f"  {p['key']}")
            print(f"    → {p['venue']} ({p['year']})")

    # Save results
    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
