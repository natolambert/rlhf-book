#!/usr/bin/env python3
"""Check and optionally normalize the book bibliography.

The bibliography is grouped by the chapter where each key first appears. By
default this script only reports duplicate keys, missing citations, unused
entries, and order drift. Pass --fix to rewrite bib.bib by removing unused
entries and sorting the remaining entries by first citation appearance.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


BIB_ENTRY_START_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,")
CITATION_RE = re.compile(
    r"(?<![\w:-])-?@(?:\{([^}\n]+)\}|"
    r"([A-Za-z0-9_](?:[A-Za-z0-9_]|[:.#$%&\-+?<>~/](?=[A-Za-z0-9_]))*))"
)
FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
SECTION_HEADER_RE = re.compile(
    r"^### First appearing in (chapter \d+|appendix [A-Z]) .+ ###$"
)
SECTION_END_RE = re.compile(r"^### End (chapter \d+|appendix [A-Z]) ###$")
CROSS_REF_PREFIXES = (
    "chap:",
    "def:",
    "eq:",
    "fig:",
    "lst:",
    "sec:",
    "tbl:",
    "thm:",
)


@dataclass(frozen=True)
class BibEntry:
    key: str
    line: int
    text: str


@dataclass(frozen=True)
class Section:
    id: str
    header: str
    end: str
    path: Path


@dataclass(frozen=True)
class CitationLocation:
    key: str
    section_id: str
    chapter_path: Path
    line: int
    order: int


@dataclass
class BibReport:
    entries: list[BibEntry]
    duplicates: dict[str, int]
    duplicate_lines: dict[str, list[int]]
    citations: list[CitationLocation]
    first_locations: dict[str, CitationLocation]
    missing_keys: set[str]
    unused_keys: set[str]
    expected_order: list[str]
    current_used_order: list[str]

    @property
    def has_order_drift(self) -> bool:
        return self.expected_order != self.current_used_order

    @property
    def has_problems(self) -> bool:
        return bool(
            self.duplicates
            or self.missing_keys
            or self.unused_keys
            or self.has_order_drift
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bib-file",
        type=Path,
        default=Path("book/chapters/bib.bib"),
        help="Path to the BibTeX file to check.",
    )
    parser.add_argument(
        "--chapters-dir",
        type=Path,
        default=Path("book/chapters"),
        help="Directory containing book chapter markdown files.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Remove unused entries and sort bib.bib by first citation appearance.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with a non-zero status if any issues remain.",
    )
    return parser.parse_args()


def parse_bib_entries(content: str) -> list[BibEntry]:
    entries: list[BibEntry] = []
    for match in BIB_ENTRY_START_RE.finditer(content):
        start = match.start()
        brace_count = 0
        end = None
        for index in range(start, len(content)):
            char = content[index]
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = index + 1
                    break

        if end is None:
            raise ValueError(f"Could not find end of entry for key {match.group(1)}")

        line = content.count("\n", 0, start) + 1
        entries.append(BibEntry(key=match.group(1), line=line, text=content[start:end]))

    return entries


def extract_page_title(path: Path) -> str:
    content = path.read_text()
    frontmatter = re.search(r"^---\n(.*?)\n---", content, re.DOTALL | re.MULTILINE)
    if frontmatter:
        title = re.search(r'^page-title:\s*"?([^"\n]+)"?\s*$', frontmatter.group(1), re.MULTILINE)
        if title:
            value = title.group(1).strip()
            value = re.sub(r"^Appendix [A-Z]:\s*", "", value)
            return value

    heading = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if heading:
        return heading.group(1).strip()

    return path.stem.replace("-", " ").title()


def section_id_from_marker(marker: str) -> str:
    if marker.startswith("chapter "):
        return "chapter-" + marker.removeprefix("chapter ")
    return "appendix-" + marker.removeprefix("appendix ")


def existing_section_lines(content: str) -> tuple[dict[str, str], dict[str, str]]:
    headers: dict[str, str] = {}
    ends: dict[str, str] = {}
    for line in content.splitlines():
        header = SECTION_HEADER_RE.match(line)
        if header:
            headers[section_id_from_marker(header.group(1))] = line
            continue

        end = SECTION_END_RE.match(line)
        if end:
            ends[section_id_from_marker(end.group(1))] = line

    return headers, ends


def chapter_files(chapters_dir: Path) -> list[Path]:
    files = []
    for path in sorted(chapters_dir.glob("*.md")):
        if path.name in {"README.md", "appendix-00-references.md"}:
            continue
        files.append(path)
    return files


def section_for_path(
    path: Path,
    existing_headers: dict[str, str],
    existing_ends: dict[str, str],
) -> Section | None:
    chapter = re.match(r"^(\d+)-", path.name)
    if chapter:
        number = str(int(chapter.group(1)))
        section_id = f"chapter-{number}"
        title = extract_page_title(path)
        return Section(
            id=section_id,
            header=existing_headers.get(
                section_id, f"### First appearing in chapter {number} ({title}) ###"
            ),
            end=existing_ends.get(section_id, f"### End chapter {number} ###"),
            path=path,
        )

    appendix = re.match(r"^appendix-([a-z])-", path.name)
    if appendix:
        letter = appendix.group(1).upper()
        section_id = f"appendix-{letter}"
        title = extract_page_title(path)
        return Section(
            id=section_id,
            header=existing_headers.get(
                section_id, f"### First appearing in appendix {letter} ({title}) ###"
            ),
            end=existing_ends.get(section_id, f"### End appendix {letter} ###"),
            path=path,
        )

    return None


def book_sections(chapters_dir: Path, bib_content: str) -> list[Section]:
    existing_headers, existing_ends = existing_section_lines(bib_content)
    sections = [
        section_for_path(path, existing_headers, existing_ends)
        for path in chapter_files(chapters_dir)
    ]
    return [section for section in sections if section is not None]


def strip_fenced_code(content: str) -> str:
    return FENCED_CODE_RE.sub(lambda match: "\n" * match.group(0).count("\n"), content)


def extract_citations(content: str) -> list[tuple[str, int]]:
    citations: list[tuple[str, int]] = []
    content = strip_fenced_code(content)
    for match in CITATION_RE.finditer(content):
        key = match.group(1) or match.group(2)
        if key.startswith(CROSS_REF_PREFIXES):
            continue
        line = content.count("\n", 0, match.start()) + 1
        citations.append((key, line))
    return citations


def collect_citations(sections: list[Section]) -> list[CitationLocation]:
    locations: list[CitationLocation] = []
    order = 0
    for section in sections:
        content = section.path.read_text()
        for key, line in extract_citations(content):
            locations.append(
                CitationLocation(
                    key=key,
                    section_id=section.id,
                    chapter_path=section.path,
                    line=line,
                    order=order,
                )
            )
            order += 1
    return locations


def analyze_bib(bib_content: str, chapters_dir: Path) -> BibReport:
    entries = parse_bib_entries(bib_content)
    key_counts = Counter(entry.key for entry in entries)
    duplicates = {key: count for key, count in key_counts.items() if count > 1}

    duplicate_lines: dict[str, list[int]] = defaultdict(list)
    for entry in entries:
        if entry.key in duplicates:
            duplicate_lines[entry.key].append(entry.line)

    sections = book_sections(chapters_dir, bib_content)
    citations = collect_citations(sections)

    first_locations: dict[str, CitationLocation] = {}
    for citation in citations:
        first_locations.setdefault(citation.key, citation)

    unique_bib_keys = set(key_counts)
    cited_keys = set(first_locations)
    missing_keys = cited_keys - unique_bib_keys
    unused_keys = unique_bib_keys - cited_keys

    expected_order = [
        citation.key
        for citation in sorted(first_locations.values(), key=lambda item: item.order)
        if citation.key in unique_bib_keys
    ]
    current_used_order = [entry.key for entry in entries if entry.key in cited_keys]

    return BibReport(
        entries=entries,
        duplicates=duplicates,
        duplicate_lines=dict(duplicate_lines),
        citations=citations,
        first_locations=first_locations,
        missing_keys=missing_keys,
        unused_keys=unused_keys,
        expected_order=expected_order,
        current_used_order=current_used_order,
    )


def print_duplicate_report(report: BibReport, entries_by_key: dict[str, BibEntry]) -> None:
    print("=" * 70)
    print("DUPLICATE KEYS")
    print("=" * 70)
    if not report.duplicates:
        print("  No duplicates found!")
        return

    for key, count in sorted(report.duplicates.items()):
        lines = report.duplicate_lines[key]
        print(f"  {key}: appears {count} times at lines {lines}")
        if key in entries_by_key:
            print(entries_by_key[key].text)


def print_unused_report(report: BibReport) -> None:
    print("\n" + "=" * 70)
    print("UNUSED BIB ENTRIES")
    print("=" * 70)
    print(
        f"\nFound {len(report.unused_keys)} unused entries out of "
        f"{len(set(entry.key for entry in report.entries))} total:\n"
    )
    if not report.unused_keys:
        print("  No unused entries found!")
        return

    unused_entries = [entry for entry in report.entries if entry.key in report.unused_keys]
    for entry in unused_entries:
        print(f"  Line {entry.line:4d}: {entry.key}")


def print_missing_report(report: BibReport) -> None:
    print("\n" + "=" * 70)
    print("MISSING BIB ENTRIES")
    print("=" * 70)
    if not report.missing_keys:
        print("  No missing entries found!")
        return

    for key in sorted(report.missing_keys):
        location = report.first_locations[key]
        print(f"  {key}: {location.chapter_path}:{location.line}")


def print_order_report(report: BibReport) -> None:
    print("\n" + "=" * 70)
    print("FIRST-APPEARANCE ORDER")
    print("=" * 70)
    if not report.has_order_drift:
        print("  Bibliography entries already match first citation order.")
        return

    print("  Bibliography entries are not in first citation order.")
    print("  First mismatches:")
    mismatch_count = 0
    for index, (expected, current) in enumerate(
        zip(report.expected_order, report.current_used_order), start=1
    ):
        if expected == current:
            continue
        expected_loc = report.first_locations[expected]
        current_loc = report.first_locations.get(current)
        current_detail = ""
        if current_loc:
            current_detail = f" first used in {current_loc.chapter_path.name}:{current_loc.line}"
        print(
            f"  {index:4d}: expected {expected} "
            f"({expected_loc.chapter_path.name}:{expected_loc.line}), "
            f"found {current}{current_detail}"
        )
        mismatch_count += 1
        if mismatch_count >= 20:
            break


def print_summary(report: BibReport) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    unique_keys = set(entry.key for entry in report.entries)
    print(f"  Total bib entries: {len(report.entries)}")
    print(f"  Unique bib keys: {len(unique_keys)}")
    print(f"  Duplicate keys: {len(report.duplicates)}")
    print(f"  Citations found in markdown: {len(set(report.first_locations))}")
    print(f"  Missing entries: {len(report.missing_keys)}")
    print(f"  Unused entries: {len(report.unused_keys)}")
    print(f"  First-appearance order drift: {report.has_order_drift}")


def print_report(report: BibReport) -> None:
    first_entries_by_key: dict[str, BibEntry] = {}
    for entry in report.entries:
        first_entries_by_key.setdefault(entry.key, entry)

    print_duplicate_report(report, first_entries_by_key)
    print_unused_report(report)
    print_missing_report(report)
    print_order_report(report)
    print_summary(report)


def build_clean_bib(bib_content: str, chapters_dir: Path, report: BibReport) -> str:
    if report.duplicates:
        duplicate_list = ", ".join(sorted(report.duplicates))
        raise ValueError(f"Refusing to rewrite bib with duplicate keys: {duplicate_list}")

    entries_by_key = {entry.key: entry for entry in report.entries}
    sections = book_sections(chapters_dir, bib_content)
    entries_by_section: dict[str, list[BibEntry]] = defaultdict(list)

    for key in report.expected_order:
        location = report.first_locations[key]
        entries_by_section[location.section_id].append(entries_by_key[key])

    chunks: list[str] = []
    for section in sections:
        entries = entries_by_section.get(section.id, [])
        if not entries:
            continue

        chunks.append(section.header)
        chunks.extend(entry.text.rstrip() for entry in entries)
        chunks.append(section.end)

    return "\n\n".join(chunks).rstrip() + "\n"


def main() -> int:
    args = parse_args()

    bib_content = args.bib_file.read_text()
    report = analyze_bib(bib_content, args.chapters_dir)
    print_report(report)

    if args.fix:
        print("\n" + "=" * 70)
        print("APPLYING FIXES")
        print("=" * 70)
        new_content = build_clean_bib(bib_content, args.chapters_dir, report)
        if new_content == bib_content:
            print("  No changes needed.")
        else:
            args.bib_file.write_text(new_content)
            removed = len(report.unused_keys)
            print(f"  Rewrote {args.bib_file}")
            print(f"  Removed {removed} unused entries.")
            print("  Sorted entries by first citation appearance.")
            bib_content = new_content
            report = analyze_bib(bib_content, args.chapters_dir)
            print("\nPost-fix verification:")
            print_summary(report)

    if args.strict and report.has_problems:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
