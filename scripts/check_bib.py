#!/usr/bin/env python3
"""Check bib.bib for duplicate keys and unused entries."""

import re
from pathlib import Path
from collections import Counter

def extract_bib_entry(content: str, key: str, start_line: int) -> str:
    """Extract the full bib entry starting from a given line."""
    lines = content.split('\n')
    entry_lines = []
    brace_count = 0
    started = False

    for i, line in enumerate(lines[start_line - 1:], start_line):
        if not started:
            if re.match(rf'^@\w+\{{{re.escape(key)},', line):
                started = True
                brace_count += line.count('{') - line.count('}')
                entry_lines.append(line)
        else:
            brace_count += line.count('{') - line.count('}')
            entry_lines.append(line)
            if brace_count <= 0:
                break

    return '\n'.join(entry_lines)

def main():
    bib_path = Path("chapters/bib.bib")
    chapters_dir = Path("chapters")

    # Extract all bib keys with line numbers
    bib_content = bib_path.read_text()
    key_pattern = re.compile(r'^@\w+\{([^,]+),', re.MULTILINE)

    keys_with_lines = []
    for i, line in enumerate(bib_content.split('\n'), 1):
        match = re.match(r'^@\w+\{([^,]+),', line)
        if match:
            keys_with_lines.append((match.group(1), i))

    all_keys = [k for k, _ in keys_with_lines]

    # Find duplicates
    print("=" * 70)
    print("DUPLICATE KEYS (with full entries for comparison)")
    print("=" * 70)
    key_counts = Counter(all_keys)
    duplicates = {k: v for k, v in key_counts.items() if v > 1}

    if duplicates:
        for key, count in sorted(duplicates.items()):
            lines = [line for k, line in keys_with_lines if k == key]
            print(f"\n>>> {key}: appears {count} times at lines {lines}")
            print("-" * 70)
            for line_num in lines:
                entry = extract_bib_entry(bib_content, key, line_num)
                print(f"[Line {line_num}]")
                print(entry)
                print()
    else:
        print("  No duplicates found!")

    # Read all markdown files and find citations
    print("\n" + "=" * 70)
    print("UNUSED BIB ENTRIES")
    print("=" * 70)

    all_citations = set()
    md_files = list(chapters_dir.glob("*.md"))

    for md_file in md_files:
        content = md_file.read_text()
        # Match [@key] or [@key1; @key2] or [@key1;@key2] patterns
        citations = re.findall(r'\[@([^\]@;]+?)(?:[;\s]|(?=\]))', content)
        all_citations.update(citations)
        # Also match multi-citations like [@key1; @key2]
        multi_cites = re.findall(r'\[([^\]]+)\]', content)
        for mc in multi_cites:
            if '@' in mc:
                # Extract all @key references
                refs = re.findall(r'@([^;\s\]]+)', mc)
                all_citations.update(refs)

    unique_keys = set(all_keys)
    unused_keys = unique_keys - all_citations

    # Sort unused keys and show with line numbers
    unused_with_lines = []
    for key in unused_keys:
        for k, line in keys_with_lines:
            if k == key:
                unused_with_lines.append((key, line))
                break

    unused_with_lines.sort(key=lambda x: x[1])

    print(f"\nFound {len(unused_keys)} unused entries out of {len(unique_keys)} total:\n")
    for key, line in unused_with_lines:
        print(f"  Line {line:4d}: {key}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total bib entries: {len(all_keys)}")
    print(f"  Unique bib keys: {len(unique_keys)}")
    print(f"  Duplicate keys: {len(duplicates)}")
    print(f"  Citations found in markdown: {len(all_citations)}")
    print(f"  Unused entries: {len(unused_keys)}")

    # Show citations that don't exist in bib (potential typos)
    missing = all_citations - unique_keys
    if missing:
        print(f"\n  WARNING: {len(missing)} citations reference non-existent keys:")
        for key in sorted(missing):
            print(f"    - {key}")

if __name__ == "__main__":
    main()
