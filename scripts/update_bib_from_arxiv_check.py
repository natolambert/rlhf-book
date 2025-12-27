#!/usr/bin/env python3
"""
Update bib file entries using results from check_arxiv_publications.py.

Reads the arxiv_check_results.json output and updates bib entries for papers
that have been published in peer-reviewed venues.

Usage:
    python update_bib_from_arxiv_check.py [--results arxiv_check_results.json] [--bib-file bib.bib]
"""

import argparse
import json
import re
from pathlib import Path

# Venue short names to full conference/journal names
VENUE_TO_FULL = {
    "ICLR": "International Conference on Learning Representations (ICLR)",
    "ICML": "International Conference on Machine Learning (ICML)",
    "NeurIPS": "Advances in Neural Information Processing Systems (NeurIPS)",
    "NIPS": "Advances in Neural Information Processing Systems (NeurIPS)",
    "ACL": "Annual Meeting of the Association for Computational Linguistics (ACL)",
    "EMNLP": "Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    "NAACL": "Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)",
    "NAACL-HLT": "Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)",
    "Trans. Mach. Learn. Res.": "Transactions on Machine Learning Research (TMLR)",
    "FAccT": "ACM Conference on Fairness, Accountability, and Transparency (FAccT)",
    "EuroSys": "European Conference on Computer Systems (EuroSys)",
    "DATE": "Design, Automation and Test in Europe (DATE)",
}

# Venues that are journals (not conferences)
JOURNAL_VENUES = {"Trans. Mach. Learn. Res.", "Electron. Colloquium Comput. Complex."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("arxiv_check_results.json"),
        help="Path to results JSON from check_arxiv_publications.py",
    )
    parser.add_argument(
        "--bib-file",
        type=Path,
        default=Path("chapters/bib.bib"),
        help="Path to bib file to update",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes",
    )
    args = parser.parse_args()

    # Load results
    if not args.results.exists():
        print(f"Error: {args.results} not found. Run check_arxiv_publications.py first.")
        return 1

    results = json.loads(args.results.read_text())
    published = results.get("published", [])

    if not published:
        print("No published papers found in results.")
        return 0

    # Load bib file
    bib_content = args.bib_file.read_text()

    # Build a map of key -> (start, end, entry_text)
    entry_positions = {}
    pattern = re.compile(r"^(@\w+\{([^,]+),.*?^})", re.MULTILINE | re.DOTALL)
    for match in pattern.finditer(bib_content):
        key = match.group(2)
        entry_positions[key] = (match.start(), match.end(), match.group(1))

    print(f"Found {len(published)} papers with published versions\n")

    replacements = []  # (start, end, new_entry)

    for paper in published:
        key = paper["key"]
        venue = paper["venue"]
        year = paper["year"]

        if key not in entry_positions:
            print(f"  ⚠ {key} not found in bib")
            continue

        start, end, entry = entry_positions[key]
        original = entry

        is_journal = venue in JOURNAL_VENUES
        venue_full = VENUE_TO_FULL.get(venue, venue)

        # Update entry type
        if is_journal:
            entry = re.sub(r"^@\w+\{", "@article{", entry)
        else:
            entry = re.sub(r"^@\w+\{", "@inproceedings{", entry)

        # Remove arxiv-related fields
        entry = re.sub(r"\s*journal\s*=\s*\{arXiv preprint[^}]*\},?\n?", "\n", entry)
        entry = re.sub(r"\s*eprint\s*=\s*\{[^}]*\},?\n?", "\n", entry)
        entry = re.sub(r"\s*archiveprefix\s*=\s*\{[^}]*\},?\n?", "\n", entry)
        entry = re.sub(r"\s*primaryclass\s*=\s*\{[^}]*\},?\n?", "\n", entry)

        # Add/update venue field
        if is_journal:
            if re.search(r"journal\s*=", entry):
                entry = re.sub(r"journal\s*=\s*\{[^}]*\}", f"journal = {{{venue_full}}}", entry)
            else:
                entry = re.sub(
                    r"(title\s*=\s*\{[^}]*\},?)",
                    f"\\1\n  journal = {{{venue_full}}},",
                    entry,
                )
        else:
            if re.search(r"booktitle\s*=", entry):
                entry = re.sub(
                    r"booktitle\s*=\s*\{[^}]*\}",
                    f"booktitle = {{{venue_full}}}",
                    entry,
                )
            else:
                entry = re.sub(
                    r"(title\s*=\s*\{[^}]*\},?)",
                    f"\\1\n  booktitle = {{{venue_full}}},",
                    entry,
                )

        # Update year
        entry = re.sub(r"year\s*=\s*\{?\d+\}?", f"year = {{{year}}}", entry)

        if entry != original:
            replacements.append((start, end, entry))
            print(f"  ✓ {key} -> {venue} ({year})")

    if args.dry_run:
        print(f"\nDry run: would update {len(replacements)} entries")
        return 0

    # Apply replacements in reverse order to preserve positions
    replacements.sort(key=lambda x: x[0], reverse=True)
    new_content = bib_content
    for start, end, new_entry in replacements:
        new_content = new_content[:start] + new_entry + new_content[end:]

    args.bib_file.write_text(new_content)
    print(f"\nUpdated {len(replacements)} entries in {args.bib_file}")
    return 0


if __name__ == "__main__":
    exit(main())
