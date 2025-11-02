#!/usr/bin/env python3
"""
Strip problematic Unicode characters from markdown files.
Converts them to ASCII equivalents where possible.
"""

import sys
import glob
import re

# Mapping of Unicode characters to ASCII/LaTeX equivalents
UNICODE_MAP = {
    '\u2060': '',  # Word joiner (invisible) - remove
    '\u2011': '-',  # Non-breaking hyphen
    '\u202F': ' ',  # Narrow no-break space
    '\u2018': "'",  # Left single quotation mark
    '\u2019': "'",  # Right single quotation mark
    '\u201C': '"',  # Left double quotation mark
    '\u201D': '"',  # Right double quotation mark
    '\u2013': '--',  # En dash
    '\u2014': '---',  # Em dash
    '\u2026': '...',  # Ellipsis
    '\u00B0': 'degrees',  # Degree symbol
    '\u00D7': 'x',  # Multiplication sign
    '\u2248': '~',  # Almost equal to
    '\u223C': '~',  # Tilde operator
}

def fix_unicode_in_file(filepath, dry_run=True):
    """Fix Unicode characters in a single file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    changes = []

    # Apply replacements
    for unicode_char, replacement in UNICODE_MAP.items():
        if unicode_char in content:
            count = content.count(unicode_char)
            content = content.replace(unicode_char, replacement)
            char_name = f'U+{ord(unicode_char):04X}'
            changes.append(f"  - Replaced {count} occurrences of {char_name} ({repr(unicode_char)}) with {repr(replacement)}")

    if content != original_content:
        print(f"\n{filepath}:")
        for change in changes:
            print(change)

        if not dry_run:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print("  ✓ File updated")
        else:
            print("  (dry run - no changes made)")
        return True
    return False

def main():
    dry_run = '--dry-run' in sys.argv or '-n' in sys.argv

    if dry_run:
        print("DRY RUN MODE - No files will be modified\n")
    else:
        print("MODIFYING FILES - Unicode characters will be replaced\n")

    files_modified = 0
    for filepath in sorted(glob.glob('chapters/*.md')):
        if fix_unicode_in_file(filepath, dry_run):
            files_modified += 1

    print(f"\n{'Would modify' if dry_run else 'Modified'} {files_modified} file(s)")

    if dry_run and files_modified > 0:
        print("\nRun without --dry-run to apply changes")

if __name__ == '__main__':
    main()