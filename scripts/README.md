# Scripts

Utilities for managing the bibliography and other book maintenance tasks.

## Bibliography Tools

- **check_bib_integrity.py** - Check bib.bib for duplicate keys and unused entries. Reports citations in markdown that reference non-existent bib keys.
  ```bash
  python scripts/check_bib_integrity.py
  ```

- **check_arxiv_publications.py** - Scan arxiv preprints to find papers that have been published at peer-reviewed venues (uses DBLP API). Saves results to JSON for review.
  ```bash
  python scripts/check_arxiv_publications.py --delay 1.0
  python scripts/check_arxiv_publications.py --resume  # if interrupted
  ```

- **update_bib_from_arxiv_check.py** - Apply updates from check_arxiv_publications.py to the bib file. Changes entry types, adds venue/booktitle, updates years.
  ```bash
  python scripts/update_bib_from_arxiv_check.py --dry-run  # preview changes
  python scripts/update_bib_from_arxiv_check.py            # apply changes
  ```

## Other Scripts

- **ensure_pdfoutput.py** - Ensure PDF output settings in LaTeX
- **generate_library.py** - Generate library files
- **normalize_tex_unicode.py** - Normalize unicode in TeX files
- **report_non_ascii.py** - Report non-ASCII characters in files
- **strip_unicode_from_markdown.py** - Strip unicode from markdown files
- **strip_xetex_branch.py** - Strip XeTeX-specific branches
