---
name: pre-submit-pr
description: Validate changes before PR submission
---

# /pre-submit-pr

Validate changes before submitting a pull request.

## Usage

```
/pre-submit-pr
```

## Instructions

1. **Lint & format** (for changes under `code/`):
   ```bash
   cd "$(git rev-parse --show-toplevel)/code" && uvx ruff@0.14.5 check . && uvx ruff@0.14.5 format --check .
   ```
   Auto-fix with `uvx ruff@0.14.5 check --fix .` and `uvx ruff@0.14.5 format .` if issues are found. The `ruff@0.14.5` pin matches `.github/workflows/lint.yml`; unpinned `uvx ruff` can diverge from CI.

2. **Changelog** (for changes under `code/`):
   CI requires `code/CHANGELOG.md` to be modified for any PR touching `code/` (format is convention, not enforced). Check that an entry exists under `## Unreleased` for this PR, format: `- YYYY-MM-DD: [PR #N](https://github.com/natolambert/rlhf-book/pull/N) description.`

3. **Run code review** (for significant changes):
   Invoke `pr-review-toolkit:review-pr` for deeper analysis.
   Include findings in the report under a "### Code Review" section.

4. **Summarize PR readiness**

## Output Format

```
## Pre-Submit PR Report

### Automated Checks
| Check | Status | Details |
|-------|--------|---------|
| Ruff lint | PASS/FAIL | [details] |
| Ruff format | PASS/FAIL | [details] |
| Changelog | PASS/MISSING | [details] |

### Code Review
[pr-review-toolkit:review-pr findings, if run]

### Verdict: READY FOR PR / ISSUES TO ADDRESS

### Summary for PR Description
[2-3 sentences summarizing changes]
```

## Blocking Issues

These block PR submission:
- Ruff lint or format failures (CI-enforced)
- Missing changelog entry in `code/CHANGELOG.md` (CI-enforced: file must be modified)

## Non-Blocking (Flag for Reviewers)

Note in PR but don't block:
- some TODOs in code (unless excessive)
- some print statements (unless excessive)
