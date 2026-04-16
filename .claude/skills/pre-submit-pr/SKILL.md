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
   cd code && uvx ruff check . && uvx ruff format --check .
   ```
   Auto-fix with `uvx ruff check --fix .` and `uvx ruff format .` if issues are found.

2. **Summarize PR readiness**

## Output Format

```
## Pre-Submit PR Report

### Automated Checks
| Check | Status | Details |
|-------|--------|---------|
| Ruff lint | PASS/FAIL | [details] |
| Ruff format | PASS/FAIL | [details] |

### Verdict: READY FOR PR / ISSUES TO ADDRESS

### Summary for PR Description
[2-3 sentences summarizing changes]
```

## Blocking Issues

These block PR submission:
- Ruff lint or format failures (CI-enforced)

5. **Run code review** (for significant changes):
   Invoke `pr-review-toolkit:review-pr` for deeper analysis.
   Include findings in the report under a "### Code Review" section.

## Non-Blocking (Flag for Reviewers)

Note in PR but don't block:
- some TODOs in code (unless excessive)
- some print statements (unless excessive)
