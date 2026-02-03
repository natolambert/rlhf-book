---
name: push-to-pr
description: Create a new PR or push commits to an existing PR for the current branch.
allowed-tools: Bash(git:*), Bash(gh:*)
---

# Push to PR

Create a new PR or push commits to an existing PR for the current branch.

## Usage

```
/push-to-pr [commit message]
```

## Instructions

When this command is invoked:

### 1. Check current state

```bash
# Get current branch
git branch --show-current

# Check if there are uncommitted changes
git status --porcelain

# Check if branch has a remote tracking branch
git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null

# Check if there's an existing PR for this branch
gh pr list --head $(git branch --show-current) --json number,url,title
```

### 2. If there are uncommitted changes

- Stage all changes: `git add -A`
- Create a commit with the provided message, or generate one based on the changes
- Use this commit format:
```bash
git commit -m "$(cat <<'EOF'
<commit message>

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

### 3. If no PR exists for this branch

- Push the branch: `git push -u origin $(git branch --show-current)`
- Create a new PR using:
```bash
gh pr create --title "<title based on branch/changes>" --body "$(cat <<'EOF'
## Summary
<brief summary of changes>

## Changes
<bullet points of what changed>

🤖 Generated with [Claude Code](https://claude.ai/code)
EOF
)"
```
- Return the PR URL to the user

### 4. If a PR already exists

- Push the new commits: `git push`
- Report that commits were pushed to the existing PR
- Return the existing PR URL

## Example

```
/push-to-pr "Add tool use diagram to chapter 14.5"
```

This will either:
- Create a new PR if one doesn't exist
- Push a new commit to the existing PR if one does exist
