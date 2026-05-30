---
name: serve-course-lecture
description: Start or verify live preview for RLHF Book course lecture slides without breaking relative image assets. Use when serving, opening, checking, or debugging `teach/course/lec*.md` slides.
allowed-tools: Bash(uv:*), Bash(git:*), Bash(lsof:*), Bash(kill:*), Read
---

# Serve Course Lecture

Use this for live preview of `teach/course/lec*.md`.

## Rule

Course lecture images are written as `assets/...`. The rendered HTML must therefore be generated and served from `teach/course/`, not `teach/`, or images will resolve to the wrong directory.

## Start Lec5

From the repo root:

```bash
uv run --extra teach python -c "from colloquium.serve import serve; serve('teach/course/lec5-chap7.md', port=8081, output_dir='teach/course')"
```

Open:

```text
http://127.0.0.1:8081/lec5-chap7.html
```

## General Pattern

For `teach/course/<lecture>.md`, set `output_dir='teach/course'` and open `http://127.0.0.1:<port>/<lecture>.html`.

## Verification

Before reporting success, verify both the page and at least one image:

```bash
uv run --extra teach python -c "import urllib.request as u; urls=['http://127.0.0.1:8081/lec5-chap7.html','http://127.0.0.1:8081/assets/rlvr-system.png']; [print(x, (r := u.urlopen(x, timeout=3)).status, r.getheader('content-type')) for x in urls]"
```

The image check must return `200 image/...`.

## If The Port Is Already Wrong

If a previous server is running from the wrong output directory, find and stop only that port before restarting:

```bash
lsof -ti tcp:8081
kill <pid>
```
