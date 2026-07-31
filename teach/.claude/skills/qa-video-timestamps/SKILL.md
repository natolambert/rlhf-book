---
name: qa-video-timestamps
description: Extract per-question (MM:SS) timestamps from a recorded course video by OCR-ing the slide counter, then add YouTube deep links to the course page's Q&A question lists. Use after a Q&A or lecture recording is published, or when adding/refreshing "Show questions" timestamp links in book/templates/course.html.
allowed-tools: Bash(uv:*), Bash(uvx:*), Bash(git:*), Read, Edit, Write
---

# Q&A Video Timestamps

Turn a published lecture/Q&A recording into per-question `(MM:SS)` YouTube
links on the course page. Works by OCR-ing the "Lambert n/N" slide counter
that every colloquium deck prints in the footer — no transcript needed.
macOS only (uses the Vision framework via pyobjc).

## Prerequisites

- The deck's question slides and their slide numbers, from the built deck:

  ```bash
  uv run --extra teach colloquium build teach/course/<deck>.md -o /tmp/deck-build/
  ```

  Slide numbers equal colloquium's `#N` hash anchors, so the same numbers
  serve both the slide deep links and the timestamp extraction targets.

## Steps

1. **Download the video** (1080p, video-only — 480p is too small to OCR the
   footer counter reliably):

   ```bash
   uvx yt-dlp -f "bv*[height<=1080][ext=mp4]/bv*[height<=1080]" --no-playlist \
     -o "video.mp4" "https://www.youtube.com/watch?v=<ID>"
   ```

2. **Coarse sweep** — sample a frame every 5 s and OCR the counter
   (`ocr_slides.py` in this skill directory; crops assume the standard
   recording layout, slides left ~73% / webcam right):

   ```bash
   uv run --with opencv-python --with pyobjc-framework-Vision \
     --with pyobjc-framework-Quartz python ocr_slides.py video.mp4 5 > times.tsv
   ```

3. **Determine the recorded deck's total slide count** — check the modal
   value of column 3 in `times.tsv`. It may differ from today's build (e.g.
   colloquium later auto-appended a References slide). If it differs, do NOT
   assume an offset: extract probe frames at a few recorded slide numbers,
   view them, and compare against the current deck's titles to locate where
   the numbering diverges. In the qa-02 case the counts differed (26 vs 27)
   but all content-slide numbers were identical.

4. **Refine to the exact switch second** (no padding — links must land ON
   the transition, not before it; a flat pad was measurably early):

   ```bash
   uv run --with opencv-python --with pyobjc-framework-Vision \
     --with pyobjc-framework-Quartz python refine_timestamps.py \
     video.mp4 times.tsv <recorded_total> <slide,slide,...>
   ```

   Prints `slide  seconds  MM:SS` per target — the first second the slide is
   confirmed on screen.

5. **Spot-check visually** — extract a frame at one computed timestamp,
   Read it, and confirm the question slide (and its `n/N` counter) is
   showing.

6. **Add the links** in `book/templates/course.html`: inside the row's
   `<ul class="qa-questions">`, append after each question anchor:

   ```html
   <a class="qa-time" href="https://www.youtube.com/watch?v=<ID>&t=<S>s"
      target="_blank" rel="noopener noreferrer">(M:SS)</a>
   ```

   The `.qa-time` styling (red, matching the Watch action, light + dark)
   already exists in the template.

## Gotchas

- OCR misreads at 480p look plausible ("18/28" → "16/20") — always use 1080p.
- Filter readings by the expected total to drop misreads; demo segments
  (browser windows, etc.) legitimately return no counter.
- Timestamps are per-recording: if a deck is re-recorded, re-run the sweep;
  if the deck markdown changes after recording, slide anchors on the course
  page may also need re-checking against the published build.
