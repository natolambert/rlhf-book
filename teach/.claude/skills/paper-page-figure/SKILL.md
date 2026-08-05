---
name: paper-page-figure
description: Turn the first page of a paper (arXiv or any PDF) into a slide-ready PNG for a colloquium deck's 30-40% right column. Use when a slide cites a landmark paper and a screenshot of the paper itself is the best visual.
allowed-tools: Bash(curl:*), Bash(magick:*), Bash(identify:*), Bash(cp:*), Read, Edit
---

# Paper Page Figure

Render the first page of a paper as a PNG for slides. A full paper page is
portrait (~1:1.27), which is exactly the right shape for a `columns: 62/38`
right column — prefer the full page over cropping to the title/abstract
region unless the slide needs the abstract text to be readable.

## Steps

1. Download the PDF (for arXiv, the bare `/pdf/<id>` URL, no `.pdf` suffix
   needed):

   ```bash
   curl -sL "https://arxiv.org/pdf/2112.00861" -o paper.pdf
   ```

2. Convert page 1 to PNG. Density 200 is plenty for a slide column; `-trim`
   removes the page margins and a small white border re-adds clean padding;
   `PNG8:` keeps black-text-on-white pages tiny (~80KB) with no visible loss:

   ```bash
   magick -density 200 'paper.pdf[0]' -trim +repage \
     -bordercolor white -border 24 -resize x1400 -strip \
     PNG8:paper-firstpage.png
   ```

3. Copy the reviewed result into the assets directory the deck uses
   (`teach/course/assets/` for course lectures, the talk's own `assets/`
   otherwise) with a descriptive name, e.g. `hhh-paper-2021.png`.

4. Reference it as a right column with no caption (the page is
   self-captioning):

   ```markdown
   <!-- columns: 62/38 -->
   ## Slide title

   Body text... [@citekey]

   |||

   ![](assets/hhh-paper-2021.png)
   ```

## Notes

- Work in the session scratchpad, not the repo; only `cp` the final PNG in.
- If the page carries a vertical arXiv stamp on the left edge, leave it —
  it reads as authenticity. `-trim` already keeps it because it is ink.
- For a landscape variant (title + authors + abstract only), crop before
  resizing: add `-crop x62%+0+0 -trim` after the first `-trim +repage`.
  Only use this when the abstract must be readable on the slide.
- Screenshots of web pages (announcements, tweets) are a different
  workflow: headless Chrome `--screenshot`, see the lecture decks' recent
  assets for examples.
