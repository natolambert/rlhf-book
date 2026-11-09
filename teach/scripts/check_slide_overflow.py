"""Flag slides whose content overflows the slide, collides with the footer, or has broken images.

Loads a built colloquium deck in headless Chrome, activates each slide in turn
(the way the presenter view shows it), and measures every element in the slide
body against the slide and footer bounds.

Usage:
    uv run python teach/scripts/check_slide_overflow.py build/html/teach/Cornell-CS4789-2026/index.html

Exits 1 and prints one line per problem slide if anything overflows.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome",
    "chromium",
]

TOLERANCE_PX = 2

PROBE = """
<script>
window.addEventListener("load", () => {
  const tol = %(tol)d;
  const slides = [...document.querySelectorAll("section.slide")];
  const problems = [];
  slides.forEach((s) => s.classList.remove("active"));
  slides.forEach((slide, i) => {
    slide.classList.add("active");
    // Run the deck's own on-show fitting (captioned figures, wide display math).
    if (window.colloquiumFitDisplayMathIn) window.colloquiumFitDisplayMathIn(slide);
    if (window.colloquiumFitCaptionedFiguresIn) window.colloquiumFitCaptionedFiguresIn(slide);
    const box = slide.getBoundingClientRect();
    const footer = slide.querySelector(".colloquium-footer");
    const meta = slide.querySelector(".colloquium-slide-meta");
    const floor = footer ? footer.getBoundingClientRect().top : box.bottom;
    const metaBox = meta ? meta.getBoundingClientRect() : null;
    const title = (slide.querySelector("h1, h2") || {}).textContent || "";
    const issues = [];
    slide.querySelectorAll(".slide-content *").forEach((el) => {
      // KaTeX's MathML copy is visually hidden but still has layout.
      if (el.closest(".katex-mathml")) return;
      // SVG internals (KaTeX stretchy delimiters) are clipped by their <svg>.
      if (el.ownerSVGElement) return;
      const r = el.getBoundingClientRect();
      if (r.width === 0 && r.height === 0) return;
      const tag = el.tagName.toLowerCase();
      if (tag === "img" && el.complete && el.naturalWidth === 0) {
        issues.push(`broken image ${el.getAttribute("src")}`);
      }
      const over = Math.round(r.bottom - floor);
      const right = Math.round(r.right - box.right);
      if (over > tol) issues.push(`${tag} ${over}px below content area`);
      if (right > tol) issues.push(`${tag} ${right}px past right edge`);
      if (metaBox && metaBox.width > 0 && r.right > metaBox.left + tol && r.left < metaBox.right - tol
          && r.bottom > metaBox.top + tol && r.top < metaBox.bottom - tol) {
        issues.push(`${tag} overlaps the citation`);
      }
    });
    if (issues.length) {
      problems.push({ slide: i + 1, title: title.trim(), issues: [...new Set(issues)].slice(0, 4) });
    }
    slide.classList.remove("active");
  });
  const pre = document.createElement("pre");
  pre.id = "__overflow_report";
  pre.textContent = JSON.stringify(problems);
  document.body.appendChild(pre);
});
</script>
"""


def find_chrome() -> str:
    for path in CHROME_PATHS:
        if Path(path).exists() or subprocess.run(["which", path], capture_output=True).returncode == 0:
            return path
    sys.exit("No Chrome/Chromium found")


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    deck = Path(sys.argv[1]).resolve()
    html = deck.read_text()
    probed = html.replace("</body>", PROBE % {"tol": TOLERANCE_PX} + "</body>", 1)

    # Write next to the original so relative asset paths still resolve.
    with tempfile.NamedTemporaryFile("w", suffix=".html", dir=deck.parent, delete=False) as f:
        f.write(probed)
        tmp = Path(f.name)
    try:
        out = subprocess.run(
            [
                find_chrome(),
                "--headless=new",
                "--disable-gpu",
                "--window-size=1280,720",
                "--virtual-time-budget=30000",
                "--dump-dom",
                tmp.as_uri(),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout
    finally:
        tmp.unlink()

    m = re.search(r'<pre id="__overflow_report">(.*?)</pre>', out, re.S)
    if not m:
        sys.exit("Probe did not report (page failed to load?)")
    problems = json.loads(m.group(1).replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">"))
    if not problems:
        print(f"OK: no overflow in {deck}")
        return 0
    for p in problems:
        print(f"slide {p['slide']} ({p['title']}): " + "; ".join(p["issues"]))
    return 1


if __name__ == "__main__":
    sys.exit(main())
