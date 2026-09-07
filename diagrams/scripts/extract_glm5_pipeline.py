"""Crop Figure 5 from arXiv:2602.15763v2, preserving its PDF artwork.

Download https://arxiv.org/pdf/2602.15763v2 to
diagrams/generated/pdf/glm5-source.pdf before running from the repo root.
Requires pypdf and Poppler. Outputs PDF, SVG, and 400 dpi PNG.
"""
from pathlib import Path
import subprocess

from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject

ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "diagrams/generated"
for kind in ("pdf", "png", "svg"):
    (GENERATED / kind).mkdir(parents=True, exist_ok=True)

reader = PdfReader(GENERATED / "pdf/glm5-source.pdf")
page = reader.pages[3]
# The figure's /Im7 Form XObject has BBox (0, 0, 939, 528), scaled by
# 0.42172 and translated to (108, 497.331). Keep a one-point margin.
box = RectangleObject((107, 496.331, 505.19508, 721.00016))
for name in ("mediabox", "cropbox", "trimbox", "bleedbox", "artbox"):
    setattr(page, name, box)
writer = PdfWriter()
writer.add_page(page)
writer.add_metadata({
    "/Title": "GLM-5 training pipeline (Figure 5)",
    "/Author": "GLM-5 Team, Zhipu AI and Tsinghua University",
    "/Subject": "Figure 5 from arXiv:2602.15763v2. CC BY 4.0. Cropped to the figure; artwork unchanged.",
})
pdf = GENERATED / "pdf/glm5-pipeline.pdf"
with pdf.open("wb") as output:
    writer.write(output)
subprocess.run(["pdftocairo", "-svg", str(pdf),
                str(GENERATED / "svg/glm5-pipeline.svg")], check=True)
subprocess.run(["pdftoppm", "-cropbox", "-r", "400", "-png", "-singlefile",
                str(pdf), str(GENERATED / "png/glm5-pipeline")], check=True)
