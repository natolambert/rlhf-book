# Training-recipe diagrams

These two schematics extend the visual language of Chapter 3's figures 5 and 6
(`book/images/rlhf-basic.png` and `rlhf-complex.png`): plain outlined model boxes,
gray open arrowheads, rounded connections, and italic training-operation labels.
Boxes are model checkpoints; arrows describe how they are trained.

The shared style is `../_shared/styles_training_recipes.tex`. Define
`\figdark=1` to use the site's slate palette. Dark PNGs have transparent canvases,
slate-filled boxes, and light text, outlines, and arrows.

## Specialist teachers and MOPD

![Specialist teachers and MOPD](../../../book/images/rlhf-mopd.png)

`rlhf_mopd_tikz.tex` shows an initial shared SFT checkpoint, then separate SFT
and RL for each domain teacher, followed by multi-teacher on-policy distillation
(MOPD) into one general student. The `1`, `2`, …, `N` rows represent arbitrary
domains, not a model's reported teacher count. MOPD transfers knowledge using
teacher output distributions on student rollouts; it does not average weights.
Student initialization and the rollout/loss mechanics are outside this overview.

This is the requested generic training pattern, inspired by the
[MiMo-V2-Flash report, §4.1](https://arxiv.org/html/2601.02780v1#S4.SS1) and
[Nemotron 3 Ultra report, §3](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf),
as discussed in the [course slides](https://rlhfbook.com/teach/course/conversation-01/#14).
It is not a literal reconstruction of either complete recipe. MiMo describes
specialized RL/SFT without requiring every teacher to follow the same two stages.
Nemotron adds student RL, varied teacher paths, multiple distillation rounds, and
other finishing steps that are deliberately omitted here.

## Sequential reinforcement learning

![Sequential reinforcement learning](../../../book/images/rlhf-sequential-rl.png)

`rlhf_sequential_rl_tikz.tex` shows overall SFT followed by reasoning RL,
agentic RL, and general RL on successive checkpoints of one model. This uses
the stage order in the supplied **GLM-5** slide and
[GLM-5 technical report, §3](https://arxiv.org/html/2602.15763v1#S3).
The supplied reference is GLM-5, so this schematic does not assert a
GLM-5.2-specific recipe. It isolates the sequential RL stages: the complete
GLM-5 recipe also performs final on-policy cross-stage distillation (§3.5),
which is not depicted.

## Build and chapter assets

From the repository root:

```bash
# Both figures; final PNGs at the book's 400 dpi convention.
make -C diagrams training-recipes TIKZ_DENSITY=400

cp diagrams/generated/png/rlhf_mopd_tikz.png book/images/rlhf-mopd.png
cp diagrams/generated/png/rlhf_mopd_tikz-dark.png book/images/rlhf-mopd-dark.png
cp diagrams/generated/svg/rlhf_mopd_tikz.svg book/images/rlhf-mopd.svg
cp diagrams/generated/png/rlhf_sequential_rl_tikz.png book/images/rlhf-sequential-rl.png
cp diagrams/generated/png/rlhf_sequential_rl_tikz-dark.png book/images/rlhf-sequential-rl-dark.png
cp diagrams/generated/svg/rlhf_sequential_rl_tikz.svg book/images/rlhf-sequential-rl.svg
```

The target requires LaTeX, ImageMagick, and either `pdf2svg` or Poppler's
`pdftocairo`. The default PNG density is 800 dpi; use 300 for quick previews.
PDF/SVG exports remain vector. All intermediate files and generated exports
stay under `diagrams/generated/`; reviewed web and print assets are checked
into `book/images/`. Dark PNGs must be previewed over a dark background.

The new schematics are staged for the forthcoming text after DeepSeek R1;
they are not inserted into the chapter yet. Suggested markup:

```markdown
![A schematic of specialist post-training: shared SFT, domain-specific SFT and RL, then multi-teacher on-policy distillation into one student.](images/rlhf-mopd.png){#fig:rlhf-mopd data-dark-src="images/rlhf-mopd-dark.png"}

![A schematic of sequential post-training: overall SFT followed by reasoning, agentic, and general reinforcement learning.](images/rlhf-sequential-rl.png){#fig:rlhf-sequential-rl data-dark-src="images/rlhf-sequential-rl-dark.png"}
```


## Nature Figure 2 reproduction

`book/images/deepseek-r1-pipeline.png` and `.svg` reproduce Figure 2 from
Guo et al. (DeepSeek-AI Team), *Nature* **645**, 633–638 (2025),
[doi:10.1038/s41586-025-09422-z](https://doi.org/10.1038/s41586-025-09422-z).
The [published PDF](https://www.nature.com/articles/s41586-025-09422-z.pdf),
print page 637 (PDF page 5), licenses the article and its figures under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) unless a credit line
states otherwise. [Figure 2](https://www.nature.com/articles/s41586-025-09422-z/figures/2)
has no separate restriction. The reproduced figure remains © The Author(s) 2025,
CC BY 4.0. The chapter caption gives attribution, source, license, and the crop.

The artwork is unchanged: the PDF was cropped to Figure 2 on print page 635
(PDF page 3) with a two-point margin, excluding the publisher's caption and page
text. The crop rectangle is `(81.736, 69.398, 514.636, 276.769)` in PDF points,
measured from the page's lower-left corner. PNG output is 2405 × 1153 pixels at
400 dpi; SVG retains vector paths. The cropped PDF stays in
`diagrams/generated/pdf/`; the chapter PNG retains the original white background.

Reproduce from the repository root with `pypdf` and Poppler installed:

```bash
mkdir -p diagrams/generated/{pdf,png,svg}
curl -L --fail 'https://www.nature.com/articles/s41586-025-09422-z.pdf' \
  -o diagrams/generated/pdf/deepseek-r1-nature-source.pdf
uv run python - <<'PYTHON'
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject

reader = PdfReader('diagrams/generated/pdf/deepseek-r1-nature-source.pdf')
page = reader.pages[2]
box = RectangleObject((81.736, 69.398, 514.636, 276.769))
for name in ('mediabox', 'cropbox', 'trimbox', 'bleedbox', 'artbox'):
    setattr(page, name, box)
writer = PdfWriter()
writer.add_page(page)
writer.add_metadata({
    '/Title': 'DeepSeek-R1 multistage pipeline (Nature Figure 2)',
    '/Author': 'Daya Guo et al. (DeepSeek-AI Team)',
    '/Subject': 'Figure 2 from Nature 645, 633–638 (2025), doi:10.1038/s41586-025-09422-z. CC BY 4.0. Cropped to the original figure; artwork unchanged.',
})
with Path('diagrams/generated/pdf/deepseek-r1-pipeline.pdf').open('wb') as output:
    writer.write(output)
PYTHON
pdftocairo -svg diagrams/generated/pdf/deepseek-r1-pipeline.pdf \
  diagrams/generated/svg/deepseek-r1-pipeline.svg
pdftoppm -cropbox -r 400 -png -singlefile \
  diagrams/generated/pdf/deepseek-r1-pipeline.pdf \
  diagrams/generated/png/deepseek-r1-pipeline
cp diagrams/generated/png/deepseek-r1-pipeline.png book/images/
cp diagrams/generated/svg/deepseek-r1-pipeline.svg book/images/
```

The source PDF's SHA-256 at extraction was
`916fdaafc3b44143a9744481d079744f773d73d06372f82b1dba1076525927f2`.
Nature also serves an [official 1787 × 847 PNG](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41586-025-09422-z/MediaObjects/41586_2025_9422_Fig2_HTML.png),
which was used to check that the vector crop preserves the complete figure.
