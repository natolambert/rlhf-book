# RLHF Book - Build System

Built on [**Pandoc book template**](https://github.com/wikiti/pandoc-book-template).

This directory contains the source files for building the RLHF Book in multiple formats (HTML, PDF, EPUB, DOCX).

## Usage

From the repository root:

```bash
make          # Build all formats
make html     # Build HTML site
make pdf      # Build PDF (requires LaTeX)
make epub     # Build EPUB
make files    # Copy assets to build output
```

### Known Conversion Issues

With the nested structure used for the website the section links between chapters in the PDF are broken.
We are opting for this in favor of a better web experience, but best practice is to not put any links to `rlhfbook.com` within the markdown files. Non-html versions will not be well suited to them.

### Common Failures When Editing with Coding Agents

Coding agents (Claude, Cursor, etc.) often introduce Unicode characters that break the Pandoc PDF build with errors like `Cannot decode byte '\xe2': Data.Text.Encoding: Invalid UTF-8 stream`. Watch for:

- **Curly apostrophes** (`'` U+2019) instead of straight apostrophes (`'`) - common in "don't", "it's", possessives
- **Em-dashes** (`—` U+2014) and **en-dashes** (`–` U+2013) instead of double-hyphens (`--`)
- **Non-breaking spaces** (`\xa0` U+00A0) instead of regular spaces
- **Curly quotes** (`"` `"` U+201C/U+201D) instead of straight quotes (`"`)

To find these: `xxd book/chapters/filename.md | grep -i 'e2 80\|c2 a0'`

To fix: `python3 -c "content = open('file.md').read(); content = content.replace('\u2019', \"'\").replace('\u2014', '--'); open('file.md', 'w').write(content)"`

## Installing

Please, check [this page](http://pandoc.org/installing.html) for more information.

### Linux

*NOTE*: This is not fully tested.

Original build instructions:
```sh
sudo apt-get install pandoc
```

This template uses [make](https://www.gnu.org/software/make/) to build the output files, so don't
forget to install it too:

```sh
sudo apt-get install make
```

To export to PDF files, make sure to install the following packages:

```sh
sudo apt-get install texlive-fonts-recommended texlive-xetex
```

User-tested build instructions (see this [issue](https://github.com/natolambert/rlhf-book/issues/117)):
> On my PopOS 22.04 Linux system let me share how I got this book to build:
>
> Install Pandoc 3.6.4 to avoid warning messages regarding pandoc-crossref
>
> `brew install pandoc-crossref` is needed to get the build working right.
>
> `sudo apt install fonts-dejavu` because my system repository had no ttf-dejavu package available
>
> `make clean` to remove build artifacts
>
> `git pull` in the project directory to update the local copy with the GitHub repository copy

### Mac
```
brew install pandoc
brew install make
brew install pandoc-crossref
```

Or, fork a process from the [github action](https://github.com/natolambert/rlhf-book/blob/main/.github/workflows/static.yml) that auto-builds new versions on MacOS.

## Folder Structure

```
book/
├── chapters/     # Markdown source files (one per chapter)
├── images/       # Image assets referenced in chapters
├── assets/       # Brand assets (covers, logos)
├── templates/    # Pandoc templates for each output format
├── scripts/      # Build utilities
├── data/         # Library data (JSON)
└── preorder/     # Pre-order redirect page
```

## Setup

Edit the *metadata.yml* file in the repository root to set configuration data:

```yml
---
title: My book title
author: Daniel Herzog
rights: MIT License
lang: en-US
tags: [pandoc, book, my-book, etc]
abstract: |
  Your summary.
mainfont: DejaVu Sans

# Filter preferences:
# - pandoc-crossref
linkReferences: true
---
```

You can find the list of all available keys on
[this page](http://pandoc.org/MANUAL.html#extension-yaml_metadata_block).

## Creating Chapters

Creating a new chapter is as simple as creating a new markdown file in the *chapters/* folder;
you'll end up with something like this:

```
chapters/01-introduction.md
chapters/02-installation.md
chapters/03-usage.md
chapters/04-references.md
```

Pandoc and Make will join them automatically ordered by name; that's why the numeric prefixes are
being used.

All you need to specify for each chapter at least one title:

```md
# Introduction

This is the first paragraph of the introduction chapter.

## First

This is the first subsection.

## Second

This is the second subsection.
```

Each title (*#*) will represent a chapter, while each subtitle (*##*) will represent a chapter's
section. You can use as many levels of sections as markdown supports.

### Manual control over page ordering

You may prefer to have manual control over page ordering instead of using numeric prefixes.

To do so, replace the CHAPTERS variable in the Makefile with your own order. For example:

```
CHAPTERS += $(addprefix ./book/chapters/,\
 01-introduction.md\
 02-installation.md\
 03-usage.md\
 04-references.md\
)
```

### Links between chapters

Anchor links can be used to link chapters within the book:

```md
// chapters/01-introduction.md
# Introduction

For more information, check the [Usage] chapter.

// chapters/02-installation.md
# Usage

...
```

If you want to rename the reference, use this syntax:

```md
For more information, check [this](#usage) chapter.
```

Anchor names should be downcased, and spaces, colons, semicolons... should be replaced with hyphens.
Instead of `Chapter title: A new era`, you have: `#chapter-title-a-new-era`.

### Links between sections

It's the same as anchor links:

```md
# Introduction

## First

For more information, check the [Second] section.

## Second

...
```

Or, with an alternative name:

```md
For more information, check [this](#second) section.
```

## Inserting Objects

### Insert an image

Use Markdown syntax to insert an image with a caption:

```md
![A cool seagull.](images/seagull.png)
```

Pandoc will automatically convert the image into a figure, using the title (the text between the
brackets) as a caption.

If you want to resize the image, you may use this syntax, available since Pandoc 1.16:

```md
![A cool seagull.](images/seagull.png){ width=50% height=50% }
```

### Insert a table

Use markdown table, and use the `Table: <Your table description>` syntax to add a caption:

```md
| Index | Name |
| ----- | ---- |
| 0     | AAA  |
| 1     | BBB  |
| ...   | ...  |

Table: This is an example table.
```

### Insert an equation

Wrap a LaTeX math equation between `$` delimiters for inline (tiny) formulas:

```md
This, $\mu = \sum_{i=0}^{N} \frac{x_i}{N}$, the mean equation, ...
```

Pandoc will transform them automatically into images using online services.

If you want to center the equation instead of inlining it, use double `$$` delimiters:

```md
$$\mu = \sum_{i=0}^{N} \frac{x_i}{N}$$
```

[Here](https://www.codecogs.com/latex/eqneditor.php)'s an online equation editor.

### Cross references

Originally, this template used LaTeX labels for auto numbering on images, tables, equations or
sections, like this:

```md
Please, admire the gloriousnes of Figure \ref{seagull_image}.

![A cool seagull.\label{seagull_image}](images/seagull.png)
```

**However, these references only works when exporting to a LaTeX-based format (i.e. PDF, LaTeX).**

In case you need cross references support on other formats, this template now support cross
references using [Pandoc filters](https://pandoc.org/filters.html). If you want to use them, use a
valid plugin and with its own syntax.

Using [pandoc-crossref](https://github.com/lierdakil/pandoc-crossref) is highly recommended, but
there are other alternatives which use a similar syntax, like
[pandoc-xnos](https://github.com/tomduck/pandoc-xnos).

First, enable the filter on the *Makefile* by updating the `FILTER_ARGS` variable with your new
filter(s):

```make
FILTER_ARGS = --filter pandoc-crossref
```

Then, you may use the filter cross references. For example, *pandoc-crossref* uses
`{#<type>:<id>}` for definitions and `@<type>:id` for referencing. Some examples:

```md
List of references:

- Check @fig:seagull.
- Check @tbl:table.
- Check @eq:equation.

List of elements to reference:

![A cool seagull](images/seagull.png){#fig:seagull}

$$ y = mx + b $$ {#eq:equation}

| Index | Name |
| ----- | ---- |
| 0     | AAA  |
| 1     | BBB  |
| ...   | ...  |

Table: This is an example table. {#tbl:table}
```

Check the desired filter settings and usage for more information
([pandoc-crossref usage](http://lierdakil.github.io/pandoc-crossref/)).

### Content filters

If you need to modify the MD content before passing it to pandoc, you may use `CONTENT_FILTERS`. By
setting this makefile variable, it will be passed to the markdown content before passing it to
pandoc. For example, to replace all occurrences of `@pagebreak` with
`<div style="page-break-before: always;"></div>` you may use a `sed` filter:

```
CONTENT_FILTERS = sed 's/@pagebreak/"<div style=\"page-break-before: always;\"><\/div>"/g'
```

To use multiple filters, you may include multiple pipes on the `CONTENT_FILTERS` variable:

```
CONTENT_FILTERS = \
  sed 's/@pagebreak/"<div style=\"page-break-before: always;\"><\/div>"/g' | \
  sed 's/@image/[Cool image](\/images\/image.png)/g'
```

## Output

This template uses *Makefile* to automatize the building process. Instead of using the *pandoc cli
util*, we're going to use some *make* commands.

### Export to PDF

Please note that PDF file generation requires some extra dependencies (~ 800 MB):

```sh
sudo apt-get install texlive-xetex ttf-dejavu
```

After installing the dependencies, use this command:

```sh
make pdf
```

The generated file will be placed in *build/pdf*.

### Export to EPUB

Use this command:

```sh
make epub
```

The generated file will be placed in *build/epub*.

### Export to HTML

Use this command:

```sh
make html
```

The generated file(s) will be placed in *build/html*.

### Export to DOCX

Use this command:

```sh
make docx
```

The generated file(s) will be placed in *build/docx*.

### Extra configuration

If you want to configure the output, you'll probably have to look the
[Pandoc Manual](http://pandoc.org/MANUAL.html) for further information about pdf (LaTeX) generation,
custom styles, etc, and modify the Makefile file accordingly.

### Templates

Output files are generated using [pandoc templates](https://pandoc.org/MANUAL.html#templates). All
templates are located under the `templates/` folder, and may be modified as you will. Some basic
format templates are already included on this repository, in case you need something to start
with.

## References

- [Pandoc](http://pandoc.org/)
- [Pandoc Manual](http://pandoc.org/MANUAL.html)
- [Wikipedia: Markdown](http://wikipedia.org/wiki/Markdown)
