####################################################################################################
# Configuration
####################################################################################################

# Build configuration

BUILD = build
MAKEFILE = Makefile
OUTPUT_FILENAME = book
OUTPUT_FILENAME_HTML = index
METADATA = metadata.yml
CHAPTERS = $(wildcard chapters/*.md) # was CHAPTERS = chapters/*.md
TOC = --toc --toc-depth 3
METADATA_ARGS = --metadata-file $(METADATA)
IMAGES = $(shell find images -type f)
TEMPLATES = $(shell find templates/ -type f)
COVER_IMAGE = images/cover.png
EPUB_COVER_IMAGE = images/rlhf-book-cover.png # EPUB-specific cover image
MATH_FORMULAS = --mathjax # --webtex, is default for PDF/ebook. Consider resetting if issues.
EPUB_MATH_FORMULAS = --mathml # Use MathML for EPUB format for better e-reader compatibility
BIBLIOGRAPHY = --bibliography=chapters/bib.bib --citeproc --csl=templates/ieee.csl

# Chapters content
CONTENT = awk 'FNR==1 && NR!=1 {print "\n\n"}{print}' $(CHAPTERS)
CONTENT_FILTERS = tee # Use this to add sed filters or other piped commands

# Debugging

DEBUG_ARGS = --verbose

# Pandoc filtes - uncomment the following variable to enable cross references filter. For more
# information, check the "Cross references" section on the README.md file.

FILTER_ARGS = --filter pandoc-crossref

# Combined arguments

ARGS = $(TOC) $(MATH_FORMULAS) $(METADATA_ARGS) $(FILTER_ARGS) $(DEBUG_ARGS) $(BIBLIOGRAPHY)
EPUB_ARGS_BASE = $(TOC) $(EPUB_MATH_FORMULAS) $(METADATA_ARGS) $(FILTER_ARGS) $(DEBUG_ARGS) $(BIBLIOGRAPHY)
	
PANDOC_COMMAND = pandoc

# Per-format options

DOCX_ARGS = --standalone --reference-doc templates/docx.docx
EPUB_ARGS = --template templates/epub.html --epub-cover-image $(EPUB_COVER_IMAGE)
HTML_ARGS = --template templates/html.html --standalone --to html5 --listings
PDF_ARGS = --template templates/pdf.tex --pdf-engine pdflatex --listings
LATEX_ARGS = --template templates/pdf.tex --pdf-engine pdflatex --listings
NESTED_HTML_TEMPLATE = templates/chapter.html
ARXIV_ZIP = $(BUILD)/arxiv.zip

# Add this with your other file variables at the top
JS_FILES = $(shell find templates -name '*.js')  # Restrict JS discovery to source templates

# Per-format file dependencies

BASE_DEPENDENCIES = $(MAKEFILE) $(CHAPTERS) $(METADATA) $(IMAGES) $(TEMPLATES)
DOCX_DEPENDENCIES = $(BASE_DEPENDENCIES)
EPUB_DEPENDENCIES = $(BASE_DEPENDENCIES)
HTML_DEPENDENCIES = $(BASE_DEPENDENCIES)
PDF_DEPENDENCIES = $(BASE_DEPENDENCIES)

# Detected Operating System

OS = $(shell sh -c 'uname -s 2>/dev/null || echo Unknown')

# OS specific commands

ifeq ($(OS),Darwin) # Mac OS X
	COPY_CMD = cp -P
else # Linux
	COPY_CMD = cp --parent
endif

MKDIR_CMD = mkdir -p
RMDIR_CMD = rm -r
ARXIV_DIR = $(BUILD)/arxiv
ECHO_BUILDING = @echo "building $@..."
ECHO_BUILT = @echo "$@ was built\n"

####################################################################################################
# Basic actions
####################################################################################################

.PHONY: all book clean epub html pdf docx nested_html latex

all:	book

book:	epub html pdf docx

clean:
	$(RMDIR_CMD) $(BUILD)

# Debugging output for chapters and HTML output paths
$(info Chapters found: $(CHAPTERS))
$(info HTML output will be: $(CHAPTER_HTMLS))
$(info JS files found: $(JS_FILES))

####################################################################################################
# File builders
####################################################################################################

epub:	$(BUILD)/epub/$(OUTPUT_FILENAME).epub

html:	nested_html $(BUILD)/html/$(OUTPUT_FILENAME_HTML).html $(BUILD)/html/library.html
	
pdf:	$(BUILD)/pdf/$(OUTPUT_FILENAME).pdf

docx:	$(BUILD)/docx/$(OUTPUT_FILENAME).docx

latex:	$(BUILD)/latex/$(OUTPUT_FILENAME).tex

$(BUILD)/epub/$(OUTPUT_FILENAME).epub:	$(EPUB_DEPENDENCIES)
	$(ECHO_BUILDING)
	$(MKDIR_CMD) $(BUILD)/epub
	$(CONTENT) | $(CONTENT_FILTERS) | $(PANDOC_COMMAND) $(EPUB_ARGS_BASE) $(EPUB_ARGS) -o $@
	$(ECHO_BUILT)


$(BUILD)/docx/$(OUTPUT_FILENAME).docx:	$(DOCX_DEPENDENCIES)
	$(ECHO_BUILDING)
	$(MKDIR_CMD) $(BUILD)/docx
	$(CONTENT) | $(CONTENT_FILTERS) | $(PANDOC_COMMAND) $(ARGS) $(DOCX_ARGS) -o $@
	$(ECHO_BUILT)
	
$(BUILD)/html/$(OUTPUT_FILENAME_HTML).html:	$(HTML_DEPENDENCIES)
	$(ECHO_BUILDING)
	$(MKDIR_CMD) $(BUILD)/html
	$(MKDIR_CMD) $(BUILD)/html/c
	$(CONTENT) | $(CONTENT_FILTERS) | $(PANDOC_COMMAND) $(ARGS) $(HTML_ARGS) -o $@
	$(COPY_CMD) $(IMAGES) $(BUILD)/html/
	$(COPY_CMD) templates/nav.js $(BUILD)/html/
	$(COPY_CMD) templates/header-anchors.js $(BUILD)/html/
	$(COPY_CMD) templates/table-scroll.js $(BUILD)/html/
	$(COPY_CMD) templates/nav.js $(BUILD)/html/c/
	$(COPY_CMD) templates/header-anchors.js $(BUILD)/html/c/
	$(COPY_CMD) templates/table-scroll.js $(BUILD)/html/c/
	cp templates/style.css $(BUILD)/html/style.css || echo "Failed to copy style.css"
	@mkdir -p $(BUILD)/html/data
	@test -f data/library.json && cp data/library.json $(BUILD)/html/data/library.json || echo "No library data to copy"
	$(ECHO_BUILT)

$(BUILD)/html/library.html: templates/library.html
	$(MKDIR_CMD) $(BUILD)/html
	cp templates/library.html $@

# Nested HTML build targets
NESTED_HTML_DIR = $(BUILD)/html/c/
CHAPTER_HTMLS = $(patsubst chapters/%.md,$(NESTED_HTML_DIR)/%.html,$(CHAPTERS))

# Rule to build each HTML file from each Markdown file
$(NESTED_HTML_DIR)/%.html: chapters/%.md $(HTML_DEPENDENCIES)
	$(MKDIR_CMD) $(NESTED_HTML_DIR)
	$(PANDOC_COMMAND) $(ARGS) --template $(NESTED_HTML_TEMPLATE) --standalone --to html5 -o $@ $< --mathjax
	@echo "Built HTML for $<"

# Aggregate target for nested chapter HTML files
nested_html: $(CHAPTER_HTMLS)
	@echo "All nested HTML files built"

# ArXiv‑compatible LaTeX build rule
$(BUILD)/latex/$(OUTPUT_FILENAME).tex: $(PDF_DEPENDENCIES)
	$(ECHO_BUILDING)
	$(MKDIR_CMD) -p $(BUILD)/latex

	# 1. Generate the LaTeX file with Pandoc (tell Pandoc where to find images)
	$(CONTENT) \
	  | $(CONTENT_FILTERS) \
	  | $(PANDOC_COMMAND) $(ARGS) $(LATEX_ARGS) --resource-path=. -o $@

	# 2. Flatten image paths — copy every referenced image into the build dir root
	$(foreach img,$(IMAGES), cp $(img) $(BUILD)/latex/$(notdir $(img));)

	# 3a. Force pdfLaTeX mode for arXiv
	python3 scripts/ensure_pdfoutput.py $@

	# 3b. Strip directory prefixes in \includegraphics paths (portable)
	perl -0pi -e 's|(\\includegraphics(?:\[[^]]*\])?\{)[^/}]+/|\1|g' $@

	# 3c. Restore missing \includegraphics inside \pandocbounded{}
	perl -CSD -pi -e 's/\\pandocbounded\{([^{}]+)\}\}/\\pandocbounded{\\includegraphics{$$1}}/g' $@

	# 3d. Unicode → ASCII/TeX normalisation (map accents and punctuation)
	python3 scripts/normalize_tex_unicode.py $@

	# 3e. Drop XeTeX/LuaTeX-only branch so arXiv's pdfLaTeX build doesn't demand Unicode engines
	python3 scripts/strip_xetex_branch.py $@

	# 4. Copy bibliography and CSL files required by arXiv
	cp chapters/bib.bib    $(BUILD)/latex/
	cp templates/ieee.csl  $(BUILD)/latex/

	# 5. Warn (but don\'t fail) if any non-ASCII bytes remain
	python3 scripts/report_non_ascii.py $@

	# 6. Package arXiv-ready source bundle
	rm -f $(ARXIV_ZIP)
	(cd $(BUILD)/latex && zip -rq ../$(notdir $(ARXIV_ZIP)) .)

	$(ECHO_BUILT)




$(BUILD)/pdf/$(OUTPUT_FILENAME).pdf:	$(PDF_DEPENDENCIES)
	$(ECHO_BUILDING)
	$(MKDIR_CMD) $(BUILD)/pdf
	$(CONTENT) | $(CONTENT_FILTERS) | $(PANDOC_COMMAND) $(ARGS) $(PDF_ARGS) -o $@
	$(ECHO_BUILT)

# copy faveicon.ico to build/ and into  build/c/ with bash commands
# also copy from build/pdf/book.pdf into build/html/
# then copy images dir to build/html/chapters/
files:
	test -f favicon.ico || (echo "favicon.ico not found" && exit 1)
	mkdir -p $(BUILD)/html/c/
	cp favicon.ico $(BUILD)/html/ || echo "Failed to copy to $(BUILD)/html/"
	cp favicon.ico $(BUILD)/html/c/ || echo "Failed to copy to $(BUILD)/html/c/"
	cp -R preorder $(BUILD)/html/ || echo "Failed to copy preorder static pages"
	cp $(BUILD)/pdf/book.pdf $(BUILD)/html/ || echo "Failed to copy to $(BUILD)/html/"
	cp $(BUILD)/epub/book.epub $(BUILD)/html/ || echo "Failed to copy EPUB to $(BUILD)/html/"
	cp -r images $(BUILD)/html/c/ || echo "Failed to copy to $(BUILD)/html/chapters/"
	cp ./templates/nav.js $(BUILD)/html/ || echo "Failed to copy nav.js to $(BUILD)/html/"
	cp ./templates/nav.js $(BUILD)/html/c/ || echo "Failed to copy nav.js to $(BUILD)/html/c/"
	cp ./templates/header-anchors.js $(BUILD)/html/ || echo "Failed to copy header-anchors.js to $(BUILD)/html/"
	cp ./templates/header-anchors.js $(BUILD)/html/c/ || echo "Failed to copy header-anchors.js to $(BUILD)/html/c/"
	cp ./templates/table-scroll.js $(BUILD)/html/ || echo "Failed to copy table-scroll.js to $(BUILD)/html/"
	cp ./templates/table-scroll.js $(BUILD)/html/c/ || echo "Failed to copy table-scroll.js to $(BUILD)/html/c/"
