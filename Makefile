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
MATH_FORMULAS = --mathjax # --webtex, is default for PDF/ebook. Consider resetting if issues.
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
	
PANDOC_COMMAND = pandoc

# Per-format options

DOCX_ARGS = --standalone --reference-doc templates/docx.docx
EPUB_ARGS = --template templates/epub.html --epub-cover-image $(COVER_IMAGE)
HTML_ARGS = --template templates/html.html --standalone --to html5 --listings
PDF_ARGS = --template templates/pdf.tex --pdf-engine xelatex --listings
NESTED_HTML_TEMPLATE = templates/chapter.html

# Add this with your other file variables at the top
JS_FILES = $(shell find . -name '*.js')  # This will find all .js files in the current directory and subdirectories

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

# Chapters content and dependencies defined first
NESTED_HTML_DIR = $(BUILD)/html/c/
CHAPTER_HTMLS = $(patsubst chapters/%.md,$(NESTED_HTML_DIR)/%.html,$(CHAPTERS))

# Debugging output for chapters and HTML output paths
$(info Chapters found: $(CHAPTERS))
$(info HTML output will be: $(CHAPTER_HTMLS))
$(info JS files found: $(JS_FILES))

####################################################################################################
# File builders
####################################################################################################

epub:	$(BUILD)/epub/$(OUTPUT_FILENAME).epub

html:	$(BUILD)/html/$(OUTPUT_FILENAME_HTML).html nested_html

pdf:	$(BUILD)/pdf/$(OUTPUT_FILENAME).pdf

docx:	$(BUILD)/docx/$(OUTPUT_FILENAME).docx

latex:	$(BUILD)/latex/$(OUTPUT_FILENAME).tex

$(BUILD)/epub/$(OUTPUT_FILENAME).epub:	$(EPUB_DEPENDENCIES)
	$(ECHO_BUILDING)
	$(MKDIR_CMD) $(BUILD)/epub
	$(CONTENT) | $(CONTENT_FILTERS) | $(PANDOC_COMMAND) $(ARGS) $(EPUB_ARGS) -o $@
	$(ECHO_BUILT)


$(BUILD)/docx/$(OUTPUT_FILENAME).docx:	$(DOCX_DEPENDENCIES)
	$(ECHO_BUILDING)
	$(MKDIR_CMD) $(BUILD)/docx
	$(CONTENT) | $(CONTENT_FILTERS) | $(PANDOC_COMMAND) $(ARGS) $(DOCX_ARGS) -o $@
	$(ECHO_BUILT)
	
$(BUILD)/html/$(OUTPUT_FILENAME_HTML).html:	$(HTML_DEPENDENCIES)
	$(ECHO_BUILDING)
	$(MKDIR_CMD) $(BUILD)/html
	$(CONTENT) | $(CONTENT_FILTERS) | $(PANDOC_COMMAND) $(ARGS) $(HTML_ARGS) -o $@
	$(COPY_CMD) $(IMAGES) $(BUILD)/html/
	$(COPY_CMD) $(JS_FILES) $(BUILD)/html/
	$(COPY_CMD) $(JS_FILES) $(BUILD)/html/c/  # Copy to nested directory
	$(ECHO_BUILT)


# Rule to build each HTML file from each Markdown file
$(NESTED_HTML_DIR)/%.html: chapters/%.md $(HTML_DEPENDENCIES)
	$(MKDIR_CMD) $(NESTED_HTML_DIR)
	$(PANDOC_COMMAND) $(ARGS) --template $(NESTED_HTML_TEMPLATE) --standalone --to html5 -o $@ $< --mathjax
	@echo "Built HTML for $<"

# Single rule for building all nested HTML
nested_html: $(CHAPTER_HTMLS)
	@echo "All nested HTML files built"


# ArXiv‑compatible LaTeX build rule
$(BUILD)/latex/$(OUTPUT_FILENAME).tex: $(PDF_DEPENDENCIES)
	$(ECHO_BUILDING)
	$(MKDIR_CMD) -p $(BUILD)/latex

	# 1. Generate the LaTeX file with Pandoc (tell Pandoc where to find images)
	$(CONTENT) \
	  | $(CONTENT_FILTERS) \
	  | $(PANDOC_COMMAND) $(ARGS) $(PDF_ARGS) --resource-path=. -o $@

	# 2. Flatten image paths — copy every referenced image into the build dir root
	$(foreach img,$(IMAGES), cp $(img) $(BUILD)/latex/$(notdir $(img));)

	# 3a. Strip directory prefixes in \includegraphics paths
	sed -E -i '' 's|(\\includegraphics(\[[^]]*\])?\{)[^/}]+/|\1|g' $@

	# 3b. Restore missing \includegraphics inside \pandocbounded{}
	perl -CSD -pi -e 's/\\pandocbounded\{([^{}]+)\}\}/\\pandocbounded{\\includegraphics{$$1}}/g' $@

	# 3c. Unicode → ASCII/TeX normalisation (one perl pass per rule for clarity)
	perl -CSD -pi -e 's/\x{2060}//g;'                                        $@  # WORD JOINER
	perl -CSD -pi -e 's/\x{03C4}/\\tau/g;'                                  $@  # τ
	perl -CSD -pi -e 's/[\x{2018}\x{2019}]/\x27/g;'                       $@  # curly apostrophes
	perl -CSD -pi -e 's/[\x{201C}\x{201D}]/\x22/g;'                       $@  # curly quotes
	perl -CSD -pi -e 's/\x{2026}/.../g;'                                    $@  # ellipsis
	perl -CSD -pi -e 's/\x{00A9}/\\textcopyright{}/g;'                    $@  # © symbol

	# 4. Copy bibliography and CSL files required by arXiv
	cp chapters/bib.bib    $(BUILD)/latex/
	cp templates/ieee.csl  $(BUILD)/latex/

	# 5. Warn (but don\'t fail) if any non‑ASCII bytes remain
	@REM_BYTES=$$(grep -nP "[\x80-\xFF]" $@ || true); \
	if [ -n "$$REM_BYTES" ]; then \
	  echo "[WARN] Non‑ASCII bytes still present in $@:"; \
	  echo "$$REM_BYTES" | head; \
	else \
	  echo "[INFO] All bytes ASCII‑safe after post‑processing."; \
	fi

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
	cp $(BUILD)/pdf/book.pdf $(BUILD)/html/ || echo "Failed to copy to $(BUILD)/html/"
	cp -r images $(BUILD)/html/c/ || echo "Failed to copy to $(BUILD)/html/chapters/"
	cp ./templates/nav.js $(BUILD)/html/ || echo "Failed to copy nav.js to $(BUILD)/html/"
	cp ./templates/nav.js $(BUILD)/html/c/ || echo "Failed to copy nav.js to $(BUILD)/html/c/"
	cp ./templates/header-anchors.js $(BUILD)/html/ || echo "Failed to copy header-anchors.js to $(BUILD)/html/"
	cp ./templates/header-anchors.js $(BUILD)/html/c/ || echo "Failed to copy header-anchors.js to $(BUILD)/html/c/"