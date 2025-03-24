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

.PHONY: all book clean epub html pdf docx nested_html

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

html:	$(BUILD)/html/$(OUTPUT_FILENAME_HTML).html nested_html

nested_html: $(CHAPTER_HTMLS)
	$(ECHO_BUILT)
	
pdf:	$(BUILD)/pdf/$(OUTPUT_FILENAME).pdf

docx:	$(BUILD)/docx/$(OUTPUT_FILENAME).docx

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
	$(COPY_CMD) $(IMAGES) $(BUILD)/html/ --mathjax
	$(COPY_CMD) $(JS_FILES) $(BUILD)/html/
	$(COPY_CMD) $(JS_FILES) $(BUILD)/html/c/  # Copy to nested directory
	$(ECHO_BUILT)

# Nested HTML build targets
NESTED_HTML_DIR = $(BUILD)/html/c/
CHAPTER_HTMLS = $(patsubst chapters/%.md,$(NESTED_HTML_DIR)/%.html,$(CHAPTERS))

# Rule to build each HTML file from each Markdown file
$(NESTED_HTML_DIR)/%.html: chapters/%.md $(HTML_DEPENDENCIES)
	$(MKDIR_CMD) $(NESTED_HTML_DIR)
	$(PANDOC_COMMAND) $(ARGS) --template $(NESTED_HTML_TEMPLATE) --standalone --to html5 -o $@ $< --mathjax
	@echo "Built HTML for $<"

# Single rule for building all nested HTML
nested_html: $(CHAPTER_HTMLS)
	@echo "All nested HTML files built"

# Main HTML target
$(BUILD)/html/$(OUTPUT_FILENAME_HTML).html: nested_html
	$(MKDIR_CMD) $(BUILD)/html
	$(CONTENT) | $(CONTENT_FILTERS) | $(PANDOC_COMMAND) $(ARGS) $(HTML_ARGS) -o $@
	$(COPY_CMD) $(IMAGES) $(BUILD)/html/
	@echo "Main HTML index built"

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