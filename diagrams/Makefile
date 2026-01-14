# Makefile for generating RLHF Book diagrams
#
# Usage:
#   make all        - Generate all diagrams
#   make d2         - Generate D2 pipeline diagrams only
#   make tokens     - Generate token strip diagrams only
#   make clean      - Remove generated files
#   make install    - Copy final diagrams to images/ directory

GENERATED_DIR := generated
IMAGES_DIR := ../images

# D2 source files
D2_SOURCES := $(wildcard d2/*.d2)
D2_PNGS := $(patsubst d2/%.d2,$(GENERATED_DIR)/%.png,$(D2_SOURCES))
D2_SVGS := $(patsubst d2/%.d2,$(GENERATED_DIR)/%.svg,$(D2_SOURCES))

# Token strip outputs
TOKEN_STRIPS := $(GENERATED_DIR)/pref_rm_tokens.png \
                $(GENERATED_DIR)/orm_tokens.png \
                $(GENERATED_DIR)/prm_tokens.png \
                $(GENERATED_DIR)/value_fn_tokens.png

.PHONY: all d2 tokens clean install help

all: d2 tokens
	@echo "All diagrams generated in $(GENERATED_DIR)/"

# D2 diagrams - requires d2 CLI to be installed
d2: $(D2_PNGS)
	@echo "D2 pipeline diagrams generated"

$(GENERATED_DIR)/%.png: d2/%.d2 | $(GENERATED_DIR)
	@echo "Generating $@..."
	d2 --theme 0 $< $@

$(GENERATED_DIR)/%.svg: d2/%.d2 | $(GENERATED_DIR)
	@echo "Generating $@..."
	d2 --theme 0 $< $@

# Token strip diagrams - requires matplotlib
# Uses two generators:
#   - generate_token_strips.py for Pref RM and PRM (simple strips)
#   - generate_multilane_strips.py for ORM and Value (multi-lane with targets/usage)
# Generates both PNG (digital) and SVG (print)
tokens: | $(GENERATED_DIR)
	uv run python scripts/generate_token_strips.py --output-dir $(GENERATED_DIR) --format png
	uv run python scripts/generate_token_strips.py --output-dir $(GENERATED_DIR) --format svg
	uv run python scripts/generate_multilane_strips.py --output-dir $(GENERATED_DIR) --format png
	uv run python scripts/generate_multilane_strips.py --output-dir $(GENERATED_DIR) --format svg
	@echo "Token strip diagrams generated (PNG + SVG)"

# Create output directory
$(GENERATED_DIR):
	mkdir -p $(GENERATED_DIR)

# Copy diagrams to images/ for use in book
install: all
	@echo "Copying diagrams to $(IMAGES_DIR)/"
	cp $(GENERATED_DIR)/*.png $(IMAGES_DIR)/
	@echo "Done. Diagrams installed to $(IMAGES_DIR)/"

# Clean generated files
clean:
	rm -rf $(GENERATED_DIR)/*
	@echo "Cleaned $(GENERATED_DIR)/"

# Help
help:
	@echo "RLHF Book Diagram Generator"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Generate all diagrams (D2 + token strips)"
	@echo "  d2       - Generate D2 pipeline diagrams only"
	@echo "  tokens   - Generate token strip diagrams only"
	@echo "  install  - Copy generated diagrams to images/"
	@echo "  clean    - Remove generated files"
	@echo ""
	@echo "Requirements:"
	@echo "  - d2 CLI: brew install d2"
	@echo "  - Python + matplotlib: uv add matplotlib"
