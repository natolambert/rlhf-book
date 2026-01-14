# Makefile for generating RLHF Book diagrams
#
# Usage:
#   make all        - Generate all diagrams
#   make tokens     - Generate token strip diagrams
#   make clean      - Remove generated files
#   make install    - Copy final diagrams to images/ directory

GENERATED_DIR := generated
IMAGES_DIR := ../images

.PHONY: all tokens clean install help

all: tokens
	@echo "All diagrams generated in $(GENERATED_DIR)/"

# Token strip diagrams - requires matplotlib
# Uses two generators:
#   - generate_token_strips.py for Pref RM and PRM (simple strips)
#   - generate_multilane_strips.py for ORM and Value (multi-lane with targets/usage)
# Generates both PNG (digital) and SVG (print) in separate folders
tokens: | $(GENERATED_DIR)
	@mkdir -p $(GENERATED_DIR)/png $(GENERATED_DIR)/svg
	uv run python scripts/generate_token_strips.py --output-dir $(GENERATED_DIR)/png --format png
	uv run python scripts/generate_token_strips.py --output-dir $(GENERATED_DIR)/svg --format svg
	uv run python scripts/generate_multilane_strips.py --output-dir $(GENERATED_DIR)/png --format png
	uv run python scripts/generate_multilane_strips.py --output-dir $(GENERATED_DIR)/svg --format svg
	@echo "Token strip diagrams generated (PNG in png/, SVG in svg/)"

# Create output directory
$(GENERATED_DIR):
	mkdir -p $(GENERATED_DIR)

# Copy diagrams to images/ for use in book
install: all
	@echo "Copying diagrams to $(IMAGES_DIR)/"
	cp $(GENERATED_DIR)/png/*.png $(IMAGES_DIR)/
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
	@echo "  all      - Generate all diagrams"
	@echo "  tokens   - Generate token strip diagrams"
	@echo "  install  - Copy generated diagrams to images/"
	@echo "  clean    - Remove generated files"
	@echo ""
	@echo "Requirements:"
	@echo "  - Python + matplotlib: uv add matplotlib"
