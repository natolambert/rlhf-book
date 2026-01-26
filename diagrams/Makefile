# Makefile for generating RLHF Book diagrams
#
# Usage:
#   make all        - Generate all diagrams
#   make tokens     - Generate token strip diagrams
#   make tikz       - Generate TikZ diagrams
#   make figures    - Generate standalone figure scripts
#   make clean      - Remove generated files

GENERATED_DIR := generated

.PHONY: all tokens tikz figures clean help

# LaTeX engine path (adjust if different on your system)
PDFLATEX := $(shell which pdflatex || echo "/usr/local/texlive/2023basic/bin/universal-darwin/pdflatex")

all: tokens tikz figures
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

# Standalone figure scripts (cartpole, tool_use)
figures: | $(GENERATED_DIR)
	@mkdir -p $(GENERATED_DIR)/png $(GENERATED_DIR)/svg
	uv run python scripts/generate_cartpole.py --output-dir $(GENERATED_DIR)/png --format png
	uv run python scripts/generate_cartpole.py --output-dir $(GENERATED_DIR)/svg --format svg
	uv run python scripts/generate_tool_use.py --output-dir $(GENERATED_DIR)/png --format png
	uv run python scripts/generate_tool_use.py --output-dir $(GENERATED_DIR)/svg --format svg
	@echo "Standalone figures generated (PNG in png/, SVG in svg/)"

# TikZ diagrams - requires pdflatex and convert (ImageMagick) or sips (macOS)
# Compiles .tex files in tikz/ to PDF, then converts to 400dpi PNG
# SVG conversion requires pdf2svg (brew install pdf2svg)
tikz: | $(GENERATED_DIR)
	@mkdir -p $(GENERATED_DIR)/png $(GENERATED_DIR)/svg $(GENERATED_DIR)/pdf
	@for texfile in tikz/*_tikz.tex; do \
		if [ -f "$$texfile" ]; then \
			base=$$(basename "$$texfile" .tex); \
			echo "Compiling $${base}..."; \
			cd tikz && $(PDFLATEX) -interaction=nonstopmode $$(basename $$texfile) > /dev/null 2>&1; cd ..; \
			if [ -f "tikz/$${base}.pdf" ]; then \
				cp tikz/$${base}.pdf $(GENERATED_DIR)/pdf/$${base}.pdf; \
				if command -v magick > /dev/null; then \
					magick -density 400 tikz/$${base}.pdf -trim -quality 100 $(GENERATED_DIR)/png/$${base}.png; \
					echo "  -> $(GENERATED_DIR)/png/$${base}.png (400dpi, trimmed)"; \
				elif command -v convert > /dev/null; then \
					convert -density 400 tikz/$${base}.pdf -trim -quality 100 $(GENERATED_DIR)/png/$${base}.png; \
					echo "  -> $(GENERATED_DIR)/png/$${base}.png (400dpi, trimmed)"; \
				elif command -v sips > /dev/null; then \
					sips -s format png tikz/$${base}.pdf --out $(GENERATED_DIR)/png/$${base}.png > /dev/null 2>&1; \
				fi; \
				if command -v pdf2svg > /dev/null; then \
					pdf2svg tikz/$${base}.pdf $(GENERATED_DIR)/svg/$${base}.svg; \
					echo "  -> $(GENERATED_DIR)/svg/$${base}.svg"; \
				else \
					echo "  (skipping SVG - install pdf2svg: brew install pdf2svg)"; \
				fi; \
			else \
				echo "  ERROR: PDF not generated for $${base}"; \
			fi; \
			rm -f tikz/*.aux tikz/*.log; \
		fi \
	done
	@echo "TikZ diagrams generated in $(GENERATED_DIR)/{pdf,png,svg}/"

# Create output directory
$(GENERATED_DIR):
	mkdir -p $(GENERATED_DIR)

# Clean generated files
clean:
	rm -rf $(GENERATED_DIR)/*
	rm -f tikz/*.pdf tikz/*.aux tikz/*.log
	@echo "Cleaned $(GENERATED_DIR)/ and tikz/*.pdf"

# Help
help:
	@echo "RLHF Book Diagram Generator"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Generate all diagrams"
	@echo "  tokens   - Generate token strip diagrams (reward models)"
	@echo "  tikz     - Generate TikZ diagrams (policy gradients, distillation)"
	@echo "  figures  - Generate standalone figures (cartpole, tool_use)"
	@echo "  clean    - Remove generated files"
	@echo ""
	@echo "Requirements:"
	@echo "  - Python + matplotlib: uv add matplotlib"
	@echo "  - TikZ: pdflatex (TeX Live)"
	@echo "  - SVG export: brew install pdf2svg (optional)"
