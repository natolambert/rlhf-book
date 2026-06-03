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

# Shared TikZ styles live here; exposed via TEXINPUTS so sources in any
# topic subfolder can \input{styles_rlhf.tex} without a relative path.
TIKZ_SHARED := $(abspath tikz/_shared)

# Rasterization density for TikZ PNGs. PDFs/SVGs stay vector (resolution-free);
# this only controls the PNG. Override with `make tikz TIKZ_DENSITY=1200`.
TIKZ_DENSITY ?= 800

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
	uv run python scripts/generate_interleaved_thinking.py --output-dir $(GENERATED_DIR)/png --format png
	uv run python scripts/generate_interleaved_thinking.py --output-dir $(GENERATED_DIR)/svg --format svg
	@echo "Standalone figures generated (PNG in png/, SVG in svg/)"

# TikZ diagrams - requires pdflatex and convert (ImageMagick) or sips (macOS)
# Sources are organized into topic subfolders under tikz/ (e.g. policy-gradients/).
# Each *_tikz.tex is compiled in place to PDF, then converted to a PNG at $(TIKZ_DENSITY) dpi.
# Shared styles are found via TEXINPUTS (see TIKZ_SHARED). Output basenames stay
# flat, so generated/png/<base>.png is unchanged by the folder reorganization.
# SVG conversion requires pdf2svg (brew install pdf2svg)
tikz: | $(GENERATED_DIR)
	@mkdir -p $(GENERATED_DIR)/png $(GENERATED_DIR)/svg $(GENERATED_DIR)/pdf
	@for texfile in $$(find tikz -name '*_tikz.tex' | sort); do \
		dir=$$(dirname "$$texfile"); \
		base=$$(basename "$$texfile" .tex); \
		echo "Compiling $${base} ($$dir)..."; \
		( cd "$$dir" && TEXINPUTS=".:$(TIKZ_SHARED):" $(PDFLATEX) -interaction=nonstopmode "$${base}.tex" > /dev/null 2>&1 ); \
		if [ -f "$$dir/$${base}.pdf" ]; then \
			cp "$$dir/$${base}.pdf" $(GENERATED_DIR)/pdf/$${base}.pdf; \
			if command -v magick > /dev/null; then \
				magick -density $(TIKZ_DENSITY) "$$dir/$${base}.pdf" -trim -quality 100 $(GENERATED_DIR)/png/$${base}.png; \
				echo "  -> $(GENERATED_DIR)/png/$${base}.png ($(TIKZ_DENSITY)dpi, trimmed)"; \
			elif command -v convert > /dev/null; then \
				convert -density $(TIKZ_DENSITY) "$$dir/$${base}.pdf" -trim -quality 100 $(GENERATED_DIR)/png/$${base}.png; \
				echo "  -> $(GENERATED_DIR)/png/$${base}.png ($(TIKZ_DENSITY)dpi, trimmed)"; \
			elif command -v sips > /dev/null; then \
				sips -s format png "$$dir/$${base}.pdf" --out $(GENERATED_DIR)/png/$${base}.png > /dev/null 2>&1; \
			fi; \
			if command -v pdf2svg > /dev/null; then \
				pdf2svg "$$dir/$${base}.pdf" $(GENERATED_DIR)/svg/$${base}.svg; \
				echo "  -> $(GENERATED_DIR)/svg/$${base}.svg"; \
			else \
				echo "  (skipping SVG - install pdf2svg: brew install pdf2svg)"; \
			fi; \
		else \
			echo "  ERROR: PDF not generated for $${base}"; \
		fi; \
		rm -f "$$dir/$${base}.aux" "$$dir/$${base}.log" "$$dir/$${base}.pdf"; \
	done
	@echo "TikZ diagrams generated in $(GENERATED_DIR)/{pdf,png,svg}/"

# Create output directory
$(GENERATED_DIR):
	mkdir -p $(GENERATED_DIR)

# Clean generated files
clean:
	rm -rf $(GENERATED_DIR)/*
	find tikz \( -name '*.pdf' -o -name '*.aux' -o -name '*.log' \) -delete
	@echo "Cleaned $(GENERATED_DIR)/ and tikz build artifacts"

# Help
help:
	@echo "RLHF Book Diagram Generator"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Generate all diagrams"
	@echo "  tokens   - Generate token strip diagrams (reward models)"
	@echo "  tikz     - Generate TikZ diagrams (policy gradients, distillation)"
	@echo "  figures  - Generate standalone figures (cartpole, tool_use, interleaved_thinking)"
	@echo "  clean    - Remove generated files"
	@echo ""
	@echo "Requirements:"
	@echo "  - Python + matplotlib: uv add matplotlib"
	@echo "  - TikZ: pdflatex (TeX Live)"
	@echo "  - SVG export: brew install pdf2svg (optional)"
