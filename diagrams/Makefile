# Makefile for generating RLHF Book diagrams
#
# Usage:
#   make all        - Generate all diagrams
#   make tokens     - Generate token strip diagrams
#   make clean      - Remove generated files

GENERATED_DIR := generated

.PHONY: all tokens grpo tikz clean help

# LaTeX engine path (adjust if different on your system)
PDFLATEX := $(shell which pdflatex || echo "/usr/local/texlive/2023basic/bin/universal-darwin/pdflatex")

all: tokens grpo tikz
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

# PPO vs GRPO architecture comparison diagram (matplotlib version)
# Based on DeepSeek-Math/GRPO paper Figure 1
grpo: | $(GENERATED_DIR)
	@mkdir -p $(GENERATED_DIR)/png $(GENERATED_DIR)/svg
	uv run python scripts/generate_ppo_grpo.py --output-dir $(GENERATED_DIR)/png --format png
	uv run python scripts/generate_ppo_grpo.py --output-dir $(GENERATED_DIR)/svg --format svg
	@echo "PPO vs GRPO diagram generated (matplotlib)"

# TikZ diagrams - requires pdflatex and sips (macOS) or convert (ImageMagick)
# Compiles .tex files in tikz/ to PDF, then converts to PNG
# SVG conversion requires pdf2svg (brew install pdf2svg)
tikz: | $(GENERATED_DIR)
	@mkdir -p $(GENERATED_DIR)/png $(GENERATED_DIR)/svg
	@for texfile in tikz/*_test.tex; do \
		if [ -f "$$texfile" ]; then \
			base=$$(basename "$$texfile" _test.tex); \
			echo "Compiling $$base..."; \
			cd tikz && $(PDFLATEX) -interaction=nonstopmode $$(basename $$texfile) > /dev/null 2>&1; cd ..; \
			cp tikz/$${base}_test.pdf $(GENERATED_DIR)/png/$${base}.pdf; \
			if command -v sips > /dev/null; then \
				sips -s format png tikz/$${base}_test.pdf --out $(GENERATED_DIR)/png/$${base}.png > /dev/null 2>&1; \
			elif command -v convert > /dev/null; then \
				convert -density 300 tikz/$${base}_test.pdf $(GENERATED_DIR)/png/$${base}.png; \
			fi; \
			if command -v pdf2svg > /dev/null; then \
				pdf2svg tikz/$${base}_test.pdf $(GENERATED_DIR)/svg/$${base}.svg; \
			else \
				echo "  (skipping SVG - install pdf2svg: brew install pdf2svg)"; \
			fi; \
			rm -f tikz/*.aux tikz/*.log; \
		fi \
	done
	@echo "TikZ diagrams generated"

# Create output directory
$(GENERATED_DIR):
	mkdir -p $(GENERATED_DIR)

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
	@echo "  grpo     - Generate PPO vs GRPO diagram (matplotlib)"
	@echo "  tikz     - Generate TikZ diagrams (ppo_grpo_tikz, etc.)"
	@echo "  clean    - Remove generated files"
	@echo ""
	@echo "Requirements:"
	@echo "  - Python + matplotlib: uv add matplotlib"
	@echo "  - TikZ: pdflatex (TeX Live)"
	@echo "  - SVG export: brew install pdf2svg (optional)"
