# Makefile for generating RLHF Book diagrams
#
# Usage:
#   make all        - Generate all diagrams
#   make tokens     - Generate token strip diagrams
#   make tikz       - Generate TikZ diagrams
#   make training-recipes - Generate the two new training recipes (light + dark)
#   make figures    - Generate standalone figure scripts
#   make clean      - Remove generated files

GENERATED_DIR := generated

.PHONY: all tokens tikz tikz-dark training-recipes dark figures clean help

# LaTeX engine path (adjust if different on your system)
PDFLATEX := $(shell which pdflatex || echo "/usr/local/texlive/2023basic/bin/universal-darwin/pdflatex")

# Shared TikZ styles live here; exposed via TEXINPUTS so sources in any
# topic subfolder can \input{styles_rlhf.tex} without a relative path.
TIKZ_SHARED := $(abspath tikz/_shared)

# Rasterization density for TikZ PNGs. PDFs/SVGs stay vector (resolution-free);
# this only controls the PNG. Override with `make tikz TIKZ_DENSITY=1200`.
TIKZ_DENSITY ?= 800

TRAINING_RECIPE_TEX := tikz/03-training-overview/rlhf_mopd_tikz.tex \
	tikz/03-training-overview/rlhf_sequential_rl_tikz.tex

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

# Dark-mode TikZ variants: compile each *_tikz.tex with \def\figdark{1} so the
# shared/inline styles recolor UNBOUNDED ink (arrows, edge labels, notes) to the
# site's dark text color, then rasterize to a TRANSPARENT PNG <base>-dark.png
# (web/dark-mode only -- no SVG/PDF). Light builds (`make tikz`) are unaffected.
tikz-dark: | $(GENERATED_DIR)
	@mkdir -p $(GENERATED_DIR)/png
	@for texfile in $$(find tikz -name '*_tikz.tex' | sort); do \
		dir=$$(dirname "$$texfile"); \
		base=$$(basename "$$texfile" .tex); \
		echo "Compiling $${base} (dark)..."; \
		( cd "$$dir" && TEXINPUTS=".:$(TIKZ_SHARED):" $(PDFLATEX) -interaction=nonstopmode -jobname "$${base}-dark" "\def\figdark{1}\input{$${base}.tex}" > /dev/null 2>&1 ); \
		if [ -f "$$dir/$${base}-dark.pdf" ]; then \
			if command -v magick > /dev/null; then \
				magick -density $(TIKZ_DENSITY) -background none "$$dir/$${base}-dark.pdf" -trim -quality 100 $(GENERATED_DIR)/png/$${base}-dark.png; \
			elif command -v convert > /dev/null; then \
				convert -density $(TIKZ_DENSITY) -background none "$$dir/$${base}-dark.pdf" -trim -quality 100 $(GENERATED_DIR)/png/$${base}-dark.png; \
			fi; \
			echo "  -> $(GENERATED_DIR)/png/$${base}-dark.png (transparent)"; \
		else \
			echo "  ERROR: dark PDF not generated for $${base}"; \
		fi; \
		rm -f "$$dir/$${base}-dark.aux" "$$dir/$${base}-dark.log" "$$dir/$${base}-dark.pdf"; \
	done
	@echo "Dark TikZ variants generated in $(GENERATED_DIR)/png/"

# Build only the two new training recipes, without regenerating other figures.
# Every TeX intermediate stays under generated/; sources are never compiled in
# place. Light exports are PDF/SVG/PNG; dark exports are transparent PNGs.
training-recipes: $(TRAINING_RECIPE_TEX) | $(GENERATED_DIR)
	@set -eu; \
	builddir="$(abspath $(GENERATED_DIR)/latex/training-recipes)"; \
	mkdir -p "$$builddir" $(GENERATED_DIR)/pdf $(GENERATED_DIR)/png $(GENERATED_DIR)/svg; \
	if command -v magick > /dev/null 2>&1; then raster=magick; \
	elif command -v convert > /dev/null 2>&1; then raster=convert; \
	else echo "ERROR: training-recipes requires ImageMagick (magick or convert)." >&2; exit 1; fi; \
	if command -v pdf2svg > /dev/null 2>&1; then vector=pdf2svg; \
	elif command -v pdftocairo > /dev/null 2>&1; then vector=pdftocairo; \
	else echo "ERROR: training-recipes requires pdf2svg or pdftocairo for SVG export." >&2; exit 1; fi; \
	for texfile in $(TRAINING_RECIPE_TEX); do \
		base=$$(basename "$$texfile" .tex); \
		sourcedir="$$(pwd)/$$(dirname "$$texfile")"; \
		for theme in light dark; do \
			job="$$base"; background=white; alphamode=remove; texinput="\input{$$base.tex}"; \
			if [ "$$theme" = dark ]; then \
				job="$$base-dark"; background=none; alphamode=on; texinput="\def\figdark{1}\input{$$base.tex}"; \
			fi; \
			echo "Compiling $$job ($$theme)..."; \
			if ! TEXINPUTS=".:$$sourcedir:$(TIKZ_SHARED):" $(PDFLATEX) \
				-interaction=nonstopmode -halt-on-error -file-line-error \
				-output-directory="$$builddir" -jobname="$$job" "$$texinput" \
				> "$$builddir/$$job.build.log" 2>&1; then \
				cat "$$builddir/$$job.build.log" >&2; exit 1; \
			fi; \
			"$$raster" -density $(TIKZ_DENSITY) -background "$$background" \
				"$$builddir/$$job.pdf" -trim +repage -alpha "$$alphamode" -quality 100 "$(GENERATED_DIR)/png/$$job.png"; \
			if [ "$$theme" = light ]; then \
				cp "$$builddir/$$job.pdf" "$(GENERATED_DIR)/pdf/$$job.pdf"; \
				if [ "$$vector" = pdf2svg ]; then \
					pdf2svg "$$builddir/$$job.pdf" "$(GENERATED_DIR)/svg/$$job.svg"; \
				else \
					pdftocairo -svg "$$builddir/$$job.pdf" "$(GENERATED_DIR)/svg/$$job.svg"; \
				fi; \
			fi; \
		done; \
	done
	@echo "Training recipes generated in $(GENERATED_DIR)/{pdf,png,svg}/ (including dark PNGs)"

# All dark-mode variants (matplotlib + TikZ): transparent PNGs for the web.
dark: | $(GENERATED_DIR)
	@mkdir -p $(GENERATED_DIR)/png
	uv run python scripts/generate_token_strips.py --output-dir $(GENERATED_DIR)/png --format png --theme dark
	uv run python scripts/generate_multilane_strips.py --output-dir $(GENERATED_DIR)/png --format png --theme dark
	uv run python scripts/generate_cartpole.py --output-dir $(GENERATED_DIR)/png --format png --theme dark
	uv run python scripts/generate_tool_use.py --output-dir $(GENERATED_DIR)/png --format png --theme dark
	$(MAKE) tikz-dark
	@echo "All dark-mode variants generated in $(GENERATED_DIR)/png/"

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
	@echo "  training-recipes - Generate MOPD + sequential RL recipes (light PDF/SVG/PNG, dark PNG)"
	@echo "  figures  - Generate standalone figures (cartpole, tool_use, interleaved_thinking)"
	@echo "  clean    - Remove generated files"
	@echo ""
	@echo "Requirements:"
	@echo "  - Python + matplotlib: uv add matplotlib"
	@echo "  - TikZ: pdflatex (TeX Live)"
	@echo "  - SVG export: brew install pdf2svg (optional)"
	@echo "  - training-recipes: ImageMagick + either pdf2svg or pdftocairo (Poppler)"
	@echo "  - Override TikZ PNG density with TIKZ_DENSITY=1200 (default: 800 dpi)"
