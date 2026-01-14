# Makefile for generating RLHF Book diagrams
#
# Usage:
#   make all        - Generate all diagrams
#   make tokens     - Generate token strip diagrams
#   make clean      - Remove generated files

GENERATED_DIR := generated

.PHONY: all tokens grpo clean help

all: tokens grpo
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

# PPO vs GRPO architecture comparison diagram
# Based on DeepSeek-Math/GRPO paper Figure 1
grpo: | $(GENERATED_DIR)
	@mkdir -p $(GENERATED_DIR)/png $(GENERATED_DIR)/svg
	uv run python scripts/generate_ppo_grpo.py --output-dir $(GENERATED_DIR)/png --format png
	uv run python scripts/generate_ppo_grpo.py --output-dir $(GENERATED_DIR)/svg --format svg
	@echo "PPO vs GRPO diagram generated"

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
	@echo "  grpo     - Generate PPO vs GRPO architecture diagram"
	@echo "  clean    - Remove generated files"
	@echo ""
	@echo "Requirements:"
	@echo "  - Python + matplotlib: uv add matplotlib"
