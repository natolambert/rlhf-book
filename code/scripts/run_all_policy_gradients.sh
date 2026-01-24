#!/bin/bash
# Run all policy gradient algorithms sequentially
# Each produces ~1250 wandb logging steps

set -e

cd "$(dirname "$0")/.."
export WANDB_PROJECT="${WANDB_PROJECT:-rlhf-book}"

echo "Starting policy gradient training runs..."
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo ""

# RLOO
echo "=========================================="
echo "Running RLOO..."
echo "=========================================="
uv run python -m policy_gradients.train --config policy_gradients/configs/rloo.yaml

# PPO
echo "=========================================="
echo "Running PPO..."
echo "=========================================="
uv run python -m policy_gradients.train --config policy_gradients/configs/ppo.yaml

# Dr. GRPO
echo "=========================================="
echo "Running Dr. GRPO..."
echo "=========================================="
uv run python -m policy_gradients.train --config policy_gradients/configs/drgrpo.yaml

# GSPO
echo "=========================================="
echo "Running GSPO..."
echo "=========================================="
uv run python -m policy_gradients.train --config policy_gradients/configs/gspo.yaml

# CISPO
echo "=========================================="
echo "Running CISPO..."
echo "=========================================="
uv run python -m policy_gradients.train --config policy_gradients/configs/cispo.yaml

echo ""
echo "=========================================="
echo "All runs complete!"
echo "Check wandb project: https://wandb.ai/natolambert/$WANDB_PROJECT"
echo "=========================================="
