#!/bin/bash
# Launch a quick SimPO sanity run in the background.

set -euo pipefail

cd "$(dirname "$0")/.."

ts="$(date +%Y%m%d-%H%M%S)"
export WANDB_PROJECT="${WANDB_PROJECT:-rlhf-book}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-simpo-small-${ts}}"

MAX_SAMPLES="${MAX_SAMPLES:-640}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACC="${GRAD_ACC:-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
BETA="${BETA:-2.0}"
GAMMA="${GAMMA:-0.5}"
SAMPLE_EVERY="${SAMPLE_EVERY:-0}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"

LOG_FILE="/tmp/simpo-small-${ts}.log"

if [ "${RUN_IN_BACKGROUND}" = "1" ]; then
  nohup uv run python -m direct_alignment.train \
    --config direct_alignment/configs/simpo.yaml \
    --max_samples "${MAX_SAMPLES}" \
    --num_epochs "${NUM_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACC}" \
    --max_length "${MAX_LENGTH}" \
    --learning_rate "${LEARNING_RATE}" \
    --beta "${BETA}" \
    --gamma "${GAMMA}" \
    --sample_every "${SAMPLE_EVERY}" \
    > "${LOG_FILE}" 2>&1 &

  PID=$!

  sleep 2
  if ! ps -p "${PID}" >/dev/null 2>&1; then
    echo "SimPO run exited early. Last log lines:"
    tail -n 80 "${LOG_FILE}" || true
    exit 1
  fi

  echo "Started SimPO small run."
  echo "  PID: ${PID}"
  echo "  Log: ${LOG_FILE}"
  echo "  Run: ${WANDB_RUN_NAME}"
  echo
  echo "Monitor with:"
  echo "  tail -f ${LOG_FILE}"
else
  echo "Running SimPO in foreground."
  echo "  Log: ${LOG_FILE}"
  echo "  Run: ${WANDB_RUN_NAME}"
  PYTHONUNBUFFERED=1 uv run python -m direct_alignment.train \
    --config direct_alignment/configs/simpo.yaml \
    --max_samples "${MAX_SAMPLES}" \
    --num_epochs "${NUM_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACC}" \
    --max_length "${MAX_LENGTH}" \
    --learning_rate "${LEARNING_RATE}" \
    --beta "${BETA}" \
    --gamma "${GAMMA}" \
    --sample_every "${SAMPLE_EVERY}" \
    2>&1 | tee "${LOG_FILE}"
fi
