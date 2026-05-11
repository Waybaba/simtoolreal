#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_NAME="${RUN_NAME:-isaacgym_repro_3090_video}"
LOG_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-outputs/${RUN_NAME}_${LOG_STAMP}.log}"

mkdir -p outputs

# Old Isaac Gym is safest on the RTX 3090. With CUDA_VISIBLE_DEVICES=0, the
# process sees the physical 3090 as cuda:0.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
PYTHON_DIR="$(dirname "${PYTHON_BIN}")"
export PATH="$(cd "${PYTHON_DIR}" && pwd):${PATH}"

NUM_ENVS="${NUM_ENVS:-12288}"
NUM_BLOCKS="${NUM_BLOCKS:-6}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
HORIZON_LENGTH="${HORIZON_LENGTH:-}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-98304}"
MINI_EPOCHS="${MINI_EPOCHS:-}"
SAVE_FREQUENCY="${SAVE_FREQUENCY:-}"
SEQ_LENGTH="${SEQ_LENGTH:-}"

CAPTURE_VIDEO="${CAPTURE_VIDEO:-true}"
CAPTURE_VIDEO_FREQ="${CAPTURE_VIDEO_FREQ:-6000}"
CAPTURE_VIDEO_LEN="${CAPTURE_VIDEO_LEN:-600}"

WANDB_ACTIVATE="${WANDB_ACTIVATE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-simtoolreal}"
WANDB_ENTITY="${WANDB_ENTITY:-waybabag}"
WANDB_GROUP="${WANDB_GROUP:-$(date +%Y-%m-%d)}"

EXTRA_ARGS=()
if [[ -n "${MAX_EPOCHS}" ]]; then
  EXTRA_ARGS+=(--max-epochs "${MAX_EPOCHS}")
fi
if [[ -n "${HORIZON_LENGTH}" ]]; then
  EXTRA_ARGS+=(--horizon-length "${HORIZON_LENGTH}")
fi
if [[ -n "${MINIBATCH_SIZE}" ]]; then
  EXTRA_ARGS+=(--minibatch-size "${MINIBATCH_SIZE}")
fi
if [[ -n "${MINI_EPOCHS}" ]]; then
  EXTRA_ARGS+=(--mini-epochs "${MINI_EPOCHS}")
fi
if [[ -n "${SAVE_FREQUENCY}" ]]; then
  EXTRA_ARGS+=(--save-frequency "${SAVE_FREQUENCY}")
fi
if [[ -n "${SEQ_LENGTH}" ]]; then
  EXTRA_ARGS+=(--seq-length "${SEQ_LENGTH}")
fi

if [[ "${CAPTURE_VIDEO}" == "true" ]]; then
  EXTRA_ARGS+=(--capture-video)
else
  EXTRA_ARGS+=(--no-capture-video)
fi

if [[ "${WANDB_ACTIVATE}" == "true" ]]; then
  EXTRA_ARGS+=(--wandb-activate)
else
  EXTRA_ARGS+=(--no-wandb-activate)
fi

echo "Starting ${RUN_NAME}"
echo "GPU: physical 0 / RTX 3090 via CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Log: ${LOG_PATH}"

"${PYTHON_BIN}" isaacgymenvs/launch_training.py \
  --custom-experiment-name "${RUN_NAME}" \
  --num-envs "${NUM_ENVS}" \
  --num-blocks "${NUM_BLOCKS}" \
  --capture-video-freq "${CAPTURE_VIDEO_FREQ}" \
  --capture-video-len "${CAPTURE_VIDEO_LEN}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-group "${WANDB_GROUP}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_PATH}"
