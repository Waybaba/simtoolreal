#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN_NAME="${RUN_NAME:-full_isaaclab_cleanrl_5080_8190_headless_traj_$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="${LOG_PATH:-outputs/${RUN_NAME}.log}"

mkdir -p outputs

# PyTorch cuda:0 maps to the RTX 5080 on this machine.
NUM_ENVS="${NUM_ENVS:-8190}"
NUM_STEPS="${NUM_STEPS:-16}"
SEQ_LENGTH="${SEQ_LENGTH:-16}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-131040}"
TOTAL_UPDATES="${TOTAL_UPDATES:-1000000}"

CAPTURE_VIDEO_FREQ="${CAPTURE_VIDEO_FREQ:-18000}"
CAPTURE_VIDEO_LEN="${CAPTURE_VIDEO_LEN:-300}"
TRAJECTORY_LOG_MAX_ENVS="${TRAJECTORY_LOG_MAX_ENVS:-4}"
TRAJECTORY_LOG_VIDEO_LEN_MULTIPLIER="${TRAJECTORY_LOG_VIDEO_LEN_MULTIPLIER:-2}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

nohup conda run --no-capture-output -n isaaclab510 python -u scripts/train_simtoolreal_cleanrl.py \
  --task SimToolReal-Direct-v0 \
  --num_envs "${NUM_ENVS}" \
  --num_steps "${NUM_STEPS}" \
  --seq_length "${SEQ_LENGTH}" \
  --minibatch_size "${MINIBATCH_SIZE}" \
  --total_updates "${TOTAL_UPDATES}" \
  --wandb \
  --no-capture_video \
  --capture_video_freq "${CAPTURE_VIDEO_FREQ}" \
  --capture_video_len "${CAPTURE_VIDEO_LEN}" \
  --trajectory_log \
  --trajectory_log_selection first_line \
  --trajectory_log_max_envs "${TRAJECTORY_LOG_MAX_ENVS}" \
  --trajectory_log_video_len_multiplier "${TRAJECTORY_LOG_VIDEO_LEN_MULTIPLIER}" \
  --no-trajectory_log_include_obs \
  --device cuda:0 \
  --env_device cuda:0 \
  --kit_active_gpu 1 \
  --kit_physics_gpu 0 \
  --run_name "${RUN_NAME}" \
  > "${LOG_PATH}" 2>&1 &

PID=$!

echo "Started ${RUN_NAME}"
echo "PID: ${PID}"
echo "Log: ${LOG_PATH}"
echo "Tail: tail -f ${LOG_PATH}"
