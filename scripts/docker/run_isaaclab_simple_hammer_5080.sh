#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

RUN_NAME="${RUN_NAME:-isaaclab_simple_hammer_5080_$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="${LOG_PATH:-outputs/${RUN_NAME}.log}"

mkdir -p outputs

NUM_ENVS="${NUM_ENVS:-8190}"
NUM_STEPS="${NUM_STEPS:-16}"
SEQ_LENGTH="${SEQ_LENGTH:-16}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-131040}"
TOTAL_UPDATES="${TOTAL_UPDATES:-1000000}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-2}"
SAVE_FREQUENCY="${SAVE_FREQUENCY:-3000}"
CHECKPOINT="${CHECKPOINT:-}"

CAPTURE_VIDEO="${CAPTURE_VIDEO:-false}"
CAPTURE_VIDEO_FREQ="${CAPTURE_VIDEO_FREQ:-18000}"
CAPTURE_VIDEO_LEN="${CAPTURE_VIDEO_LEN:-300}"
TRAJECTORY_LOG="${TRAJECTORY_LOG:-true}"
TRAJECTORY_LOG_MAX_ENVS="${TRAJECTORY_LOG_MAX_ENVS:-4}"
TRAJECTORY_LOG_VIDEO_LEN_MULTIPLIER="${TRAJECTORY_LOG_VIDEO_LEN_MULTIPLIER:-2}"

WANDB="${WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-simtoolreal}"
WANDB_ENTITY="${WANDB_ENTITY:-waybabag}"
WANDB_GROUP="${WANDB_GROUP:-$(date +%Y-%m-%d)}"

LEARNING_RATE="${LEARNING_RATE:-1.0e-4}"
KL_THRESHOLD="${KL_THRESHOLD:-0.016}"
ADAPTIVE_LR="${ADAPTIVE_LR:-true}"
FORCE_SCALE="${FORCE_SCALE:-20.0}"
TORQUE_SCALE="${TORQUE_SCALE:-2.0}"
FORCE_CONSECUTIVE_NEAR_GOAL_STEPS="${FORCE_CONSECUTIVE_NEAR_GOAL_STEPS:-true}"
USE_OTHERS_EXPERIENCE="${USE_OTHERS_EXPERIENCE:-lf}"
OFF_POLICY_RATIO="${OFF_POLICY_RATIO:-1.0}"
EASY_GOAL_DEBUG="${EASY_GOAL_DEBUG:-false}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -x /workspace/isaaclab/isaaclab.sh ]]; then
  PYTHON_RUN=(/workspace/isaaclab/isaaclab.sh -p -u)
elif command -v conda >/dev/null 2>&1; then
  PYTHON_RUN=(conda run --no-capture-output -n isaaclab510 python -u)
else
  PYTHON_RUN=(python -u)
fi

VIDEO_ARG="--no-capture_video"
if [[ "${CAPTURE_VIDEO}" == "true" ]]; then
  VIDEO_ARG="--capture_video"
fi

TRAJECTORY_ARG="--no-trajectory_log"
if [[ "${TRAJECTORY_LOG}" == "true" ]]; then
  TRAJECTORY_ARG="--trajectory_log"
fi

WANDB_ARG="--no-wandb"
if [[ "${WANDB}" == "true" ]]; then
  WANDB_ARG="--wandb"
fi

ADAPTIVE_LR_ARG="--no-adaptive_lr"
if [[ "${ADAPTIVE_LR}" == "true" ]]; then
  ADAPTIVE_LR_ARG="--adaptive_lr"
fi

CHECKPOINT_ARGS=()
if [[ -n "${CHECKPOINT}" ]]; then
  CHECKPOINT_ARGS=(--checkpoint "${CHECKPOINT}")
fi

FORCE_CONSECUTIVE_ARG="--no-force_consecutive_near_goal_steps"
if [[ "${FORCE_CONSECUTIVE_NEAR_GOAL_STEPS}" == "true" ]]; then
  FORCE_CONSECUTIVE_ARG="--force_consecutive_near_goal_steps"
fi

EASY_GOAL_ARGS=()
if [[ "${EASY_GOAL_DEBUG}" == "true" ]]; then
  EASY_GOAL_ARGS=(--easy_goal_debug)
fi

echo "Starting ${RUN_NAME}"
echo "Simplification: one fixed hammer asset, no object size randomization, no delay/noise"
echo "GPU: cuda:0 inside the runtime; Docker wrapper maps this to physical RTX 5080"
echo "Log: ${LOG_PATH}"
if [[ -n "${CHECKPOINT}" ]]; then
  echo "Checkpoint: ${CHECKPOINT}"
fi

"${PYTHON_RUN[@]}" scripts/train_simtoolreal_cleanrl.py \
  --task SimToolReal-Direct-v0 \
  --num_envs "${NUM_ENVS}" \
  --num_steps "${NUM_STEPS}" \
  --seq_length "${SEQ_LENGTH}" \
  --minibatch_size "${MINIBATCH_SIZE}" \
  --total_updates "${TOTAL_UPDATES}" \
  --update_epochs "${UPDATE_EPOCHS}" \
  --save_frequency "${SAVE_FREQUENCY}" \
  --learning_rate "${LEARNING_RATE}" \
  --kl_threshold "${KL_THRESHOLD}" \
  "${ADAPTIVE_LR_ARG}" \
  --use_others_experience "${USE_OTHERS_EXPERIENCE}" \
  --off_policy_ratio "${OFF_POLICY_RATIO}" \
  "${CHECKPOINT_ARGS[@]}" \
  "${WANDB_ARG}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_group "${WANDB_GROUP}" \
  "${VIDEO_ARG}" \
  --capture_video_freq "${CAPTURE_VIDEO_FREQ}" \
  --capture_video_len "${CAPTURE_VIDEO_LEN}" \
  "${TRAJECTORY_ARG}" \
  --trajectory_log_selection first_line \
  --trajectory_log_max_envs "${TRAJECTORY_LOG_MAX_ENVS}" \
  --trajectory_log_video_len_multiplier "${TRAJECTORY_LOG_VIDEO_LEN_MULTIPLIER}" \
  --no-trajectory_log_include_obs \
  --simple_hammer_debug \
  "${EASY_GOAL_ARGS[@]}" \
  --force_scale "${FORCE_SCALE}" \
  --torque_scale "${TORQUE_SCALE}" \
  "${FORCE_CONSECUTIVE_ARG}" \
  --device cuda:0 \
  --env_device cuda:0 \
  --kit_active_gpu 0 \
  --kit_physics_gpu 0 \
  --run_name "${RUN_NAME}" \
  2>&1 | tee "${LOG_PATH}"
