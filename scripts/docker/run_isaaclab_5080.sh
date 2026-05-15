#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

IMAGE_NAME="${IMAGE_NAME:-simtoolreal-isaaclab:latest}"
GPU="1"
NAME="${NAME:-simtoolreal-isaaclab-5080}"
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
SHM_SIZE="${SHM_SIZE:-16g}"
DETACH="${DETACH:-false}"
mkdir -p \
  "${REPO_ROOT}/outputs" \
  "${REPO_ROOT}/outputs/wandb" \
  "${REPO_ROOT}/outputs/wandb_artifacts" \
  "${REPO_ROOT}/outputs/wandb_cache" \
  "${REPO_ROOT}/outputs/wandb_data"

TOTAL_CPUS="$(nproc)"
DEFAULT_CPU_COUNT="$((TOTAL_CPUS / 2))"
if [[ "${DEFAULT_CPU_COUNT}" -lt 1 ]]; then
  DEFAULT_CPU_COUNT=1
fi
DEFAULT_CPU_FIRST="${DEFAULT_CPU_COUNT}"
DEFAULT_CPU_LAST="$((TOTAL_CPUS - 1))"
if [[ "${DEFAULT_CPU_FIRST}" -gt "${DEFAULT_CPU_LAST}" ]]; then
  DEFAULT_CPU_FIRST=0
fi
DEFAULT_CPUSET_CPUS="${DEFAULT_CPU_FIRST}-${DEFAULT_CPU_LAST}"

MEM_TOTAL_KB="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
DEFAULT_MEMORY_MIB="$((MEM_TOTAL_KB / 2048))"
DEFAULT_MEMORY="${DEFAULT_MEMORY_MIB}m"

CPUSET_CPUS="${CPUSET_CPUS:-${DEFAULT_CPUSET_CPUS}}"
MEMORY="${MEMORY:-${DEFAULT_MEMORY}}"
MEMORY_SWAP="${MEMORY_SWAP:-${MEMORY}}"

if [[ "${GPU_OVERRIDE:-}" != "" && "${GPU_OVERRIDE}" != "1" ]]; then
  echo "Isaac Lab Docker defaults to the RTX 5080 on this machine. Use GPU_OVERRIDE=1 or unset it." >&2
  exit 1
fi

echo "Isaac Lab Docker target: physical GPU 1 / RTX 5080"
echo "CPU limit: ${CPUSET_CPUS} (override with CPUSET_CPUS=...)"
echo "Memory limit: ${MEMORY} (override with MEMORY=...)"
echo "Detached: ${DETACH} (override with DETACH=true)"

DOCKER_ARGS=(
  --name "${NAME}"
  --entrypoint ""
  --gpus "device=${GPU}"
  --ipc=host
  --shm-size "${SHM_SIZE}"
  --ulimit memlock=-1
  --ulimit stack=67108864
  --network host
  --cpuset-cpus "${CPUSET_CPUS}"
  --memory "${MEMORY}"
  --memory-swap "${MEMORY_SWAP}"
  -e ACCEPT_EULA=Y
  -e OMNI_KIT_ALLOW_ROOT=1
  -e NVIDIA_VISIBLE_DEVICES="${GPU}"
  -e NVIDIA_DRIVER_CAPABILITIES=all
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID
  -e CUDA_VISIBLE_DEVICES=0
  -e WANDB_DIR=/workspace/simtoolreal/outputs/wandb
  -e WANDB_ARTIFACT_DIR=/workspace/simtoolreal/outputs/wandb_artifacts
  -e WANDB_CACHE_DIR=/workspace/simtoolreal/outputs/wandb_cache
  -e WANDB_DATA_DIR=/workspace/simtoolreal/outputs/wandb_data
  -v "${REPO_ROOT}:/workspace/simtoolreal"
  -w /workspace/simtoolreal
)

if [[ "${DETACH}" == "true" ]]; then
  DOCKER_ARGS=(-d "${DOCKER_ARGS[@]}")
else
  DOCKER_ARGS=(--rm "${DOCKER_ARGS[@]}")
fi

FORWARDED_ENV_VARS=(
  RUN_NAME
  LOG_PATH
  NUM_ENVS
  NUM_STEPS
  SEQ_LENGTH
  MINIBATCH_SIZE
  TOTAL_UPDATES
  UPDATE_EPOCHS
  SAVE_FREQUENCY
  CHECKPOINT
  LEARNING_RATE
  KL_THRESHOLD
  ADAPTIVE_LR
  FORCE_SCALE
  TORQUE_SCALE
  FORCE_CONSECUTIVE_NEAR_GOAL_STEPS
  USE_OTHERS_EXPERIENCE
  OFF_POLICY_RATIO
  EASY_GOAL_DEBUG
  CAPTURE_VIDEO
  CAPTURE_VIDEO_FREQ
  CAPTURE_VIDEO_LEN
  TRAJECTORY_LOG
  TRAJECTORY_LOG_MAX_ENVS
  TRAJECTORY_LOG_VIDEO_LEN_MULTIPLIER
  WANDB
  WANDB_PROJECT
  WANDB_ENTITY
  WANDB_GROUP
  WANDB_API_KEY
  WANDB_MODE
)

for env_var in "${FORWARDED_ENV_VARS[@]}"; do
  if [[ -v "${env_var}" ]]; then
    DOCKER_ARGS+=(-e "${env_var}=${!env_var}")
  fi
done

TTY_ARGS=()
if [[ "${DETACH}" != "true" && -t 0 && -t 1 ]]; then
  TTY_ARGS=(-it)
fi

if [[ -f "${HOME}/.netrc" ]]; then
  DOCKER_ARGS+=(-v "${HOME}/.netrc:/home/simuser/.netrc:ro")
fi

if [[ -d "${HOME}/.config/wandb" ]]; then
  DOCKER_ARGS+=(-v "${HOME}/.config/wandb:/home/simuser/.config/wandb:ro")
fi

if [[ $# -eq 0 ]]; then
  docker run "${TTY_ARGS[@]}" "${DOCKER_ARGS[@]}" "${IMAGE_NAME}" /bin/bash
else
  docker run "${TTY_ARGS[@]}" "${DOCKER_ARGS[@]}" "${IMAGE_NAME}" "$@"
fi
