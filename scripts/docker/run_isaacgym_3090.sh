#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

IMAGE_NAME="${IMAGE_NAME:-simtoolreal-isaacgym:latest}"
GPU="0"
NAME="${NAME:-simtoolreal-isaacgym-3090}"
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
SHM_SIZE="${SHM_SIZE:-16g}"
mkdir -p "${REPO_ROOT}/outputs"

TOTAL_CPUS="$(nproc)"
DEFAULT_CPU_COUNT="$((TOTAL_CPUS / 2))"
if [[ "${DEFAULT_CPU_COUNT}" -lt 1 ]]; then
  DEFAULT_CPU_COUNT=1
fi
DEFAULT_CPU_LAST="$((DEFAULT_CPU_COUNT - 1))"
DEFAULT_CPUSET_CPUS="0-${DEFAULT_CPU_LAST}"

MEM_TOTAL_KB="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
DEFAULT_MEMORY_MIB="$((MEM_TOTAL_KB / 2048))"
DEFAULT_MEMORY="${DEFAULT_MEMORY_MIB}m"

CPUSET_CPUS="${CPUSET_CPUS:-${DEFAULT_CPUSET_CPUS}}"
MEMORY="${MEMORY:-${DEFAULT_MEMORY}}"
MEMORY_SWAP="${MEMORY_SWAP:-${MEMORY}}"
CAPTURE_VIDEO="${CAPTURE_VIDEO:-false}"

if [[ "${GPU_OVERRIDE:-}" != "" && "${GPU_OVERRIDE}" != "0" ]]; then
  echo "Isaac Gym Docker is 3090-only on this machine. Use GPU_OVERRIDE=0 or unset it." >&2
  exit 1
fi

echo "Isaac Gym Docker target: physical GPU 0 / RTX 3090"
echo "CPU limit: ${CPUSET_CPUS} (override with CPUSET_CPUS=...)"
echo "Memory limit: ${MEMORY} (override with MEMORY=...)"
echo "Video capture: ${CAPTURE_VIDEO} (set CAPTURE_VIDEO=true to save Isaac Gym camera videos; leave false for max throughput)"

DOCKER_ARGS=(
  --rm
  --name "${NAME}"
  --gpus "device=${GPU}"
  --ipc=host
  --shm-size "${SHM_SIZE}"
  --ulimit memlock=-1
  --ulimit stack=67108864
  --network host
  --cpuset-cpus "${CPUSET_CPUS}"
  --memory "${MEMORY}"
  --memory-swap "${MEMORY_SWAP}"
  -e NVIDIA_VISIBLE_DEVICES="${GPU}"
  -e NVIDIA_DRIVER_CAPABILITIES=all
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID
  -e CUDA_VISIBLE_DEVICES=0
  -e PYTHON_BIN=python
  -e WANDB_DIR=/workspace/simtoolreal/outputs
  -v "${REPO_ROOT}:/workspace/simtoolreal"
  -w /workspace/simtoolreal
)

FORWARDED_ENV_VARS=(
  RUN_NAME
  LOG_PATH
  NUM_ENVS
  NUM_BLOCKS
  MAX_EPOCHS
  HORIZON_LENGTH
  MINIBATCH_SIZE
  MINI_EPOCHS
  SAVE_FREQUENCY
  SEQ_LENGTH
  CAPTURE_VIDEO
  CAPTURE_VIDEO_FREQ
  CAPTURE_VIDEO_LEN
  CAPTURE_VIDEO_RESOLUTION_REDUCTION
  WANDB_ACTIVATE
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
if [[ -t 0 && -t 1 ]]; then
  TTY_ARGS=(-it)
fi

if [[ -f "${HOME}/.netrc" ]]; then
  DOCKER_ARGS+=(-v "${HOME}/.netrc:/home/simuser/.netrc:ro")
fi

if [[ -d "${HOME}/.config/wandb" ]]; then
  DOCKER_ARGS+=(-v "${HOME}/.config/wandb:/home/simuser/.config/wandb:ro")
fi

if [[ -d "${HOME}/.local/share/wandb" ]]; then
  DOCKER_ARGS+=(-v "${HOME}/.local/share/wandb:/home/simuser/.local/share/wandb:ro")
fi

if [[ $# -eq 0 ]]; then
  docker run "${TTY_ARGS[@]}" "${DOCKER_ARGS[@]}" "${IMAGE_NAME}" /bin/bash
else
  docker run "${TTY_ARGS[@]}" "${DOCKER_ARGS[@]}" "${IMAGE_NAME}" "$@"
fi
