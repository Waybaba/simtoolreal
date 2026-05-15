#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

IMAGE_NAME="${IMAGE_NAME:-simtoolreal-isaacgym:latest}"
ISAACGYM_ROOT="${ISAACGYM_ROOT:-/home/waybaba/Downloads/isaacgym_install/isaacgym}"
USER_ID="${USER_ID:-$(id -u)}"
GROUP_ID="${GROUP_ID:-$(id -g)}"

if [[ ! -f "${ISAACGYM_ROOT}/python/setup.py" ]]; then
  echo "Isaac Gym package not found at ${ISAACGYM_ROOT}" >&2
  echo "Set ISAACGYM_ROOT=/path/to/isaacgym and rerun." >&2
  exit 1
fi

export DOCKER_BUILDKIT=1

docker build \
  --build-context "isaacgym=${ISAACGYM_ROOT}" \
  --build-arg "USER_ID=${USER_ID}" \
  --build-arg "GROUP_ID=${GROUP_ID}" \
  -f docker/isaacgym/Dockerfile \
  -t "${IMAGE_NAME}" \
  .
