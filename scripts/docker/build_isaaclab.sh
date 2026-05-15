#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

IMAGE_NAME="${IMAGE_NAME:-simtoolreal-isaaclab:latest}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/waybaba/Documents/IsaacLab_install/IsaacLab510}"
ISAACSIM_VERSION="${ISAACSIM_VERSION:-5.1.0}"
USER_ID="${USER_ID:-$(id -u)}"
GROUP_ID="${GROUP_ID:-$(id -g)}"

if [[ ! -f "${ISAACLAB_ROOT}/isaaclab.sh" ]]; then
  echo "Isaac Lab not found at ${ISAACLAB_ROOT}" >&2
  echo "Set ISAACLAB_ROOT=/path/to/IsaacLab and rerun." >&2
  exit 1
fi

export DOCKER_BUILDKIT=1

docker build \
  --build-context "isaaclab=${ISAACLAB_ROOT}" \
  --build-arg "ISAACSIM_VERSION=${ISAACSIM_VERSION}" \
  --build-arg "USER_ID=${USER_ID}" \
  --build-arg "GROUP_ID=${GROUP_ID}" \
  -f docker/isaaclab/Dockerfile \
  -t "${IMAGE_NAME}" \
  .
