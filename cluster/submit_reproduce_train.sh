#!/bin/bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SBATCH_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/reproduce_train.sbatch"

mkdir -p "${REPO_DIR}/outputs/slurm_logs/simtoolreal-train"
mkdir -p "${REPO_DIR}/outputs/train_dir"

CODE_COMMIT_HASH="${CODE_COMMIT_HASH:-$(git -C "${REPO_DIR}" rev-parse HEAD)}" \
sbatch "${SBATCH_SCRIPT}"
