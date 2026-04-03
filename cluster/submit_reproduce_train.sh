#!/bin/bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SBATCH_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/reproduce_train.sbatch"

mkdir -p "${REPO_DIR}/outputs/slurm_logs/simtoolreal-train"
mkdir -p "${REPO_DIR}/outputs/train_dir"

sbatch "${SBATCH_SCRIPT}"
