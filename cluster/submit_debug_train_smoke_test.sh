#!/bin/bash

set -euo pipefail

PARTITION="aloque-compute"
MIN_FREE_GPUS=2
TOTAL_GPUS=8
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SBATCH_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/debug_train_smoke_test.sbatch"

used_gpus=$(
  squeue -h -p "${PARTITION}" -t R -o "%b" \
    | python -c 'import re,sys; total=0
for line in sys.stdin:
    m=re.search(r"gres/gpu:l40s:(\d+)", line)
    if m:
        total += int(m.group(1))
print(total)'
)
free_gpus=$((TOTAL_GPUS - used_gpus))
echo "[submit] partition=${PARTITION} used_gpus=${used_gpus} free_gpus=${free_gpus}"
if (( free_gpus < MIN_FREE_GPUS )); then
  echo "[submit] Not submitting. Need at least ${MIN_FREE_GPUS} free GPUs." >&2
  exit 2
fi

mkdir -p "${REPO_DIR}/outputs/slurm_logs/simtoolreal-train-smoke"
mkdir -p "${REPO_DIR}/outputs/train_smoke"

sbatch "${SBATCH_SCRIPT}"
