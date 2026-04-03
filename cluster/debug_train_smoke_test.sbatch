#!/bin/bash
#SBATCH --job-name=simtoolreal-train-smoke
#SBATCH --partition=aloque-compute
#SBATCH --gres=gpu:l40s:1
#SBATCH --qos=al-high-8gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/slurm_logs/simtoolreal-train-smoke/%j.out
#SBATCH --error=outputs/slurm_logs/simtoolreal-train-smoke/%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COMPILER_DIR="${REPO_DIR}/.compiler"

mkdir -p "${REPO_DIR}/outputs/slurm_logs/simtoolreal-train-smoke"
cd "${REPO_DIR}"

source .venv/bin/activate

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export DISPLAY=
export PATH="${COMPILER_DIR}/bin:${PATH}"
export CC="${COMPILER_DIR}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${COMPILER_DIR}/bin/x86_64-conda-linux-gnu-g++"
export CPATH="/usr/include:/usr/include/x86_64-linux-gnu:${CPATH:-}"
export LD_LIBRARY_PATH="${COMPILER_DIR}/lib64:${COMPILER_DIR}/lib:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR="${HOME}/.cache/torch_extensions"

echo "[job] hostname=$(hostname)"
echo "[job] cwd=$(pwd)"
echo "[job] python=$(command -v python)"
echo "[job] python_version=$(python --version)"
echo "[job] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[job] gcc=$(${CC} --version | head -n 1)"

mkdir -p "${TORCH_EXTENSIONS_DIR}"
nvidia-smi || true

python cluster/train_smoke_test.py \
  headless=True \
  task=SimToolReal \
  num_envs=64 \
  wandb_activate=False \
  force_render=False \
  capture_video=False \
  hydra.run.dir=./outputs/train_smoke/$(date +%Y%m%d_%H%M%S) \
  experiment=00_smoke_train \
  train.params.config.max_epochs=1 \
  train.params.config.horizon_length=4 \
  train.params.config.minibatch_size=256 \
  train.params.config.mini_epochs=1 \
  train.params.config.save_frequency=1 \
  train.params.config.save_best_after=0
