#!/bin/bash
# Short end-to-end debug run for the formal training pipeline.

#SBATCH --job-name=simtoolreal-train-debug
#SBATCH --partition=aloque-compute
#SBATCH --gres=gpu:l40s:1
#SBATCH --qos=al-high-8gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/slurm_logs/simtoolreal-train-debug/%j.out
#SBATCH --error=outputs/slurm_logs/simtoolreal-train-debug/%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COMPILER_DIR="${REPO_DIR}/.compiler"

mkdir -p "${REPO_DIR}/outputs/slurm_logs/simtoolreal-train-debug"
mkdir -p "${REPO_DIR}/outputs/train_dir"
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

python isaacgymenvs/launch_training.py \
  --custom-experiment-name debug-repro \
  --seed 0 \
  --num-envs 1536 \
  --num-blocks 6 \
  --wandb-entity waybabag \
  --wandb-project simtoolreal \
  --wandb-group debug \
  --wandb-activate \
  --wandb-tags cluster debug short \
  --wandb-notes "Short cluster debug run before formal reproduction submission." \
  --wandb-logcode-dir . \
  --output-root ./outputs/train_dir \
  --no-capture-video \
  --max-epochs 1 \
  --horizon-length 4 \
  --minibatch-size 6144 \
  --mini-epochs 1 \
  --save-frequency 1 \
  --seq-length 4
