#!/bin/bash
#SBATCH --job-name=simtoolreal-video
#SBATCH --partition=aloque-compute
#SBATCH --gres=gpu:l40s:1
#SBATCH --qos=al-high-8gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/slurm_logs/simtoolreal-video/%j.out
#SBATCH --error=outputs/slurm_logs/simtoolreal-video/%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ISAACGYM_PYTHON_DIR="/mnt/aloque_scratch/wang100/data/isaacgym/isaacgym/python"
COMPILER_DIR="${REPO_DIR}/.compiler"

mkdir -p "${REPO_DIR}/outputs/slurm_logs/simtoolreal-video"
mkdir -p "${REPO_DIR}/outputs/videos"
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

if [[ ! -d "${ISAACGYM_PYTHON_DIR}" ]]; then
  echo "[job] missing Isaac Gym path: ${ISAACGYM_PYTHON_DIR}" >&2
  exit 1
fi

python cluster/export_headless_video.py \
  --config-path pretrained_policy/config.yaml \
  --checkpoint-path pretrained_policy/model.pth \
  --object-category hammer \
  --object-name claw_hammer \
  --task-name swing_down \
  --video-length 60 \
  --max-steps 240 \
  --headless \
  --output-dir "${REPO_DIR}/outputs/videos"
