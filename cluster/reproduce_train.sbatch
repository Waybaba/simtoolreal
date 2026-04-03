#!/bin/bash
# Formal training template for reproducing SimToolReal results on the cluster.

#SBATCH --job-name=simtoolreal-train
#SBATCH --partition=batch
#SBATCH --gres=gpu:l40s:1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/slurm_logs/simtoolreal-train/%j.out
#SBATCH --error=outputs/slurm_logs/simtoolreal-train/%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COMPILER_DIR="${REPO_DIR}/.compiler"

# Optional: pin the run to a clean worktree at a specific commit.
# Leave CODE_COMMIT_HASH empty to use the current checkout directly.
CODE_COMMIT_HASH="${CODE_COMMIT_HASH:-}"
WORKTREE_BASE="${REPO_DIR}/outputs/worktrees"

# Training knobs.
CUSTOM_EXPERIMENT_NAME="${CUSTOM_EXPERIMENT_NAME:-reproduce}"
SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-24576}"
NUM_BLOCKS="${NUM_BLOCKS:-6}"
FORCE_SCALE="${FORCE_SCALE:-20}"
TORQUE_SCALE="${TORQUE_SCALE:-2.0}"
OBJECT_ANG_VEL_PENALTY_SCALE="${OBJECT_ANG_VEL_PENALTY_SCALE:-0.0}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

# WandB knobs.
WANDB_ENTITY="${WANDB_ENTITY:-waybabag}"
WANDB_PROJECT="${WANDB_PROJECT:-simtoolreal}"
WANDB_GROUP="${WANDB_GROUP:-$(date +%Y-%m-%d)}"
WANDB_TAGS="${WANDB_TAGS:-cluster,reproduction}"
WANDB_NOTES="${WANDB_NOTES:-}"
WANDB_ACTIVATE="${WANDB_ACTIVATE:-True}"

# Video knobs. The environment already knows how to upload videos to WandB when
# wandb is active and imageio-ffmpeg is installed.
CAPTURE_VIDEO="${CAPTURE_VIDEO:-True}"
CAPTURE_VIDEO_FREQ="${CAPTURE_VIDEO_FREQ:-6000}"
CAPTURE_VIDEO_LEN="${CAPTURE_VIDEO_LEN:-600}"

mkdir -p "${REPO_DIR}/outputs/slurm_logs/simtoolreal-train"
mkdir -p "${REPO_DIR}/outputs/train_dir"

PRJ_DIR="${REPO_DIR}"
if [[ -n "${CODE_COMMIT_HASH}" ]]; then
  PROJECT_NAME="$(basename "${REPO_DIR}")"
  PRJ_DIR="${WORKTREE_BASE}/${PROJECT_NAME}/${CODE_COMMIT_HASH}"
  mkdir -p "$(dirname "${PRJ_DIR}")"
  if [[ ! -d "${PRJ_DIR}" ]]; then
    git -C "${REPO_DIR}" worktree add "${PRJ_DIR}" "${CODE_COMMIT_HASH}"
  fi
fi

cd "${PRJ_DIR}"
source .venv/bin/activate

export PYTHONPATH="${PRJ_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export DISPLAY=
export PATH="${COMPILER_DIR}/bin:${PATH}"
export CC="${COMPILER_DIR}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${COMPILER_DIR}/bin/x86_64-conda-linux-gnu-g++"
export CPATH="/usr/include:/usr/include/x86_64-linux-gnu:${CPATH:-}"
export LD_LIBRARY_PATH="${COMPILER_DIR}/lib64:${COMPILER_DIR}/lib:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR="${HOME}/.cache/torch_extensions"

echo "[job] hostname=$(hostname)"
echo "[job] repo_dir=${REPO_DIR}"
echo "[job] project_dir=${PRJ_DIR}"
echo "[job] python=$(command -v python)"
echo "[job] python_version=$(python --version)"
echo "[job] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[job] gcc=$(${CC} --version | head -n 1)"

mkdir -p "${TORCH_EXTENSIONS_DIR}"
nvidia-smi || true

cmd=(
  python isaacgymenvs/launch_training.py
  --custom-experiment-name "${CUSTOM_EXPERIMENT_NAME}"
  --seed "${SEED}"
  --num-envs "${NUM_ENVS}"
  --num-blocks "${NUM_BLOCKS}"
  --force-scale "${FORCE_SCALE}"
  --torque-scale "${TORQUE_SCALE}"
  --object-ang-vel-penalty-scale "${OBJECT_ANG_VEL_PENALTY_SCALE}"
  --wandb-entity "${WANDB_ENTITY}"
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
  --wandb-notes "${WANDB_NOTES}"
  --wandb-logcode-dir "${PRJ_DIR}"
  --output-root "./outputs/train_dir"
  --capture-video-freq "${CAPTURE_VIDEO_FREQ}"
  --capture-video-len "${CAPTURE_VIDEO_LEN}"
)

if [[ -n "${CHECKPOINT_PATH}" ]]; then
  cmd+=(--checkpoint "${CHECKPOINT_PATH}")
fi

shopt -s nocasematch
if [[ "${WANDB_ACTIVATE}" == "true" || "${WANDB_ACTIVATE}" == "1" || "${WANDB_ACTIVATE}" == "yes" ]]; then
  cmd+=(--wandb-activate)
else
  cmd+=(--no-wandb-activate)
fi

if [[ "${CAPTURE_VIDEO}" == "true" || "${CAPTURE_VIDEO}" == "1" || "${CAPTURE_VIDEO}" == "yes" ]]; then
  cmd+=(--capture-video)
else
  cmd+=(--no-capture-video)
fi
shopt -u nocasematch

if [[ -n "${WANDB_TAGS}" ]]; then
  IFS=',' read -r -a tags <<< "${WANDB_TAGS}"
  for tag in "${tags[@]}"; do
    if [[ -n "${tag}" ]]; then
      cmd+=(--wandb-tags "${tag}")
    fi
  done
fi

printf '[job] command='
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
