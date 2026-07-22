#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 GPU_ID CHECKPOINT SEED TAG" >&2
  exit 2
fi

GPU_ID="$1"
CHECKPOINT="$2"
SEED="$3"
TAG="$4"

REPO_ROOT="${REPO_ROOT:-/home/wang100/code/simtoolreal}"
ISAACLAB_PYTHON="${ISAACLAB_PYTHON:-/home/wang100/data/conda/envs/isaaclab510/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/wang100/data}"
RUN_GROUP="${RUN_GROUP:-skill_discovery_hammer_sync_20260722}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

if [[ ! -x "${ISAACLAB_PYTHON}" ]]; then
  echo "Missing isaaclab510 Python: ${ISAACLAB_PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${REPO_ROOT}/${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${REPO_ROOT}/${CHECKPOINT}" >&2
  exit 2
fi

checkpoint_name="$(basename "${CHECKPOINT}")"
checkpoint_update="${checkpoint_name#checkpoint_}"
checkpoint_update="${checkpoint_update%.pt}"
checkpoint_update="$((10#${checkpoint_update}))"
total_updates="$((checkpoint_update + 1))"
run_name="hammer_sync_${TAG}_u${checkpoint_update}_seed${SEED}_gpu${GPU_ID}_${TIMESTAMP}"
log_dir="${REPO_ROOT}/outputs/skill_discovery/hammer_sync/logs"
log_path="${log_dir}/${run_name}.log"

mkdir -p \
  "${log_dir}" \
  "${DATA_ROOT}/cache" \
  "${DATA_ROOT}/cache/pip" \
  "${DATA_ROOT}/cache/huggingface" \
  "${DATA_ROOT}/xdg/gpu${GPU_ID}/cache" \
  "${DATA_ROOT}/xdg/gpu${GPU_ID}/config" \
  "${DATA_ROOT}/xdg/gpu${GPU_ID}/data" \
  "${DATA_ROOT}/isaaclab510_portable/gpu${GPU_ID}"

echo "run_name=${run_name}"
echo "gpu=${GPU_ID} checkpoint=${CHECKPOINT} seed=${SEED} log=${log_path}"

cd "${REPO_ROOT}"
unset CUDA_VISIBLE_DEVICES
export XDG_CACHE_HOME="${DATA_ROOT}/xdg/gpu${GPU_ID}/cache"
export XDG_CONFIG_HOME="${DATA_ROOT}/xdg/gpu${GPU_ID}/config"
export XDG_DATA_HOME="${DATA_ROOT}/xdg/gpu${GPU_ID}/data"
export PIP_CACHE_DIR="${DATA_ROOT}/cache/pip"
export HF_HOME="${DATA_ROOT}/cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES

exec "${ISAACLAB_PYTHON}" -u scripts/train_simtoolreal_cleanrl.py \
  --task SimToolReal-Direct-v0 \
  --checkpoint "${CHECKPOINT}" \
  --run_name "${run_name}" \
  --wandb_group "${RUN_GROUP}" \
  --total_updates "${total_updates}" \
  --seed "${SEED}" \
  --num_envs 30 \
  --num_steps 256 \
  --seq_length 16 \
  --minibatch_size 480 \
  --update_epochs 1 \
  --learning_rate 0.0 \
  --mixed_precision \
  --numeric_guard \
  --no-adaptive_lr \
  --kl_threshold 0.010 \
  --sapg_num_blocks 1 \
  --use_others_experience none \
  --fixed_sigma coef_cond \
  --min_log_sigma -3.0 \
  --max_log_sigma -0.9 \
  --expl_reward_type none \
  --expl_reward_coef_scale 0.0 \
  --save_frequency 1000000 \
  --save_best_after 1000000 \
  --no-wandb \
  --capture_video \
  --capture_video_freq 1000000 \
  --capture_video_len 240 \
  --capture_video_env_id 0 \
  --no-capture_video_start_on_reset \
  --video_camera_mode env_subject \
  --video_camera_eye_offset 0.32 -0.72 1.04 \
  --video_camera_target_offset 0.04 -0.02 0.55 \
  --trajectory_log \
  --trajectory_log_selection first \
  --trajectory_log_max_envs 1 \
  --trajectory_log_every 1 \
  --trajectory_log_max_frames 240 \
  --no-trajectory_log_include_obs \
  --handle_head_types hammer \
  --handle_head_distribution_index 1 \
  --procedural_objects_per_distribution 1 \
  --fixed_handle_head_object \
  --fixed_handle_scale 0.225 0.0225 \
  --fixed_head_scale 0.04 0.085 0.04 \
  --fixed_handle_density 450.0 \
  --fixed_head_density 1400.0 \
  --object_scale_noise_min 1.0 \
  --object_scale_noise_max 1.0 \
  --force_scale 0.0 \
  --torque_scale 0.0 \
  --reset_position_noise_z 0.0 \
  --reset_position_noise_xy_schedule_start 0.0521 \
  --reset_position_noise_xy_schedule_end 0.10 \
  --reset_position_noise_xy_schedule_updates 1300 \
  --table_object_z_offset 0.166 \
  --randomize_object_rotation \
  --fixed_goal_pos 0.0 0.0 0.62 \
  --fixed_goal_quat_from_object_init \
  --fixed_goal_xy_from_object_init_blend 0.0 \
  --lifting_rew_scale 5.0 \
  --lifting_bonus 20.0 \
  --lifting_bonus_threshold 0.08 \
  --keypoint_rew_scale 2600.0 \
  --goal_distance_rew_scale 3600.0 \
  --goal_distance_rew_sigma 0.10 \
  --visual_distance_rew_scale 42000.0 \
  --visual_distance_rew_sigma 0.043 \
  --visual_success_bonus 170000.0 \
  --object_goal_pos_rew_scale 30000.0 \
  --object_goal_pos_rew_sigma 0.045 \
  --object_goal_xy_rew_scale 70000.0 \
  --object_goal_xy_rew_sigma 0.038 \
  --object_goal_z_rew_scale 32000.0 \
  --object_goal_z_rew_sigma 0.025 \
  --object_goal_pos_delta_rew_scale 2800.0 \
  --object_goal_xy_delta_rew_scale 5600.0 \
  --object_goal_pos_away_penalty_scale 2200.0 \
  --object_goal_xy_away_penalty_scale 4400.0 \
  --object_goal_pos_dist_penalty_scale 3300.0 \
  --object_goal_xy_dist_penalty_scale 6600.0 \
  --object_goal_pos_success_bonus 60000.0 \
  --object_goal_pos_success_tolerance 0.04 \
  --no-object_goal_rewards_require_lift \
  --distance_delta_rew_scale 180.0 \
  --reach_goal_bonus 5500.0 \
  --success_tolerance 0.04 \
  --success_steps 4 \
  --object_ang_vel_penalty_scale 0.02 \
  --force_consecutive_near_goal_steps \
  --device "cuda:${GPU_ID}" \
  --env_device "cuda:${GPU_ID}" \
  --kit_active_gpu "${GPU_ID}" \
  --kit_physics_gpu "${GPU_ID}" \
  --kit_args "--portable-root ${DATA_ROOT}/isaaclab510_portable/gpu${GPU_ID} --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/activeGpu=${GPU_ID}" \
  >"${log_path}" 2>&1
