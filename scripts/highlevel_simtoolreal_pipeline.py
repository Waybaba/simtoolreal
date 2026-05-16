#!/usr/bin/env python3
"""Smoke a hierarchical SimToolReal pipeline with a frozen low-level checkpoint.

The high-level policy proposes the next low-level goal pose. The frozen
low-level CleanRL policy then runs for a short horizon to execute that goal.
This script is intentionally a pipeline scaffold first; the high-level policy
can later be replaced with PPO while keeping the same low-level interface.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ISAAC_SRC = REPO_ROOT / "src" / "isaaclab_env"
if str(LOCAL_ISAAC_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_ISAAC_SRC))

from isaaclab.app import AppLauncher


class PipelineAppLauncher(AppLauncher):
    def __init__(
        self,
        launcher_args: argparse.Namespace,
        *,
        active_gpu_override: Optional[int] = None,
        physics_gpu_override: Optional[int] = None,
    ):
        self._active_gpu_override = active_gpu_override
        self._physics_gpu_override = physics_gpu_override
        super().__init__(launcher_args)

    def _resolve_device_settings(self, launcher_args: dict) -> None:
        super()._resolve_device_settings(launcher_args)
        if self._active_gpu_override is not None:
            launcher_args["active_gpu"] = self._active_gpu_override
        if self._physics_gpu_override is not None:
            launcher_args["physics_gpu"] = self._physics_gpu_override


parser = argparse.ArgumentParser(description="Run a hierarchical SimToolReal low-level-controller pipeline.")
parser.add_argument("--task", type=str, default="SimToolReal-Direct-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=6)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--high_level_steps", type=int, default=8)
parser.add_argument("--low_level_horizon", type=int, default=60)
parser.add_argument("--high_level_policy", choices=("random", "greedy_place", "scripted_strike"), default="greedy_place")
parser.add_argument("--task_mode", choices=("place", "switch"), default="place")
parser.add_argument("--task_target_pos", type=float, nargs=3, default=(0.12, 0.0, 0.78))
parser.add_argument("--task_tolerance", type=float, default=0.10)
parser.add_argument("--goal_delta_scale", type=float, default=0.10)
parser.add_argument("--goal_pitch_scale_degrees", type=float, default=60.0)
parser.add_argument("--capture_video", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--render_env_id", type=int, default=-1)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs" / "isaaclab_highlevel")
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--disable_fabric", action="store_true")
parser.add_argument("--env_device", type=str, default=None)
parser.add_argument("--kit_active_gpu", type=int, default=None)
parser.add_argument("--kit_physics_gpu", type=int, default=None)
parser.add_argument("--simple_hammer_debug", action=argparse.BooleanOptionalAction, default=True)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, enable_cameras=True)
args_cli = parser.parse_args()
if not args_cli.capture_video:
    args_cli.enable_cameras = False

app_launcher = PipelineAppLauncher(
    args_cli,
    active_gpu_override=args_cli.kit_active_gpu,
    physics_gpu_override=args_cli.kit_physics_gpu,
)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_env.tasks  # noqa: F401, E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import quat_from_angle_axis  # noqa: E402

from simtoolreal_lowlevel_policy import load_low_level_policy  # noqa: E402


def _apply_simple_hammer(env_cfg) -> None:
    env_cfg.handle_head_types = ("hammer",)
    env_cfg.handle_head_distribution_index = 0
    env_cfg.procedural_objects_per_distribution = 1
    env_cfg.fixed_handle_head_object = True
    env_cfg.fixed_handle_scale = (0.225, 0.03, 0.0225)
    env_cfg.fixed_head_scale = (0.04, 0.085, 0.04)
    env_cfg.fixed_handle_density = 450.0
    env_cfg.fixed_head_density = 1400.0
    env_cfg.__post_init__()
    env_cfg.object_scale_noise_multiplier_range = (1.0, 1.0)
    env_cfg.use_action_delay = False
    env_cfg.use_object_state_delay_noise = False
    env_cfg.object_state_xyz_noise_std = 0.0
    env_cfg.object_state_rotation_noise_degrees = 0.0
    env_cfg.joint_velocity_obs_noise_std = 0.0


def _set_camera(env, env_id: int) -> None:
    unwrapped = env.unwrapped
    env_id = min(max(env_id, 0), unwrapped.num_envs - 1)
    origin = unwrapped.scene.env_origins[env_id].detach().cpu()
    eye = origin + torch.tensor([1.15, 0.15, 0.95])
    target = origin + torch.tensor([0.0, 0.28, 0.58])
    unwrapped.sim.set_camera_view(
        eye=eye.tolist(),
        target=target.tolist(),
        camera_prim_path=unwrapped.cfg.viewer.cam_prim_path,
    )


def _capture_frame(env) -> np.ndarray:
    for _ in range(2):
        env.unwrapped.sim.render()
    frame = np.asarray(env.render())
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _write_low_level_goal(env, local_pos: torch.Tensor, local_quat: torch.Tensor) -> None:
    unwrapped = env.unwrapped
    env_ids = torch.arange(unwrapped.num_envs, dtype=torch.long, device=unwrapped.device)
    goal_state = unwrapped.goal_object.data.root_state_w.clone()
    goal_state[:, :3] = local_pos + unwrapped.scene.env_origins
    goal_state[:, 3:7] = local_quat
    goal_state[:, 7:13] = 0.0
    unwrapped.goal_object.write_root_state_to_sim(goal_state, env_ids=env_ids)
    unwrapped.goal_states[:, :3] = local_pos
    unwrapped.goal_states[:, 3:7] = local_quat
    unwrapped.goal_states[:, 7:13] = 0.0
    unwrapped.reset_goal_buf[:] = False


def _current_task_distance(env, task_target_pos: torch.Tensor, task_mode: str) -> torch.Tensor:
    unwrapped = env.unwrapped
    unwrapped._compute_intermediate_values()
    if task_mode == "switch":
        return torch.linalg.norm(unwrapped.obj_keypoint_pos - task_target_pos.unsqueeze(1), dim=-1).min(dim=-1).values
    return torch.linalg.norm(unwrapped.object_pos - task_target_pos, dim=-1)


def _high_level_action(args: argparse.Namespace, env, high_step: int, task_target_pos: torch.Tensor) -> torch.Tensor:
    unwrapped = env.unwrapped
    unwrapped._compute_intermediate_values()
    device = unwrapped.device
    action = torch.zeros((unwrapped.num_envs, 4), dtype=torch.float32, device=device)
    if args.high_level_policy == "random":
        return torch.rand_like(action) * 2.0 - 1.0
    if args.high_level_policy == "scripted_strike":
        phase = (high_step % max(1, args.high_level_steps)) / max(1, args.high_level_steps - 1)
        action[:, 0] = -1.0 + 2.0 * phase
        action[:, 2] = 0.6 * math.sin(math.pi * phase)
        action[:, 3] = 1.0 - 2.0 * phase
        return torch.clamp(action, -1.0, 1.0)

    direction = task_target_pos - unwrapped.object_pos
    action[:, :3] = direction / max(args.goal_delta_scale, 1.0e-6)
    return torch.clamp(action, -1.0, 1.0)


def _action_to_goal(env, action: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    unwrapped = env.unwrapped
    unwrapped._compute_intermediate_values()
    goal_pos = unwrapped.object_pos + action[:, :3] * args.goal_delta_scale
    goal_pos = torch.maximum(torch.minimum(goal_pos, unwrapped.target_volume_max), unwrapped.target_volume_min)
    pitch = action[:, 3] * math.radians(args.goal_pitch_scale_degrees)
    axis = torch.tensor((0.0, 1.0, 0.0), dtype=torch.float32, device=unwrapped.device).repeat(unwrapped.num_envs, 1)
    goal_quat = quat_from_angle_axis(pitch, axis)
    return goal_pos, goal_quat


def main() -> None:
    env_device = args_cli.env_device or args_cli.device
    run_name = args_cli.run_name or f"{args_cli.high_level_policy}_{args_cli.task_mode}_seed{args_cli.seed}"
    output_dir = args_cli.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "highlevel_rollout.mp4"
    stats_path = output_dir / "highlevel_stats.json"
    trace_path = output_dir / "highlevel_trace.jsonl"

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=env_device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.viewer.resolution = (1280, 720)
    env_cfg.viewer.cam_prim_path = "/OmniverseKit_Persp"
    env_cfg.log_dir = str(output_dir)
    if args_cli.simple_hammer_debug:
        _apply_simple_hammer(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.capture_video else None)
    try:
        obs, _ = env.reset(seed=args_cli.seed)
        render_env_id = args_cli.render_env_id if args_cli.render_env_id >= 0 else args_cli.num_envs - 1
        if args_cli.capture_video:
            _set_camera(env, render_env_id)

        device = torch.device(env.unwrapped.device)
        low_level = load_low_level_policy(
            args_cli.checkpoint,
            policy_obs_dim=int(obs["policy"].shape[-1]),
            critic_obs_dim=int(obs["critic"].shape[-1]),
            action_dim=int(env.action_space.shape[-1]),
            num_envs=args_cli.num_envs,
            device=device,
        )
        actor_state = low_level.initial_state(args_cli.num_envs, device)
        next_done = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
        task_target_pos = torch.tensor(args_cli.task_target_pos, dtype=torch.float32, device=device).reshape(1, 3).repeat(
            args_cli.num_envs, 1
        )
        prev_task_dist = _current_task_distance(env, task_target_pos, args_cli.task_mode)
        best_task_dist = prev_task_dist.clone()
        high_level_rewards: list[float] = []
        trace_rows: list[dict] = []

        writer_context = imageio.get_writer(str(video_path), fps=args_cli.fps, macro_block_size=1) if args_cli.capture_video else None
        trace_file = trace_path.open("w", encoding="utf-8")
        try:
            writer = writer_context.__enter__() if writer_context is not None else None
            with torch.inference_mode():
                for high_step in range(args_cli.high_level_steps):
                    high_action = _high_level_action(args_cli, env, high_step, task_target_pos)
                    low_goal_pos, low_goal_quat = _action_to_goal(env, high_action, args_cli)
                    _write_low_level_goal(env, low_goal_pos, low_goal_quat)
                    low_success_before = env.unwrapped.successes.detach().clone()

                    for _ in range(args_cli.low_level_horizon):
                        policy_obs = env.unwrapped._get_observations()["policy"].to(device)
                        action, actor_state = low_level.act(policy_obs, actor_state, next_done)
                        _obs, _reward, terminated, truncated, _info = env.step(action)
                        next_done = torch.logical_or(terminated, truncated).to(device)
                        if writer is not None:
                            writer.append_data(_capture_frame(env))

                    task_dist = _current_task_distance(env, task_target_pos, args_cli.task_mode)
                    best_task_dist = torch.minimum(best_task_dist, task_dist)
                    task_success = task_dist <= args_cli.task_tolerance
                    high_reward = prev_task_dist - task_dist + task_success.float()
                    high_level_rewards.append(float(high_reward.mean().item()))
                    low_success_delta = env.unwrapped.successes.detach() - low_success_before
                    row = {
                        "high_step": high_step,
                        "high_policy": args_cli.high_level_policy,
                        "task_mode": args_cli.task_mode,
                        "reward_mean": float(high_reward.mean().item()),
                        "task_dist_mean": float(task_dist.mean().item()),
                        "task_success_rate": float(task_success.float().mean().item()),
                        "low_level_success_rate": float((low_success_delta > 0).float().mean().item()),
                        "render_env_goal_pos": low_goal_pos[render_env_id].detach().cpu().tolist(),
                        "render_env_task_dist": float(task_dist[render_env_id].item()),
                    }
                    trace_rows.append(row)
                    trace_file.write(json.dumps(row) + "\n")
                    trace_file.flush()
                    prev_task_dist = task_dist
        finally:
            trace_file.close()
            if writer_context is not None:
                writer_context.__exit__(None, None, None)

        final_task_dist = _current_task_distance(env, task_target_pos, args_cli.task_mode)
        stats = {
            "checkpoint": str(args_cli.checkpoint),
            "checkpoint_update": int(low_level.checkpoint.get("update", 0)),
            "checkpoint_global_step": int(low_level.checkpoint.get("global_step", 0)),
            "num_envs": args_cli.num_envs,
            "high_level_policy": args_cli.high_level_policy,
            "task_mode": args_cli.task_mode,
            "task_target_pos": list(args_cli.task_target_pos),
            "high_level_steps": args_cli.high_level_steps,
            "low_level_horizon": args_cli.low_level_horizon,
            "goal_delta_scale": args_cli.goal_delta_scale,
            "task_tolerance": args_cli.task_tolerance,
            "mean_high_level_reward": float(np.mean(high_level_rewards)) if high_level_rewards else 0.0,
            "final_task_success_rate": float((final_task_dist <= args_cli.task_tolerance).float().mean().item()),
            "best_task_success_rate": float((best_task_dist <= args_cli.task_tolerance).float().mean().item()),
            "final_task_dist_mean": float(final_task_dist.mean().item()),
            "best_task_dist_mean": float(best_task_dist.mean().item()),
            "video_path": str(video_path) if args_cli.capture_video else None,
            "trace_path": str(trace_path),
            "trace": trace_rows,
        }
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"[HILEVEL] stats={stats_path}", flush=True)
        print(f"[HILEVEL] trace={trace_path}", flush=True)
        if args_cli.capture_video:
            print(f"[HILEVEL] video={video_path}", flush=True)
        print(
            "[HILEVEL] "
            f"best_task_success_rate={stats['best_task_success_rate']:.4f} "
            f"final_task_success_rate={stats['final_task_success_rate']:.4f} "
            f"best_task_dist_mean={stats['best_task_dist_mean']:.4f}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"[HILEVEL][ERROR] {type(err).__name__}: {err}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
