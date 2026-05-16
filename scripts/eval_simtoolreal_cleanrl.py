#!/usr/bin/env python3
"""Evaluate a CleanRL SimToolReal checkpoint and save render evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ISAAC_SRC = REPO_ROOT / "src" / "isaaclab_env"
if str(LOCAL_ISAAC_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_ISAAC_SRC))

from isaaclab.app import AppLauncher


class EvalAppLauncher(AppLauncher):
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


parser = argparse.ArgumentParser(description="Evaluate/render a SimToolReal CleanRL checkpoint.")
parser.add_argument("--task", type=str, default="SimToolReal-Direct-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=6)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--render_env_id", type=int, default=-1)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs" / "isaaclab_eval")
parser.add_argument("--output_name", type=str, default=None)
parser.add_argument("--disable_fabric", action="store_true")
parser.add_argument("--env_device", type=str, default=None)
parser.add_argument("--kit_active_gpu", type=int, default=None)
parser.add_argument("--kit_physics_gpu", type=int, default=None)
parser.add_argument("--simple_hammer_debug", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--easy_goal_debug", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--goal_trajectory", choices=("native", "fixed", "line", "circle", "arc", "strike"), default="fixed")
parser.add_argument("--goal_center", type=float, nargs=3, default=(0.0, 0.0, 0.78))
parser.add_argument("--goal_amplitude", type=float, default=0.10)
parser.add_argument("--goal_height_amplitude", type=float, default=0.06)
parser.add_argument("--goal_period", type=int, default=240)
parser.add_argument("--goal_pitch_amplitude_degrees", type=float, default=55.0)
parser.add_argument("--gated_goal_sequence", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--sequence_goal_hold_steps", type=int, default=None)
parser.add_argument("--strict_success_requires_lift", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--eval_success_tolerance", type=float, default=None)
parser.add_argument("--visual_success_tolerance", type=float, default=0.05)
parser.add_argument("--diagnostic_tolerances", type=float, nargs="*", default=(0.03, 0.05, 0.07))
parser.add_argument("--force_consecutive_near_goal_steps", action=argparse.BooleanOptionalAction, default=None)
EvalAppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, enable_cameras=True)
args_cli = parser.parse_args()

app_launcher = EvalAppLauncher(
    args_cli,
    active_gpu_override=args_cli.kit_active_gpu,
    physics_gpu_override=args_cli.kit_physics_gpu,
)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.distributions.normal import Normal  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import quat_from_angle_axis  # noqa: E402

import isaaclab_env.tasks  # noqa: F401, E402


class RunningMeanStd(nn.Module):
    def __init__(self, shape: tuple[int, ...], epsilon: float = 1.0e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float32))

    def normalize(self, x: torch.Tensor, clip: float = 5.0) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1.0e-8), -clip, clip)


class ValueNormalizer(nn.Module):
    def __init__(self, epsilon: float = 1.0e-4):
        super().__init__()
        self.rms = RunningMeanStd((1,), epsilon=epsilon)

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.sqrt(self.rms.var + 1.0e-8) + self.rms.mean


def _init_linear(layer: nn.Linear, std: float = math.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _make_mlp(input_dim: int, hidden_sizes: list[int], output_dim: int | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_size in hidden_sizes:
        layers.append(_init_linear(nn.Linear(last_dim, hidden_size)))
        layers.append(nn.ELU())
        last_dim = hidden_size
    if output_dim is not None:
        layers.append(_init_linear(nn.Linear(last_dim, output_dim), std=1.0))
    return nn.Sequential(*layers)


class ExtraParamInput(nn.Module):
    def __init__(self, base_dim: int, coef_ids: torch.Tensor | None, extra_param_size: int):
        super().__init__()
        self.base_dim = base_dim
        if coef_ids is None:
            self.register_buffer("coef_ids", torch.empty(0))
            self.extra_params = None
            self.output_dim = base_dim
        else:
            self.register_buffer("coef_ids", coef_ids.float().clone())
            self.extra_params = nn.Parameter(torch.randn((len(coef_ids), extra_param_size), dtype=torch.float32))
            self.output_dim = base_dim + extra_param_size

    def forward(self, obs_with_optional_coef: torch.Tensor) -> torch.Tensor:
        if self.extra_params is None:
            return obs_with_optional_coef
        coef = obs_with_optional_coef[:, self.base_dim].reshape(-1, 1)
        block_ids = (coef == self.coef_ids.reshape(1, -1)).float().argmax(dim=1)
        return torch.cat([obs_with_optional_coef[:, : self.base_dim], self.extra_params[block_ids]], dim=-1)


class LSTMActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int],
        lstm_hidden_size: int,
        lstm_layers: int,
        coef_ids: torch.Tensor | None,
        extra_param_size: int,
        fixed_sigma: str,
    ):
        super().__init__()
        self.input_adapter = ExtraParamInput(obs_dim, coef_ids, extra_param_size)
        self.lstm = nn.LSTM(self.input_adapter.output_dim, lstm_hidden_size, num_layers=lstm_layers)
        self.layer_norm = nn.LayerNorm(lstm_hidden_size)
        self.mlp = _make_mlp(lstm_hidden_size, hidden_sizes)
        self.mu = _init_linear(nn.Linear(hidden_sizes[-1], action_dim), std=0.01)
        self.fixed_sigma = fixed_sigma
        if fixed_sigma == "coef_cond":
            assert coef_ids is not None
            self.register_buffer("sigma_ids", coef_ids.float().clone())
            self.log_sigma = nn.Parameter(torch.zeros((len(coef_ids), action_dim), dtype=torch.float32))
        else:
            self.register_buffer("sigma_ids", torch.empty(0))
            self.log_sigma = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

    def initial_state(self, num_envs: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros((self.lstm.num_layers, num_envs, self.lstm.hidden_size), dtype=torch.float32, device=device)
        return h, torch.zeros_like(h)

    def _sigma(self, obs_with_optional_coef: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        if self.fixed_sigma == "coef_cond":
            coef = obs_with_optional_coef[:, -1].reshape(-1, 1)
            block_ids = (coef == self.sigma_ids.reshape(1, -1)).float().argmax(dim=1)
            return torch.exp(self.log_sigma[block_ids])
        return torch.exp(self.log_sigma).expand_as(mu)

    def forward_step(
        self,
        obs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        done: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size = obs.shape[0]
        h, c = state
        not_done = (1.0 - done.float()).reshape(1, batch_size, 1)
        adapted = self.input_adapter(obs).unsqueeze(0)
        out, next_state = self.lstm(adapted, (h * not_done, c * not_done))
        features = self.mlp(self.layer_norm(out.squeeze(0)))
        mu = self.mu(features)
        return mu, self._sigma(obs, mu), next_state


class AsymmetricCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_sizes: list[int], coef_ids: torch.Tensor | None, extra_param_size: int):
        super().__init__()
        self.input_adapter = ExtraParamInput(state_dim, coef_ids, extra_param_size)
        self.net = _make_mlp(self.input_adapter.output_dim, hidden_sizes, output_dim=1)

    def forward(self, state_obs: torch.Tensor) -> torch.Tensor:
        return self.net(self.input_adapter(state_obs)).squeeze(-1)


class SimToolRealAgent(nn.Module):
    def __init__(
        self,
        policy_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int],
        lstm_hidden_size: int,
        lstm_layers: int,
        coef_ids: torch.Tensor | None,
        extra_param_size: int,
        fixed_sigma: str,
        value_normalizer: ValueNormalizer | None,
    ):
        super().__init__()
        self.actor = LSTMActor(
            policy_obs_dim,
            action_dim,
            hidden_sizes,
            lstm_hidden_size,
            lstm_layers,
            coef_ids,
            extra_param_size,
            fixed_sigma,
        )
        self.critic = AsymmetricCritic(critic_obs_dim, hidden_sizes, coef_ids, extra_param_size)
        self.value_normalizer = value_normalizer


def _append_coef(obs: torch.Tensor, env_coef: torch.Tensor | None) -> torch.Tensor:
    if env_coef is None:
        return obs
    return torch.cat([obs, env_coef], dim=-1)


def _make_sapg(ckpt_args: dict[str, Any], num_envs: int, device: torch.device) -> dict[str, Any]:
    if ckpt_args.get("expl_type", "mixed_expl_learn_param") == "none":
        return {"enabled": False, "coef_ids": None, "env_coef": None}
    num_blocks = int(ckpt_args.get("sapg_num_blocks", 6))
    if num_envs % num_blocks != 0:
        raise ValueError(f"--num_envs={num_envs} must be divisible by SAPG block count {num_blocks}")
    block_size = num_envs // num_blocks
    coef_ids = torch.linspace(50.0, 0.0, num_blocks, device=device)
    block_ids = torch.arange(num_blocks, device=device).repeat_interleave(block_size)
    return {"enabled": True, "coef_ids": coef_ids, "env_coef": coef_ids[block_ids].unsqueeze(-1)}


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


def _apply_easy_goal(env_cfg) -> None:
    _apply_simple_hammer(env_cfg)
    env_cfg.reset_position_noise_x = 0.0
    env_cfg.reset_position_noise_y = 0.0
    env_cfg.reset_position_noise_z = 0.0
    env_cfg.reset_dof_pos_noise_fingers = 0.0
    env_cfg.reset_dof_pos_noise_arm = 0.0
    env_cfg.reset_dof_vel_noise = 0.0
    env_cfg.randomize_object_rotation = False
    env_cfg.fixed_goal_pos = (0.0, 0.0, 0.78)
    env_cfg.fixed_goal_quat = (1.0, 0.0, 0.0, 0.0)
    env_cfg.force_scale = 0.0
    env_cfg.torque_scale = 0.0
    env_cfg.force_consecutive_near_goal_steps = False


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


def _goal_pose_for_step(args: argparse.Namespace, step: int, num_envs: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if args.goal_trajectory == "native":
        raise ValueError("native goal trajectory uses the environment's own goal reset logic")
    center = torch.tensor(args.goal_center, dtype=torch.float32, device=device).reshape(1, 3).repeat(num_envs, 1)
    period = max(1, int(args.goal_period))
    phase = 2.0 * math.pi * ((step % period) / period)
    pos = center.clone()
    pitch = 0.0

    if args.goal_trajectory == "line":
        pos[:, 0] += args.goal_amplitude * math.sin(phase)
    elif args.goal_trajectory == "circle":
        pos[:, 0] += args.goal_amplitude * math.cos(phase)
        pos[:, 1] += args.goal_amplitude * math.sin(phase)
    elif args.goal_trajectory == "arc":
        pos[:, 0] += args.goal_amplitude * math.cos(phase)
        pos[:, 2] += args.goal_height_amplitude * max(0.0, math.sin(phase))
        pitch = math.radians(args.goal_pitch_amplitude_degrees) * math.sin(phase)
    elif args.goal_trajectory == "strike":
        stroke = (step % period) / period
        pos[:, 0] += -args.goal_amplitude + 2.0 * args.goal_amplitude * stroke
        pos[:, 2] += args.goal_height_amplitude * math.sin(math.pi * stroke)
        pitch = math.radians(args.goal_pitch_amplitude_degrees) * (1.0 - 2.0 * stroke)

    angle = torch.full((num_envs,), float(pitch), dtype=torch.float32, device=device)
    axis = torch.tensor((0.0, 1.0, 0.0), dtype=torch.float32, device=device).repeat(num_envs, 1)
    quat = quat_from_angle_axis(angle, axis)
    return pos, quat


def _goal_sequence_for_args(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if args.goal_trajectory == "native":
        raise ValueError("native goal trajectory cannot be converted to a fixed gated sequence")
    center = torch.tensor(args.goal_center, dtype=torch.float32, device=device)
    amplitude = float(args.goal_amplitude)
    height = float(args.goal_height_amplitude)
    pitch_scale = math.radians(float(args.goal_pitch_amplitude_degrees))
    points: list[tuple[float, float, float]] = []
    pitches: list[float] = []

    if args.goal_trajectory == "line":
        for alpha in (-1.0, -0.5, 0.0, 0.5, 1.0):
            points.append((alpha * amplitude, 0.0, 0.0))
            pitches.append(0.0)
    elif args.goal_trajectory == "circle":
        for angle in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
            points.append((amplitude * math.cos(angle), amplitude * math.sin(angle), 0.0))
            pitches.append(0.0)
    elif args.goal_trajectory == "arc":
        for alpha in (-1.0, -0.33, 0.33, 1.0):
            z = height * max(0.0, 1.0 - abs(alpha))
            points.append((alpha * amplitude, 0.0, z))
            pitches.append(-alpha * pitch_scale)
    elif args.goal_trajectory == "strike":
        for alpha, z, pitch in (
            (-1.0, 0.0, 1.0),
            (-0.35, height, 0.45),
            (0.35, 0.55 * height, -0.35),
            (1.0, 0.0, -1.0),
        ):
            points.append((alpha * amplitude, 0.0, z))
            pitches.append(pitch * pitch_scale)
    else:
        points.append((0.0, 0.0, 0.0))
        pitches.append(0.0)

    offsets = torch.tensor(points, dtype=torch.float32, device=device)
    positions = center.reshape(1, 3) + offsets
    angles = torch.tensor(pitches, dtype=torch.float32, device=device)
    axes = torch.tensor((0.0, 1.0, 0.0), dtype=torch.float32, device=device).repeat(len(pitches), 1)
    quats = quat_from_angle_axis(angles, axes)
    metadata = {
        "mode": args.goal_trajectory,
        "gated": True,
        "center": list(args.goal_center),
        "amplitude": amplitude,
        "height_amplitude": height,
        "pitch_amplitude_degrees": float(args.goal_pitch_amplitude_degrees),
        "target_count": int(positions.shape[0]),
        "targets": [
            {
                "pos": positions[index].detach().cpu().tolist(),
                "quat": quats[index].detach().cpu().tolist(),
            }
            for index in range(positions.shape[0])
        ],
    }
    return positions, quats, metadata


def _write_goal_pose(env, local_pos: torch.Tensor, local_quat: torch.Tensor) -> None:
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


def _apply_goal_trajectory(env, args: argparse.Namespace, step: int, device: torch.device) -> dict[str, Any]:
    pos, quat = _goal_pose_for_step(args, step, env.unwrapped.num_envs, device)
    _write_goal_pose(env, pos, quat)
    return {
        "mode": args.goal_trajectory,
        "center": list(args.goal_center),
        "amplitude": float(args.goal_amplitude),
        "height_amplitude": float(args.goal_height_amplitude),
        "period": int(args.goal_period),
        "pitch_amplitude_degrees": float(args.goal_pitch_amplitude_degrees),
    }


def _apply_gated_goal_sequence(
    env,
    sequence_positions: torch.Tensor,
    sequence_quats: torch.Tensor,
    target_indices: torch.Tensor,
) -> None:
    clamped_indices = torch.clamp(target_indices, max=sequence_positions.shape[0] - 1)
    _write_goal_pose(env, sequence_positions[clamped_indices], sequence_quats[clamped_indices])


def _strict_current_success(env, tolerance: float, *, require_lifted: bool) -> torch.Tensor:
    unwrapped = env.unwrapped
    unwrapped._compute_intermediate_values()
    reached = unwrapped.keypoints_max_dist_for_reward <= tolerance
    if require_lifted:
        reached &= unwrapped.lifted_object
    return reached.detach()


def _quat_angle_error_deg(quat_a: torch.Tensor, quat_b: torch.Tensor) -> torch.Tensor:
    quat_a = quat_a / torch.clamp(torch.linalg.norm(quat_a, dim=-1, keepdim=True), min=1.0e-8)
    quat_b = quat_b / torch.clamp(torch.linalg.norm(quat_b, dim=-1, keepdim=True), min=1.0e-8)
    dot = torch.abs(torch.sum(quat_a * quat_b, dim=-1))
    angle = 2.0 * torch.acos(torch.clamp(dot, max=1.0))
    return torch.rad2deg(angle)


def _advance_gated_goal_sequence(
    env,
    *,
    step: int,
    target_indices: torch.Tensor,
    completed_counts: torch.Tensor,
    reach_steps: torch.Tensor,
    hit_streak: torch.Tensor,
    reached_current_goal: torch.Tensor,
    hold_steps: int,
    sequence_len: int,
) -> list[dict[str, Any]]:
    unwrapped = env.unwrapped
    active = target_indices < sequence_len
    hit_streak[:] = torch.where(active & reached_current_goal, hit_streak + 1, torch.zeros_like(hit_streak))
    reached = active & (hit_streak >= hold_steps)
    env_ids = reached.nonzero(as_tuple=False).flatten()
    events: list[dict[str, Any]] = []
    if env_ids.numel() == 0:
        return events

    reached_target_indices = target_indices[env_ids].clone()
    completed_counts[env_ids] += 1
    reach_steps[env_ids, reached_target_indices] = torch.where(
        reach_steps[env_ids, reached_target_indices] < 0,
        torch.full_like(reach_steps[env_ids, reached_target_indices], step),
        reach_steps[env_ids, reached_target_indices],
    )
    target_indices[env_ids] = torch.clamp(target_indices[env_ids] + 1, max=sequence_len)
    hit_streak[env_ids] = 0
    unwrapped.near_goal_steps[env_ids] = 0
    unwrapped.reset_goal_buf[env_ids] = False

    for env_id, target_id in zip(env_ids.detach().cpu().tolist(), reached_target_indices.detach().cpu().tolist()):
        events.append({"step": step, "env_id": int(env_id), "target_id": int(target_id)})
    return events


def _capture_frame(env) -> np.ndarray:
    for _ in range(2):
        env.unwrapped.sim.render()
    frame = np.asarray(env.render())
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def main() -> None:
    if args_cli.gated_goal_sequence and args_cli.goal_trajectory == "native":
        raise ValueError("--gated_goal_sequence requires a scripted goal_trajectory, not native")
    env_device = args_cli.env_device or args_cli.device
    checkpoint = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    output_name = args_cli.output_name or (
        f"{args_cli.checkpoint.parent.parent.name}_{args_cli.checkpoint.stem}_{args_cli.goal_trajectory}_seed{args_cli.seed}"
    )
    output_dir = args_cli.output_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "eval_rollout.mp4"
    final_image_path = output_dir / "final_frame.png"
    best_image_path = output_dir / "best_frame.png"
    stats_path = output_dir / "eval_stats.json"
    render_trace_csv_path = output_dir / "render_env_trace.csv"
    env_summary_csv_path = output_dir / "env_summary.csv"

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
    if args_cli.easy_goal_debug:
        _apply_easy_goal(env_cfg)
    elif args_cli.simple_hammer_debug:
        _apply_simple_hammer(env_cfg)
    if args_cli.force_consecutive_near_goal_steps is not None:
        env_cfg.force_consecutive_near_goal_steps = bool(args_cli.force_consecutive_near_goal_steps)
    elif "force_consecutive_near_goal_steps" in ckpt_args:
        env_cfg.force_consecutive_near_goal_steps = bool(ckpt_args["force_consecutive_near_goal_steps"])

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    try:
        obs, _ = env.reset(seed=args_cli.seed)
        render_env_id = args_cli.render_env_id if args_cli.render_env_id >= 0 else args_cli.num_envs - 1
        _set_camera(env, render_env_id)

        device = torch.device(env.unwrapped.device)
        policy_obs_dim = int(obs["policy"].shape[-1])
        critic_obs_dim = int(obs["critic"].shape[-1])
        action_dim = int(env.action_space.shape[-1])
        sapg = _make_sapg(ckpt_args, args_cli.num_envs, device)
        coef_ids = sapg["coef_ids"] if sapg["enabled"] else None

        obs_rms = RunningMeanStd((policy_obs_dim,), epsilon=1.0e-4).to(device) if ckpt_args.get("norm_obs", True) else None
        state_rms = RunningMeanStd((critic_obs_dim,), epsilon=1.0e-4).to(device) if ckpt_args.get("norm_obs", True) else None
        value_normalizer = ValueNormalizer(epsilon=1.0e-4).to(device) if ckpt_args.get("norm_value", True) else None
        agent = SimToolRealAgent(
            policy_obs_dim=policy_obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=action_dim,
            hidden_sizes=list(ckpt_args.get("hidden_sizes", [1024, 1024, 512, 512])),
            lstm_hidden_size=int(ckpt_args.get("lstm_hidden_size", 1024)),
            lstm_layers=int(ckpt_args.get("lstm_layers", 1)),
            coef_ids=coef_ids,
            extra_param_size=int(ckpt_args.get("extra_param_size", 32)) if sapg["enabled"] else 0,
            fixed_sigma=ckpt_args.get("fixed_sigma", "coef_cond") if sapg["enabled"] else "fixed",
            value_normalizer=value_normalizer,
        ).to(device)
        agent.load_state_dict(checkpoint["agent"])
        if obs_rms is not None and checkpoint.get("obs_rms") is not None:
            obs_rms.load_state_dict(checkpoint["obs_rms"])
        if state_rms is not None and checkpoint.get("state_rms") is not None:
            state_rms.load_state_dict(checkpoint["state_rms"])
        if value_normalizer is not None and checkpoint.get("value_normalizer") is not None:
            value_normalizer.load_state_dict(checkpoint["value_normalizer"])
        agent.eval()

        policy_obs = obs["policy"].to(device)
        next_done = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
        actor_state = agent.actor.initial_state(args_cli.num_envs, device)
        env_coef = sapg["env_coef"]

        best_dist = float("inf")
        best_all_env_dist = float("inf")
        best_all_env_id = 0
        best_frame = None
        final_frame = None
        max_successes = torch.zeros(args_cli.num_envs, device=device)
        ever_lifted = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
        strict_success_ever = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
        visual_success_ever = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
        lifted_frame_counts = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
        strict_success_frame_counts = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
        visual_success_frame_counts = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
        max_object_z = torch.full((args_cli.num_envs,), -float("inf"), dtype=torch.float32, device=device)
        min_keypoint_dist_per_env = torch.full((args_cli.num_envs,), float("inf"), dtype=torch.float32, device=device)
        min_object_goal_pos_dist_per_env = torch.full(
            (args_cli.num_envs,), float("inf"), dtype=torch.float32, device=device
        )
        min_quat_angle_error_per_env = torch.full(
            (args_cli.num_envs,), float("inf"), dtype=torch.float32, device=device
        )
        sequence_positions = sequence_quats = None
        sequence_target_indices = sequence_completed_counts = sequence_reach_steps = sequence_hit_streak = None
        sequence_events: list[dict[str, Any]] = []
        sequence_len = 0
        manual_goal_control = args_cli.goal_trajectory != "native"
        sequence_hold_steps = (
            int(args_cli.sequence_goal_hold_steps)
            if args_cli.sequence_goal_hold_steps is not None
            else int(env.unwrapped.cfg.success_steps)
        )
        if args_cli.gated_goal_sequence:
            sequence_positions, sequence_quats, trajectory_metadata = _goal_sequence_for_args(args_cli, device)
            sequence_len = int(sequence_positions.shape[0])
            sequence_target_indices = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
            sequence_completed_counts = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
            sequence_reach_steps = -torch.ones((args_cli.num_envs, sequence_len), dtype=torch.long, device=device)
            sequence_hit_streak = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
            _apply_gated_goal_sequence(env, sequence_positions, sequence_quats, sequence_target_indices)
        elif manual_goal_control:
            trajectory_metadata = _apply_goal_trajectory(env, args_cli, 0, device)
        else:
            trajectory_metadata = {
                "mode": "native",
                "gated": False,
                "goal_sampling_type": env.unwrapped.cfg.goal_sampling_type,
                "delta_goal_distance": float(env.unwrapped.cfg.delta_goal_distance),
                "delta_rotation_degrees": float(env.unwrapped.cfg.delta_rotation_degrees),
                "target_volume_min": env.unwrapped.target_volume_min.detach().cpu().tolist(),
                "target_volume_max": env.unwrapped.target_volume_max.detach().cpu().tolist(),
            }
        reward_success_tolerance = float(env.unwrapped.cfg.success_tolerance * env.unwrapped.cfg.keypoint_scale)
        eval_success_tolerance = (
            float(args_cli.eval_success_tolerance)
            if args_cli.eval_success_tolerance is not None
            else reward_success_tolerance
        )
        visual_success_tolerance = float(args_cli.visual_success_tolerance)
        diagnostic_tolerances = sorted(
            {
                round(float(tolerance), 6)
                for tolerance in [
                    *args_cli.diagnostic_tolerances,
                    visual_success_tolerance,
                    eval_success_tolerance,
                    reward_success_tolerance,
                ]
            }
        )
        diagnostic_tolerance_tensor = torch.tensor(diagnostic_tolerances, dtype=torch.float32, device=device)
        diagnostic_near_ever = torch.zeros(
            (len(diagnostic_tolerances), args_cli.num_envs), dtype=torch.bool, device=device
        )
        diagnostic_success_ever = torch.zeros_like(diagnostic_near_ever)
        diagnostic_near_frame_counts = torch.zeros(
            (len(diagnostic_tolerances), args_cli.num_envs), dtype=torch.long, device=device
        )
        diagnostic_success_frame_counts = torch.zeros_like(diagnostic_near_frame_counts)
        render_env_trace: list[dict[str, Any]] = []
        policy_obs = env.unwrapped._get_observations()["policy"].to(device)
        with imageio.get_writer(str(video_path), fps=args_cli.fps, macro_block_size=1) as writer:
            with torch.inference_mode():
                for step in range(args_cli.steps):
                    if args_cli.gated_goal_sequence:
                        assert sequence_positions is not None and sequence_quats is not None
                        assert sequence_target_indices is not None
                        _apply_gated_goal_sequence(env, sequence_positions, sequence_quats, sequence_target_indices)
                    elif manual_goal_control:
                        trajectory_metadata = _apply_goal_trajectory(env, args_cli, step, device)
                    policy_obs = env.unwrapped._get_observations()["policy"].to(device)
                    norm_policy = obs_rms.normalize(policy_obs) if obs_rms is not None else policy_obs
                    norm_policy = _append_coef(norm_policy, env_coef)
                    mu, sigma, actor_state = agent.actor.forward_step(norm_policy, actor_state, next_done)
                    if args_cli.deterministic:
                        action = mu
                    else:
                        action = Normal(mu, sigma).sample()
                    obs, _reward, terminated, truncated, _info = env.step(action)
                    next_done = torch.logical_or(terminated, truncated).to(device)
                    policy_obs = obs["policy"].to(device)

                    keypoint_dist = env.unwrapped.keypoints_max_dist_for_reward.detach()
                    object_goal_pos_dist = torch.linalg.norm(
                        env.unwrapped.object_pos - env.unwrapped.goal_pos, dim=-1
                    ).detach()
                    quat_angle_error = _quat_angle_error_deg(env.unwrapped.object_rot, env.unwrapped.goal_rot).detach()
                    lifted_object = env.unwrapped.lifted_object.detach()
                    ever_lifted |= env.unwrapped.lifted_object.detach()
                    max_object_z = torch.maximum(max_object_z, env.unwrapped.object_pos[:, 2].detach())
                    min_keypoint_dist_per_env = torch.minimum(min_keypoint_dist_per_env, keypoint_dist)
                    min_object_goal_pos_dist_per_env = torch.minimum(
                        min_object_goal_pos_dist_per_env, object_goal_pos_dist
                    )
                    min_quat_angle_error_per_env = torch.minimum(min_quat_angle_error_per_env, quat_angle_error)
                    reached_current_goal = _strict_current_success(
                        env,
                        eval_success_tolerance,
                        require_lifted=args_cli.strict_success_requires_lift,
                    )
                    reward_success = keypoint_dist <= reward_success_tolerance
                    if args_cli.strict_success_requires_lift:
                        reward_success &= lifted_object
                    visual_success = keypoint_dist <= visual_success_tolerance
                    if args_cli.strict_success_requires_lift:
                        visual_success &= lifted_object
                    near_by_tolerance = keypoint_dist.unsqueeze(0) <= diagnostic_tolerance_tensor.unsqueeze(1)
                    success_by_tolerance = near_by_tolerance
                    if args_cli.strict_success_requires_lift:
                        success_by_tolerance &= lifted_object.unsqueeze(0)
                    diagnostic_near_ever |= near_by_tolerance
                    diagnostic_success_ever |= success_by_tolerance
                    diagnostic_near_frame_counts += near_by_tolerance.long()
                    diagnostic_success_frame_counts += success_by_tolerance.long()
                    strict_success_ever |= reached_current_goal
                    visual_success_ever |= visual_success
                    lifted_frame_counts += lifted_object.long()
                    strict_success_frame_counts += reached_current_goal.long()
                    visual_success_frame_counts += visual_success.long()
                    if args_cli.gated_goal_sequence:
                        assert sequence_target_indices is not None
                        assert sequence_completed_counts is not None
                        assert sequence_reach_steps is not None
                        assert sequence_hit_streak is not None
                        sequence_events.extend(
                            _advance_gated_goal_sequence(
                                env,
                                step=step,
                                target_indices=sequence_target_indices,
                                completed_counts=sequence_completed_counts,
                                reach_steps=sequence_reach_steps,
                                hit_streak=sequence_hit_streak,
                                reached_current_goal=reached_current_goal,
                                hold_steps=sequence_hold_steps,
                                sequence_len=sequence_len,
                            )
                        )
                    if manual_goal_control:
                        env.unwrapped.reset_goal_buf[:] = False
                    if len(render_env_trace) < args_cli.steps:
                        render_env_trace.append(
                            {
                                "step": step,
                                "gate_success": bool(reached_current_goal[render_env_id].item()),
                                "strict_success": bool(reached_current_goal[render_env_id].item()),
                                "reward_success": bool(reward_success[render_env_id].item()),
                                "visual_success": bool(visual_success[render_env_id].item()),
                                "lifted": bool(lifted_object[render_env_id].item()),
                                "keypoint_dist": float(keypoint_dist[render_env_id].item()),
                                "object_goal_pos_dist": float(object_goal_pos_dist[render_env_id].item()),
                                "quat_angle_error_deg": float(quat_angle_error[render_env_id].item()),
                                "env_success_count": float(env.unwrapped.successes[render_env_id].item()),
                                "near_goal_steps": int(env.unwrapped.near_goal_steps[render_env_id].item()),
                                "target_index": int(sequence_target_indices[render_env_id].item())
                                if sequence_target_indices is not None
                                else None,
                            }
                        )
                    dist = float(keypoint_dist[render_env_id].item())
                    min_dist, min_env_id = torch.min(keypoint_dist, dim=0)
                    if float(min_dist.item()) < best_all_env_dist:
                        best_all_env_dist = float(min_dist.item())
                        best_all_env_id = int(min_env_id.item())
                    max_successes = torch.maximum(max_successes, env.unwrapped.successes.detach())
                    frame = _capture_frame(env)
                    writer.append_data(frame)
                    final_frame = frame
                    if dist < best_dist:
                        best_dist = dist
                        best_frame = frame.copy()

        if final_frame is not None:
            imageio.imwrite(final_image_path, final_frame)
        if best_frame is not None:
            imageio.imwrite(best_image_path, best_frame)

        final_keypoint_dist = env.unwrapped.keypoints_max_dist_for_reward.detach().cpu()
        final_object_goal_dist = torch.linalg.norm(env.unwrapped.object_pos - env.unwrapped.goal_pos, dim=-1).detach().cpu()
        final_quat_angle_error = _quat_angle_error_deg(env.unwrapped.object_rot, env.unwrapped.goal_rot).detach().cpu()
        sequence_stats: dict[str, Any]
        if args_cli.gated_goal_sequence:
            assert sequence_completed_counts is not None
            assert sequence_reach_steps is not None
            assert sequence_hit_streak is not None
            sequence_complete = sequence_completed_counts >= sequence_len
            best_sequence_env_id = int(torch.argmax(sequence_completed_counts).item())
            sequence_stats = {
                "gated_goal_sequence": True,
                "sequence_target_count": sequence_len,
                "sequence_goal_hold_steps": sequence_hold_steps,
                "sequence_completion_rate": float(sequence_complete.float().mean().item()),
                "completed_target_count_mean": float(sequence_completed_counts.float().mean().item()),
                "completed_target_count_max": int(sequence_completed_counts.max().item()),
                "best_sequence_env_id": best_sequence_env_id,
                "render_env_completed_target_count": int(sequence_completed_counts[render_env_id].item()),
                "render_env_sequence_complete": bool(sequence_complete[render_env_id].item()),
                "sequence_completed_counts": sequence_completed_counts.detach().cpu().tolist(),
                "sequence_reach_steps": sequence_reach_steps.detach().cpu().tolist(),
                "sequence_hit_streak_final": sequence_hit_streak.detach().cpu().tolist(),
                "sequence_events": sequence_events,
            }
        else:
            sequence_stats = {"gated_goal_sequence": False}
        visual_idx = diagnostic_tolerances.index(round(visual_success_tolerance, 6))
        reward_idx = diagnostic_tolerances.index(round(reward_success_tolerance, 6))
        gate_idx = diagnostic_tolerances.index(round(eval_success_tolerance, 6))
        diagnostic_threshold_stats = []
        for idx, tolerance in enumerate(diagnostic_tolerances):
            near_counts = diagnostic_near_frame_counts[idx].float()
            success_counts = diagnostic_success_frame_counts[idx].float()
            diagnostic_threshold_stats.append(
                {
                    "tolerance": float(tolerance),
                    "near_rate_any": float(diagnostic_near_ever[idx].float().mean().item()),
                    "success_rate_any": float(diagnostic_success_ever[idx].float().mean().item()),
                    "near_frame_rate_mean": float((near_counts / args_cli.steps).mean().item()),
                    "success_frame_rate_mean": float((success_counts / args_cli.steps).mean().item()),
                    "render_env_near_frame_rate": float(near_counts[render_env_id].item() / args_cli.steps),
                    "render_env_success_frame_rate": float(success_counts[render_env_id].item() / args_cli.steps),
                    "render_env_near_ever": bool(diagnostic_near_ever[idx, render_env_id].item()),
                    "render_env_success_ever": bool(diagnostic_success_ever[idx, render_env_id].item()),
                }
            )
        min_dist_quantiles = torch.quantile(
            min_keypoint_dist_per_env,
            torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32, device=device),
        )
        env_summary_rows = []
        for env_id in range(args_cli.num_envs):
            row: dict[str, Any] = {
                "env_id": env_id,
                "min_keypoint_dist": float(min_keypoint_dist_per_env[env_id].item()),
                "min_object_goal_pos_dist": float(min_object_goal_pos_dist_per_env[env_id].item()),
                "min_quat_angle_error_deg": float(min_quat_angle_error_per_env[env_id].item()),
                "max_object_z": float(max_object_z[env_id].item()),
                "lifted_frame_rate": float(lifted_frame_counts[env_id].float().item() / args_cli.steps),
                "legacy_success_count": float(max_successes[env_id].item()),
            }
            for idx, tolerance in enumerate(diagnostic_tolerances):
                suffix = f"{tolerance:.6f}".rstrip("0").rstrip(".").replace(".", "p")
                row[f"success_any_tol_{suffix}"] = bool(diagnostic_success_ever[idx, env_id].item())
                row[f"success_frame_rate_tol_{suffix}"] = float(
                    diagnostic_success_frame_counts[idx, env_id].float().item() / args_cli.steps
                )
                row[f"near_any_tol_{suffix}"] = bool(diagnostic_near_ever[idx, env_id].item())
                row[f"near_frame_rate_tol_{suffix}"] = float(
                    diagnostic_near_frame_counts[idx, env_id].float().item() / args_cli.steps
                )
            env_summary_rows.append(row)
        stats = {
            "checkpoint": str(args_cli.checkpoint),
            "checkpoint_update": int(checkpoint.get("update", 0)),
            "checkpoint_global_step": int(checkpoint.get("global_step", 0)),
            "goal_trajectory": trajectory_metadata,
            "num_envs": args_cli.num_envs,
            "steps": args_cli.steps,
            "deterministic": args_cli.deterministic,
            "render_env_id": int(render_env_id),
            "video_path": str(video_path),
            "final_image_path": str(final_image_path),
            "best_image_path": str(best_image_path),
            "render_trace_csv_path": str(render_trace_csv_path),
            "env_summary_csv_path": str(env_summary_csv_path),
            "best_render_env_keypoint_dist": best_dist,
            "best_all_env_keypoint_dist": best_all_env_dist,
            "best_all_env_id": best_all_env_id,
            "min_keypoint_dist_quantiles": {
                "min": float(min_dist_quantiles[0].item()),
                "p25": float(min_dist_quantiles[1].item()),
                "median": float(min_dist_quantiles[2].item()),
                "p75": float(min_dist_quantiles[3].item()),
                "max": float(min_dist_quantiles[4].item()),
            },
            "final_mean_keypoint_dist": float(final_keypoint_dist.mean().item()),
            "final_min_keypoint_dist": float(final_keypoint_dist.min().item()),
            "final_render_env_keypoint_dist": float(final_keypoint_dist[render_env_id].item()),
            "final_mean_object_goal_pos_dist": float(final_object_goal_dist.mean().item()),
            "final_min_object_goal_pos_dist": float(final_object_goal_dist.min().item()),
            "final_mean_quat_angle_error_deg": float(final_quat_angle_error.mean().item()),
            "final_min_quat_angle_error_deg": float(final_quat_angle_error.min().item()),
            "final_render_env_quat_angle_error_deg": float(final_quat_angle_error[render_env_id].item()),
            "max_successes_per_env": max_successes.detach().cpu().tolist(),
            "legacy_env_success_rate_any": float((max_successes > 0).float().mean().item()),
            "success_rate_any": float(visual_success_ever.float().mean().item()),
            "reward_tolerance_success_rate_any": float(diagnostic_success_ever[reward_idx].float().mean().item()),
            "train_tolerance_success_rate_any": float(diagnostic_success_ever[reward_idx].float().mean().item()),
            "gate_success_rate_any": float(strict_success_ever.float().mean().item()),
            "strict_success_rate_any": float(strict_success_ever.float().mean().item()),
            "visual_success_rate_any": float(visual_success_ever.float().mean().item()),
            "strict_success_requires_lift": bool(args_cli.strict_success_requires_lift),
            "eval_success_tolerance": eval_success_tolerance,
            "gate_success_tolerance": eval_success_tolerance,
            "reward_success_tolerance": reward_success_tolerance,
            "visual_success_tolerance": visual_success_tolerance,
            "diagnostic_thresholds": diagnostic_threshold_stats,
            "strict_success_ever": strict_success_ever.detach().cpu().tolist(),
            "visual_success_ever": visual_success_ever.detach().cpu().tolist(),
            "render_env_strict_success_ever": bool(strict_success_ever[render_env_id].item()),
            "render_env_visual_success_ever": bool(visual_success_ever[render_env_id].item()),
            "render_env_reward_tolerance_success_ever": bool(diagnostic_success_ever[reward_idx, render_env_id].item()),
            "strict_success_frame_rate_mean": float((strict_success_frame_counts.float() / args_cli.steps).mean().item()),
            "render_env_strict_success_frame_rate": float(strict_success_frame_counts[render_env_id].float().item() / args_cli.steps),
            "visual_success_frame_rate_mean": float((visual_success_frame_counts.float() / args_cli.steps).mean().item()),
            "render_env_visual_success_frame_rate": float(visual_success_frame_counts[render_env_id].float().item() / args_cli.steps),
            "reward_tolerance_success_frame_rate_mean": float(
                (diagnostic_success_frame_counts[reward_idx].float() / args_cli.steps).mean().item()
            ),
            "render_env_reward_tolerance_success_frame_rate": float(
                diagnostic_success_frame_counts[reward_idx, render_env_id].float().item() / args_cli.steps
            ),
            "gate_success_frame_rate_mean": float(
                (diagnostic_success_frame_counts[gate_idx].float() / args_cli.steps).mean().item()
            ),
            "render_env_gate_success_frame_rate": float(
                diagnostic_success_frame_counts[gate_idx, render_env_id].float().item() / args_cli.steps
            ),
            "lifted_frame_rate_mean": float((lifted_frame_counts.float() / args_cli.steps).mean().item()),
            "render_env_lifted_frame_rate": float(lifted_frame_counts[render_env_id].float().item() / args_cli.steps),
            "render_env_trace": render_env_trace,
            "ever_lifted_rate": float(ever_lifted.float().mean().item()),
            "render_env_ever_lifted": bool(ever_lifted[render_env_id].item()),
            "max_object_z_mean": float(max_object_z.mean().item()),
            "render_env_max_object_z": float(max_object_z[render_env_id].item()),
            **sequence_stats,
        }
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        if render_env_trace:
            with render_trace_csv_path.open("w", newline="", encoding="utf-8") as trace_file:
                writer = csv.DictWriter(trace_file, fieldnames=list(render_env_trace[0].keys()))
                writer.writeheader()
                writer.writerows(render_env_trace)
        if env_summary_rows:
            with env_summary_csv_path.open("w", newline="", encoding="utf-8") as summary_file:
                writer = csv.DictWriter(summary_file, fieldnames=list(env_summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(env_summary_rows)
        print(f"[EVAL] video={video_path}", flush=True)
        print(f"[EVAL] final_image={final_image_path}", flush=True)
        print(f"[EVAL] best_image={best_image_path}", flush=True)
        print(f"[EVAL] stats={stats_path}", flush=True)
        print(f"[EVAL] render_trace_csv={render_trace_csv_path}", flush=True)
        print(f"[EVAL] env_summary_csv={env_summary_csv_path}", flush=True)
        print(
            "[EVAL] "
            f"visual_success_rate_any={stats['visual_success_rate_any']:.4f} "
            f"reward_tol_success_rate_any={stats['reward_tolerance_success_rate_any']:.4f} "
            f"gate_success_rate_any={stats['gate_success_rate_any']:.4f} "
            f"legacy_env_success_rate_any={stats['legacy_env_success_rate_any']:.4f} "
            f"sequence_completion_rate={stats.get('sequence_completion_rate', 0.0):.4f} "
            f"render_env_completed_targets={stats.get('render_env_completed_target_count', 0)} "
            f"render_env_visual_success={stats['render_env_visual_success_ever']} "
            f"render_env_visual_frame_rate={stats['render_env_visual_success_frame_rate']:.4f} "
            f"render_env_reward_tol_frame_rate={stats['render_env_reward_tolerance_success_frame_rate']:.4f} "
            f"render_env_gate_frame_rate={stats['render_env_gate_success_frame_rate']:.4f} "
            f"render_env_lifted_frame_rate={stats['render_env_lifted_frame_rate']:.4f} "
            f"render_env_lifted={stats['render_env_ever_lifted']} "
            f"best_render_env_keypoint_dist={best_dist:.4f} "
            f"final_mean_keypoint_dist={stats['final_mean_keypoint_dist']:.4f}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"[EVAL][ERROR] {type(err).__name__}: {err}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
