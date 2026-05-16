#!/usr/bin/env python3
"""Evaluate a CleanRL SimToolReal checkpoint and save render evidence."""

from __future__ import annotations

import argparse
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
parser.add_argument("--goal_trajectory", choices=("fixed", "line", "circle", "arc", "strike"), default="fixed")
parser.add_argument("--goal_center", type=float, nargs=3, default=(0.0, 0.0, 0.78))
parser.add_argument("--goal_amplitude", type=float, default=0.10)
parser.add_argument("--goal_height_amplitude", type=float, default=0.06)
parser.add_argument("--goal_period", type=int, default=240)
parser.add_argument("--goal_pitch_amplitude_degrees", type=float, default=55.0)
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


def _capture_frame(env) -> np.ndarray:
    for _ in range(2):
        env.unwrapped.sim.render()
    frame = np.asarray(env.render())
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def main() -> None:
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
        trajectory_metadata = _apply_goal_trajectory(env, args_cli, 0, device)
        policy_obs = env.unwrapped._get_observations()["policy"].to(device)
        with imageio.get_writer(str(video_path), fps=args_cli.fps, macro_block_size=1) as writer:
            with torch.inference_mode():
                for step in range(args_cli.steps):
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
            "best_render_env_keypoint_dist": best_dist,
            "best_all_env_keypoint_dist": best_all_env_dist,
            "best_all_env_id": best_all_env_id,
            "final_mean_keypoint_dist": float(final_keypoint_dist.mean().item()),
            "final_min_keypoint_dist": float(final_keypoint_dist.min().item()),
            "final_render_env_keypoint_dist": float(final_keypoint_dist[render_env_id].item()),
            "final_mean_object_goal_pos_dist": float(final_object_goal_dist.mean().item()),
            "final_min_object_goal_pos_dist": float(final_object_goal_dist.min().item()),
            "max_successes_per_env": max_successes.detach().cpu().tolist(),
            "success_rate_any": float((max_successes > 0).float().mean().item()),
        }
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"[EVAL] video={video_path}", flush=True)
        print(f"[EVAL] final_image={final_image_path}", flush=True)
        print(f"[EVAL] best_image={best_image_path}", flush=True)
        print(f"[EVAL] stats={stats_path}", flush=True)
        print(
            "[EVAL] "
            f"success_rate_any={stats['success_rate_any']:.4f} "
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
