#!/usr/bin/env python3
"""Export a short SimToolReal Isaac Lab rollout for external 3D visualization.

The output is intentionally simple:

* ``manifest.json`` stores static scene metadata, asset paths, joint names, and
  body names.
* ``frames.jsonl`` stores one JSON object per simulation frame.

This is meant to debug viewer alignment.  Each frame includes both joint
positions (for URDF playback) and Isaac Lab body poses (ground truth for
checking root transforms, joint order, and quaternion conventions).
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ISAAC_SRC = REPO_ROOT / "src" / "isaaclab_env"
if str(LOCAL_ISAAC_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_ISAAC_SRC))

from isaaclab.app import AppLauncher


class VizLogAppLauncher(AppLauncher):
    """AppLauncher variant that can decouple Kit renderer and PhysX CUDA ordinals."""

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


parser = argparse.ArgumentParser(description="Export SimToolReal Isaac Lab state logs for the website visualizer.")
parser.add_argument("--task", type=str, default="SimToolReal-Direct-Debug-v0", help="Gym task ID to log.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of envs to simulate and log.")
parser.add_argument("--steps", type=int, default=120, help="Number of simulation steps to export after reset.")
parser.add_argument("--log_every", type=int, default=1, help="Write one frame every N env steps.")
parser.add_argument("--action_mode", choices=("zero", "random"), default="random", help="Action source for the rollout.")
parser.add_argument("--action_scale", type=float, default=0.8, help="Scale for random actions before clamping to [-1, 1].")
parser.add_argument("--seed", type=int, default=7, help="Seed for env reset and random actions.")
parser.add_argument("--run_name", type=str, default=None, help="Optional output folder name.")
parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "outputs" / "viz_logs")
parser.add_argument("--disable_fabric", action="store_true", help="Disable Fabric scene cloning/render path.")
parser.add_argument("--env_device", type=str, default=None, help="Optional env device override. Defaults to AppLauncher --device.")
parser.add_argument("--kit_active_gpu", type=int, default=None, help="Optional Kit renderer activeGpu override.")
parser.add_argument("--kit_physics_gpu", type=int, default=None, help="Optional Kit /physics/cudaDevice override.")
VizLogAppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = VizLogAppLauncher(
    args_cli,
    active_gpu_override=args_cli.kit_active_gpu,
    physics_gpu_override=args_cli.kit_physics_gpu,
)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import isaaclab_env.tasks  # noqa: F401, E402


def _relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _tolist(value: torch.Tensor | Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return float(value.item())
        return value.tolist()
    return value


def _pose_from_state(root_state: torch.Tensor) -> dict[str, Any]:
    return {
        "pos": _tolist(root_state[:3]),
        "quat_wxyz": _tolist(root_state[3:7]),
        "lin_vel": _tolist(root_state[7:10]),
        "ang_vel": _tolist(root_state[10:13]),
    }


def _variant_to_dict(variant: Any) -> dict[str, Any]:
    return {
        "object_type": variant.object_type,
        "urdf_path": str(variant.urdf_path),
        "urdf_path_repo": _relpath(variant.urdf_path),
        "usd_path": str(variant.usd_path),
        "usd_path_repo": _relpath(variant.usd_path),
        "object_scale": list(variant.object_scale),
        "handle_scale": list(variant.handle_scale),
        "head_scale": None if variant.head_scale is None else list(variant.head_scale),
        "handle_density": variant.handle_density,
        "head_density": variant.head_density,
        "mass": variant.mass,
        "center_of_mass": list(variant.center_of_mass),
        "diagonal_inertia": list(variant.diagonal_inertia),
    }


def _make_manifest(env, env_cfg, args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    unwrapped = env.unwrapped
    actuated_joint_names = [unwrapped.robot.joint_names[index] for index in unwrapped.actuated_joint_ids]
    object_variants = [_variant_to_dict(variant) for variant in unwrapped.cfg.object_variants]
    env_variant_ids = _tolist(unwrapped.object_variant_ids)
    env_object_variants = [
        {"env_id": env_id, "variant_id": int(variant_id), **object_variants[int(variant_id)]}
        for env_id, variant_id in enumerate(env_variant_ids)
    ]

    return {
        "schema_version": 1,
        "format": {
            "manifest": "static metadata",
            "frames": "one JSON object per line in frames.jsonl",
        },
        "source": {
            "sim": "isaaclab",
            "task": args.task,
            "repo_root": str(REPO_ROOT),
            "run_dir": str(run_dir),
        },
        "timing": {
            "sim_dt": float(env_cfg.sim.dt),
            "decimation": int(env_cfg.decimation),
            "control_dt": float(unwrapped.control_dt),
            "steps": int(args.steps),
            "log_every": int(args.log_every),
        },
        "coordinate_system": {
            "world_frame": "Isaac Lab / PhysX world, meters, Z-up",
            "quaternion_order": "wxyz",
            "website_three_position_hint": "Isaac [x, y, z] -> Three [x, z, -y]",
            "note": "Use robot.link_pos_w/link_quat_wxyz as ground truth to debug URDF root and joint transforms.",
        },
        "scene": {
            "num_envs": int(unwrapped.num_envs),
            "env_spacing": float(env_cfg.scene.env_spacing),
            "env_origins_w": _tolist(unwrapped.scene.env_origins),
            "replicate_physics": bool(env_cfg.scene.replicate_physics),
            "clone_in_fabric": bool(env_cfg.scene.clone_in_fabric),
        },
        "assets": {
            "robot": {
                "name": "kuka_sharpa",
                "urdf_path": str(REPO_ROOT / "assets" / "urdf" / "kuka_sharpa_description" / "iiwa14_left_sharpa_adjusted_restricted.urdf"),
                "urdf_path_repo": "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf",
                "website_public_path": "/simtoolreal_assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf",
                "prim_path": env_cfg.robot_cfg.prim_path,
            },
            "table": {
                "kind": "cuboid",
                "size": list(env_cfg.table_cfg.spawn.size),
                "prim_path": env_cfg.table_cfg.prim_path,
            },
            "object": {
                "prim_path": env_cfg.object_cfg.prim_path,
                "variants": object_variants,
                "env_assignments": env_object_variants,
            },
            "goal_object": {
                "prim_path": env_cfg.goal_object_cfg.prim_path,
                "uses_same_variant_assignment_as_object": True,
            },
        },
        "robot_model": {
            "joint_names": list(unwrapped.robot.joint_names),
            "actuated_joint_names": actuated_joint_names,
            "actuated_joint_ids": [int(index) for index in unwrapped.actuated_joint_ids],
            "body_names": list(unwrapped.robot.body_names),
            "default_joint_pos": _tolist(unwrapped.default_joint_pos[0]),
            "joint_lower_limits": _tolist(unwrapped.joint_lower_limits),
            "joint_upper_limits": _tolist(unwrapped.joint_upper_limits),
        },
        "rollout": {
            "action_mode": args.action_mode,
            "action_scale": float(args.action_scale),
            "seed": int(args.seed),
        },
    }


def _make_frame(env, obs: dict[str, torch.Tensor], reward: torch.Tensor, done: torch.Tensor, step: int) -> dict[str, Any]:
    unwrapped = env.unwrapped
    env_origins = unwrapped.scene.env_origins
    robot_root_w = unwrapped.robot.data.root_state_w
    robot_body_w = unwrapped.robot.data.body_state_w
    object_w = unwrapped.object.data.root_state_w
    goal_w = unwrapped.goal_object.data.root_state_w
    table_w = unwrapped.table.data.root_state_w

    env_entries = []
    for env_id in range(unwrapped.num_envs):
        env_origin = env_origins[env_id]
        body_pos_w = robot_body_w[env_id, :, :3]
        object_state = object_w[env_id]
        goal_state = goal_w[env_id]
        table_state = table_w[env_id]
        root_state = robot_root_w[env_id]

        env_entries.append(
            {
                "env_id": env_id,
                "env_origin_w": _tolist(env_origin),
                "episode_step": int(unwrapped.episode_length_buf[env_id].item()),
                "done": bool(done[env_id].item()),
                "reward": float(reward[env_id].item()),
                "successes": float(unwrapped.successes[env_id].item()),
                "robot": {
                    "root_pose_w": _pose_from_state(root_state),
                    "root_pos_env": _tolist(root_state[:3] - env_origin),
                    "joint_pos": _tolist(unwrapped.robot.data.joint_pos[env_id]),
                    "joint_vel": _tolist(unwrapped.robot.data.joint_vel[env_id]),
                    "joint_targets": _tolist(unwrapped.joint_targets[env_id]),
                    "actuated_joint_pos": _tolist(unwrapped.robot.data.joint_pos[env_id, unwrapped.actuated_joint_ids]),
                    "actuated_joint_targets": _tolist(unwrapped.joint_targets[env_id, unwrapped.actuated_joint_ids]),
                    "action": _tolist(unwrapped.actions[env_id]),
                    "body_pos_w": _tolist(body_pos_w),
                    "body_quat_wxyz": _tolist(robot_body_w[env_id, :, 3:7]),
                    "body_pos_env": _tolist(body_pos_w - env_origin.unsqueeze(0)),
                },
                "table": {
                    "pose_w": _pose_from_state(table_state),
                    "pos_env": _tolist(table_state[:3] - env_origin),
                },
                "object": {
                    "variant_id": int(unwrapped.object_variant_ids[env_id].item()),
                    "object_scale": _tolist(unwrapped.object_scales[env_id]),
                    "scale_noise_multiplier": _tolist(unwrapped.object_scale_noise_multiplier[env_id]),
                    "pose_w": _pose_from_state(object_state),
                    "pos_env": _tolist(object_state[:3] - env_origin),
                },
                "goal": {
                    "pose_w": _pose_from_state(goal_state),
                    "pos_env": _tolist(goal_state[:3] - env_origin),
                },
                "debug": {
                    "policy_obs": _tolist(obs["policy"][env_id]),
                    "critic_obs": _tolist(obs["critic"][env_id]),
                },
            }
        )

    return {
        "frame": step,
        "sim_time": float(step * unwrapped.control_dt),
        "envs": env_entries,
    }


def main() -> None:
    if args_cli.log_every <= 0:
        raise ValueError("--log_every must be positive")
    torch.manual_seed(args_cli.seed)

    env_device = args_cli.env_device or args_cli.device
    run_name = args_cli.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S_simtoolreal_viz_log")
    run_dir = args_cli.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    frames_path = run_dir / "frames.jsonl"

    print(f"[VIZLOG] task={args_cli.task} app_device={args_cli.device} env_device={env_device}", flush=True)
    print(f"[VIZLOG] output={run_dir}", flush=True)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=env_device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        obs, _ = env.reset(seed=args_cli.seed)
        reward = torch.zeros(args_cli.num_envs, device=env.unwrapped.device)
        done = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=env.unwrapped.device)

        manifest = _make_manifest(env, env_cfg, args_cli, run_dir)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        generator = torch.Generator(device=env.unwrapped.device)
        generator.manual_seed(args_cli.seed)

        frames_written = 0
        with frames_path.open("w", encoding="utf-8") as stream, torch.inference_mode():
            stream.write(json.dumps(_make_frame(env, obs, reward, done, step=0)) + "\n")
            frames_written += 1
            for step in range(1, args_cli.steps + 1):
                if args_cli.action_mode == "zero":
                    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                else:
                    action = (2.0 * torch.rand(env.action_space.shape, generator=generator, device=env.unwrapped.device) - 1.0)
                    action = torch.clamp(action * args_cli.action_scale, -1.0, 1.0)
                obs, reward, terminated, truncated, _info = env.step(action)
                done = terminated | truncated
                if step % args_cli.log_every == 0:
                    stream.write(json.dumps(_make_frame(env, obs, reward, done, step=step)) + "\n")
                    frames_written += 1

        print(f"[VIZLOG] manifest={manifest_path}", flush=True)
        print(f"[VIZLOG] frames={frames_path}", flush=True)
        print(f"[VIZLOG] frames_written={frames_written}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"[VIZLOG][ERROR] {type(err).__name__}: {err}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
