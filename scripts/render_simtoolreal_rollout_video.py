#!/usr/bin/env python3
"""Render a random-policy SimToolReal Isaac Lab rollout video."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ISAAC_SRC = REPO_ROOT / "src" / "isaaclab_env"
if str(LOCAL_ISAAC_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_ISAAC_SRC))

from isaaclab.app import AppLauncher


class RolloutAppLauncher(AppLauncher):
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


parser = argparse.ArgumentParser(description="Render a random-policy SimToolReal rollout mp4.")
parser.add_argument("--task", type=str, default="SimToolReal-Direct-Debug-v0", help="Gym task ID to render.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments.")
parser.add_argument("--steps", type=int, default=180, help="Number of rollout/render steps.")
parser.add_argument("--fps", type=int, default=30, help="Output video FPS.")
parser.add_argument("--action_scale", type=float, default=1.0, help="Uniform random action range multiplier.")
parser.add_argument("--hold_steps", type=int, default=24, help="Number of frames to hold each sampled random action.")
parser.add_argument("--warmup_frames", type=int, default=20, help="Render frames to discard before writing the mp4.")
parser.add_argument("--post_step_renders", type=int, default=2, help="Fresh render calls after each env step before reading RGB.")
parser.add_argument(
    "--no_motion_boost",
    action="store_true",
    help="Keep env control smoothing/delay unchanged. By default, render rollout is boosted to make motion visible.",
)
parser.add_argument(
    "--output",
    type=Path,
    default=REPO_ROOT / "outputs" / "isaaclab_renders" / "rollout_video" / "simtoolreal_random_rollout.mp4",
)
parser.add_argument("--disable_fabric", action="store_true", help="Disable Fabric scene cloning/render path.")
parser.add_argument("--env_device", type=str, default=None, help="Optional env device override.")
parser.add_argument("--kit_active_gpu", type=int, default=None, help="Optional Kit renderer activeGpu override.")
parser.add_argument("--kit_physics_gpu", type=int, default=None, help="Optional Kit /physics/cudaDevice override.")
parser.add_argument("--seed", type=int, default=7, help="Random seed.")
RolloutAppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, enable_cameras=True)
args_cli = parser.parse_args()

app_launcher = RolloutAppLauncher(
    args_cli,
    active_gpu_override=args_cli.kit_active_gpu,
    physics_gpu_override=args_cli.kit_physics_gpu,
)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import isaaclab_env.tasks  # noqa: F401, E402


def _set_camera(env) -> tuple[list[float], list[float]]:
    origins = env.unwrapped.scene.env_origins[: env.unwrapped.num_envs].detach().cpu()
    center = origins.mean(dim=0)
    spread = torch.linalg.norm(origins.max(dim=0).values - origins.min(dim=0).values).item()
    distance = max(2.8, spread * 1.4)
    eye = center + torch.tensor([distance, -distance, 1.65])
    target = center + torch.tensor([0.0, 0.38, 0.52])
    env.unwrapped.sim.set_camera_view(
        eye=eye.tolist(),
        target=target.tolist(),
        camera_prim_path=env.unwrapped.cfg.viewer.cam_prim_path,
    )
    return eye.tolist(), target.tolist()


def _capture_rgb_frame(env) -> np.ndarray:
    for _ in range(max(1, args_cli.post_step_renders)):
        env.unwrapped.sim.render()
    frame = np.asarray(env.render())
    if frame.size == 0:
        raise RuntimeError("env.render() returned an empty frame.")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def main() -> None:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    env_device = args_cli.env_device or args_cli.device
    print(f"[ROLLOUT] task={args_cli.task} app_device={args_cli.device} env_device={env_device}", flush=True)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=env_device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.viewer.resolution = (1280, 720)
    env_cfg.viewer.cam_prim_path = "/OmniverseKit_Persp"
    if not args_cli.no_motion_boost:
        env_cfg.use_action_delay = False
        env_cfg.dof_speed_scale = 4.0
        env_cfg.arm_moving_average = 0.6
        env_cfg.hand_moving_average = 0.6

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    try:
        env.reset(seed=args_cli.seed)
        eye, target = _set_camera(env)
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        zero_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        with torch.inference_mode():
            for _ in range(args_cli.warmup_frames):
                env.step(zero_actions)
                _capture_rgb_frame(env)

        start_joint_pos = env.unwrapped.robot.data.joint_pos[:, env.unwrapped.actuated_joint_ids].clone()
        start_target = env.unwrapped.cur_targets.clone()
        start_object_pos = env.unwrapped.object.data.root_pos_w.clone()
        start_body_pos = env.unwrapped.robot.data.body_pos_w[:, env.unwrapped.fingertip_body_ids + [env.unwrapped.palm_body_id]].clone()
        max_abs_action = 0.0

        with imageio.get_writer(str(args_cli.output), fps=args_cli.fps, macro_block_size=1) as writer:
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                for step in range(args_cli.steps):
                    if step % args_cli.hold_steps == 0:
                        actions = (
                            2.0 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1.0
                        ) * args_cli.action_scale
                    max_abs_action = max(max_abs_action, float(actions.abs().max().item()))
                    env.step(actions)
                    writer.append_data(_capture_rgb_frame(env))

        joint_delta = (env.unwrapped.robot.data.joint_pos[:, env.unwrapped.actuated_joint_ids] - start_joint_pos).abs()
        target_delta = (env.unwrapped.cur_targets - start_target).abs()
        object_delta = (env.unwrapped.object.data.root_pos_w - start_object_pos).abs()
        body_delta = (
            env.unwrapped.robot.data.body_pos_w[:, env.unwrapped.fingertip_body_ids + [env.unwrapped.palm_body_id]] - start_body_pos
        ).abs()
        print(f"[ROLLOUT] saved={args_cli.output}", flush=True)
        print(f"[ROLLOUT] camera_eye={eye} camera_target={target}", flush=True)
        print(
            "[ROLLOUT] "
            f"max_abs_action={max_abs_action:.4f} "
            f"joint_delta_max={float(joint_delta.max().item()):.6f} "
            f"target_delta_max={float(target_delta.max().item()):.6f} "
            f"body_delta_max={float(body_delta.max().item()):.6f} "
            f"object_delta_max={float(object_delta.max().item()):.6f}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"[ROLLOUT][ERROR] {type(err).__name__}: {err}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
