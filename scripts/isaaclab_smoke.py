#!/usr/bin/env python3
"""Minimal Isaac Lab smoke test for this migration branch."""

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


class SmokeAppLauncher(AppLauncher):
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


parser = argparse.ArgumentParser(description="Run a short Isaac Lab smoke test.")
parser.add_argument("--task", type=str, default="SimToolReal-Smoke-Cartpole-Direct-v0", help="Gym task ID to run.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of Cartpole environments.")
parser.add_argument("--steps", type=int, default=32, help="Number of simulation steps.")
parser.add_argument("--windowed", action="store_true", help="Launch Isaac Sim with a display instead of headless.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable Fabric scene cloning/render path.")
parser.add_argument(
    "--env_device",
    type=str,
    default=None,
    help="Optional Isaac Lab env device override. Defaults to the AppLauncher --device.",
)
parser.add_argument("--kit_active_gpu", type=int, default=None, help="Optional Kit renderer activeGpu override.")
parser.add_argument("--kit_physics_gpu", type=int, default=None, help="Optional Kit /physics/cudaDevice override.")
SmokeAppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()
if args_cli.windowed:
    args_cli.headless = False

app_launcher = SmokeAppLauncher(
    args_cli,
    active_gpu_override=args_cli.kit_active_gpu,
    physics_gpu_override=args_cli.kit_physics_gpu,
)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import isaaclab_env.tasks  # noqa: F401, E402


def main() -> None:
    env_device = args_cli.env_device or args_cli.device
    print(f"[SMOKE] parsing task={args_cli.task} app_device={args_cli.device} env_device={env_device}", flush=True)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=env_device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    print("[SMOKE] creating env", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        print("[SMOKE] resetting env", flush=True)
        obs, _ = env.reset()
        print(f"[SMOKE] reset ok: task={args_cli.task} policy_obs_shape={tuple(obs['policy'].shape)}", flush=True)

        reward_mean = 0.0
        with torch.inference_mode():
            for step in range(args_cli.steps):
                actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                obs, rew, terminated, truncated, info = env.step(actions)
                reward_mean = float(rew.mean().item())

        print(f"[SMOKE] stepped {args_cli.steps} steps: last_reward_mean={reward_mean:.6f}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"[SMOKE][ERROR] {type(err).__name__}: {err}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
