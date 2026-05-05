#!/usr/bin/env python3
"""Render a visible 2D Cartpole rollout from Isaac Lab state observations."""

import argparse
import math
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ISAAC_SRC = REPO_ROOT / "src" / "isaaclab_env"
if str(LOCAL_ISAAC_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_ISAAC_SRC))

from isaaclab.app import AppLauncher


class VizAppLauncher(AppLauncher):
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


parser = argparse.ArgumentParser(description="Render a 2D Cartpole state-visualization mp4.")
parser.add_argument("--task", type=str, default="SimToolReal-Smoke-Cartpole-Direct-v0")
parser.add_argument("--steps", type=int, default=180)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--kit_active_gpu", type=int, default=None)
parser.add_argument("--kit_physics_gpu", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = VizAppLauncher(
    args_cli,
    active_gpu_override=args_cli.kit_active_gpu,
    physics_gpu_override=args_cli.kit_physics_gpu,
)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import isaaclab_env.tasks  # noqa: F401, E402


def _collect_states() -> np.ndarray:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    states = []
    try:
        obs, _ = env.reset()
        states.append(obs["policy"][0].detach().cpu().numpy())
        with torch.inference_mode():
            for step in range(args_cli.steps - 1):
                action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                action[:, 0] = 0.75 * math.sin(step * 0.12)
                obs, _, _, _, _ = env.step(action)
                states.append(obs["policy"][0].detach().cpu().numpy())
    finally:
        env.close()
    return np.asarray(states, dtype=np.float32)


def _load_font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _render_frames(states: np.ndarray) -> list[np.ndarray]:
    width, height = 1280, 720
    track_y = 465
    center_x = width // 2
    cart_w, cart_h = 150, 70
    wheel_r = 16
    pole_len = 260
    scale = 135.0
    font = _load_font(24)
    small_font = _load_font(18)

    frames = []
    for idx, state in enumerate(states):
        pole_angle = float(state[0])
        pole_vel = float(state[1])
        cart_pos = float(state[2])
        cart_vel = float(state[3])

        img = Image.new("RGB", (width, height), (238, 241, 235))
        draw = ImageDraw.Draw(img)

        for x in range(80, width - 79, 80):
            draw.line((x, 120, x, 560), fill=(220, 225, 218), width=1)
        for y in range(160, 561, 80):
            draw.line((80, y, width - 80, y), fill=(220, 225, 218), width=1)
        draw.line((120, track_y, width - 120, track_y), fill=(70, 78, 73), width=6)
        draw.line((center_x - 3 * scale, track_y - 16, center_x - 3 * scale, track_y + 16), fill=(70, 78, 73), width=3)
        draw.line((center_x + 3 * scale, track_y - 16, center_x + 3 * scale, track_y + 16), fill=(70, 78, 73), width=3)

        cart_x = center_x + cart_pos * scale
        cart_y = track_y - cart_h // 2 - 18
        x0, y0 = cart_x - cart_w / 2, cart_y - cart_h / 2
        x1, y1 = cart_x + cart_w / 2, cart_y + cart_h / 2
        draw.rounded_rectangle((x0, y0, x1, y1), radius=14, fill=(36, 86, 123), outline=(18, 42, 60), width=4)
        draw.ellipse((cart_x - 55 - wheel_r, y1 - 2, cart_x - 55 + wheel_r, y1 + 2 * wheel_r), fill=(28, 31, 34))
        draw.ellipse((cart_x + 55 - wheel_r, y1 - 2, cart_x + 55 + wheel_r, y1 + 2 * wheel_r), fill=(28, 31, 34))

        pivot = (cart_x, y0 + 8)
        pole_tip = (pivot[0] + pole_len * math.sin(pole_angle), pivot[1] - pole_len * math.cos(pole_angle))
        draw.line((pivot[0], pivot[1], pole_tip[0], pole_tip[1]), fill=(204, 92, 42), width=14)
        draw.ellipse((pivot[0] - 13, pivot[1] - 13, pivot[0] + 13, pivot[1] + 13), fill=(250, 200, 87), outline=(98, 70, 28), width=3)
        draw.ellipse((pole_tip[0] - 11, pole_tip[1] - 11, pole_tip[0] + 11, pole_tip[1] + 11), fill=(204, 92, 42))

        draw.text((42, 34), "Isaac Lab Cartpole Smoke Rollout", fill=(28, 35, 32), font=font)
        draw.text((42, 66), "State visualization from the Isaac Lab env on RTX 5080", fill=(80, 88, 82), font=small_font)
        draw.text(
            (42, 618),
            (
                f"frame {idx + 1:03d}/{len(states)}  cart={cart_pos:+.2f} m  "
                f"pole={pole_angle:+.2f} rad  cart_vel={cart_vel:+.2f}  pole_vel={pole_vel:+.2f}"
            ),
            fill=(42, 48, 45),
            font=small_font,
        )
        frames.append(np.asarray(img))
    return frames


def main() -> None:
    states = _collect_states()
    frames = _render_frames(states)
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(args_cli.output), fps=args_cli.fps, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"[STATE_VIZ] video saved: {args_cli.output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
