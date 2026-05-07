#!/usr/bin/env python3
"""Render a screenshot of the Isaac Lab SimToolReal direct environment."""

from __future__ import annotations

import argparse
import asyncio
import sys
import traceback
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ISAAC_SRC = REPO_ROOT / "src" / "isaaclab_env"
if str(LOCAL_ISAAC_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_ISAAC_SRC))

from isaaclab.app import AppLauncher


class RenderAppLauncher(AppLauncher):
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


parser = argparse.ArgumentParser(description="Render a SimToolReal Isaac Lab scene screenshot.")
parser.add_argument("--task", type=str, default="SimToolReal-Direct-Debug-v0", help="Gym task ID to render.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to render.")
parser.add_argument("--steps", type=int, default=8, help="Number of zero-action settling steps before capture.")
parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "isaaclab_renders" / "simtoolreal_two_envs.png")
parser.add_argument("--disable_fabric", action="store_true", help="Disable Fabric scene cloning/render path.")
parser.add_argument("--env_device", type=str, default=None, help="Optional env device override.")
parser.add_argument("--kit_active_gpu", type=int, default=None, help="Optional Kit renderer activeGpu override.")
parser.add_argument("--kit_physics_gpu", type=int, default=None, help="Optional Kit /physics/cudaDevice override.")
parser.add_argument("--camera_path", type=str, default="/OmniverseKit_Persp", help="USD camera prim used for capture.")
parser.add_argument(
    "--camera_view",
    choices=("iso", "front", "back", "left", "right", "top"),
    default="iso",
    help="Named camera angle for the scene capture.",
)
parser.add_argument(
    "--capture_backend",
    choices=("auto", "rgb_array", "viewport"),
    default="auto",
    help="Capture path. 'auto' tries rgb_array first, then viewport if the frame is empty/flat.",
)
parser.add_argument("--seed", type=int, default=7, help="Environment seed for repeatable scene initialization.")
RenderAppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, enable_cameras=True)
args_cli = parser.parse_args()

app_launcher = RenderAppLauncher(
    args_cli,
    active_gpu_override=args_cli.kit_active_gpu,
    physics_gpu_override=args_cli.kit_physics_gpu,
)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Sdf  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import isaaclab_env.tasks  # noqa: F401, E402


def _patch_syntheticdata_dependency_bug() -> None:
    """Work around an Isaac Sim 5.1 / NumPy 2.x SyntheticData dependency write crash."""
    import omni.graph.core as og
    import omni.graph.tools.ogn
    from omni.syntheticdata.scripts.SyntheticData import SyntheticData, SyntheticDataException

    def patched_add_dep(node, downstream_node_handle):
        if (not node.is_valid()) or (downstream_node_handle is None):
            return 0
        dependency_attribute_name = "state:_sdp_intergraph_downstream_node_handles_"
        if not node.get_attribute_exists(dependency_attribute_name):
            dep_attrib = og.Controller.create_attribute(
                node=node,
                attr_name=dependency_attribute_name,
                attr_type="uint64[]",
                attr_port=og.AttributePortType.ATTRIBUTE_PORT_TYPE_STATE,
                attr_extended_type=og.ExtendedAttributeType.REGULAR,
            )
            if (dep_attrib is None) or not node.get_attribute_exists(dependency_attribute_name):
                raise SyntheticDataException(
                    f"failed to create node dependency for {node.get_prim_path()} @ {downstream_node_handle}."
                )
            dep_attrib.set_metadata(omni.graph.tools.ogn.MetadataKeys.INTERNAL, "1")
            dep_attrib.set_metadata(omni.graph.tools.ogn.MetadataKeys.LITERAL_ONLY, "1")
        else:
            dep_attrib = node.get_attribute(dependency_attribute_name)
        dep_attrib_data = dep_attrib.get_attribute_data()
        existing = np.asarray(dep_attrib_data.get(), dtype=np.uint64).reshape(-1)
        dep_data = np.append(existing, np.uint64(downstream_node_handle)).astype(np.uint64, copy=False)
        og.AttributeValueHelper(dep_attrib).set(dep_data.tolist())
        return len(dep_data)

    def patched_remove_inactive_deps(node, active_node_handles):
        if not node.is_valid():
            return 0
        dependency_attribute_name = "state:_sdp_intergraph_downstream_node_handles_"
        if not node.get_attribute_exists(dependency_attribute_name):
            return 0
        dep_attrib = node.get_attribute(dependency_attribute_name)
        dep_attrib_data = dep_attrib.get_attribute_data()
        existing = np.asarray(dep_attrib_data.get(), dtype=np.uint64).reshape(-1)
        active = np.asarray(active_node_handles, dtype=np.uint64).reshape(-1)
        dep_data = np.intersect1d(existing, active).astype(np.uint64, copy=False)
        og.AttributeValueHelper(dep_attrib).set(dep_data.tolist())
        return len(dep_data)

    SyntheticData._add_node_downstream_intergraph_dependency = staticmethod(patched_add_dep)
    SyntheticData._remove_inactive_node_downstream_intergraph_dependencies = staticmethod(patched_remove_inactive_deps)


def _render_frame_rgb_array(env, warmup: int = 30) -> np.ndarray:
    frame = None
    for _ in range(warmup):
        frame = env.render()
    if frame is None:
        raise RuntimeError("env.render() returned None; render_mode='rgb_array' was not enabled.")
    frame = np.asarray(frame)
    if frame.size == 0:
        raise RuntimeError("env.render() returned an empty frame.")
    if float(frame.std()) < 0.5:
        raise RuntimeError(f"env.render() returned a flat frame with pixel={frame.reshape(-1, frame.shape[-1])[0].tolist()}.")
    return frame


async def _capture_viewport_async(camera_path: str, output_path: Path, warmup: int = 30) -> None:
    import omni.kit.app
    from omni.kit.viewport.utility import capture_viewport_to_file, create_viewport_window, get_active_viewport

    viewport = get_active_viewport()
    if viewport is None:
        window = create_viewport_window("SimToolReal Render", width=1280, height=720, camera_path=Sdf.Path(camera_path))
        if window is None:
            raise RuntimeError("Could not create a viewport window for capture.")
        viewport = window.viewport_api
    viewport.camera_path = Sdf.Path(camera_path)
    viewport.resolution = (1280, 720)

    app = omni.kit.app.get_app()
    for _ in range(warmup):
        await app.next_update_async()

    capture = capture_viewport_to_file(viewport, file_path=str(output_path))
    await app.next_update_async()
    await capture.wait_for_result()


def _capture_viewport(camera_path: str, output_path: Path) -> None:
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait_for(_capture_viewport_async(camera_path, output_path), timeout=20.0))


def _camera_eye_from_view(view: str, center: np.ndarray, target: np.ndarray, distance: float) -> np.ndarray:
    if view == "iso":
        offset = np.array([distance, -distance, 1.65], dtype=np.float32)
    elif view == "front":
        offset = np.array([0.0, -1.7 * distance, 1.25], dtype=np.float32)
    elif view == "back":
        offset = np.array([0.0, 1.7 * distance, 1.25], dtype=np.float32)
    elif view == "left":
        offset = np.array([-1.7 * distance, 0.0, 1.25], dtype=np.float32)
    elif view == "right":
        offset = np.array([1.7 * distance, 0.0, 1.25], dtype=np.float32)
    elif view == "top":
        offset = np.array([0.25, -0.35, max(3.4, 1.9 * distance)], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported camera view: {view}")
    return center + offset


def _set_camera_for_two_envs(env) -> tuple[np.ndarray, np.ndarray]:
    unwrapped = env.unwrapped
    camera_path = unwrapped.cfg.viewer.cam_prim_path
    if camera_path != "/OmniverseKit_Persp":
        camera_cfg = sim_utils.PinholeCameraCfg(focal_length=20.0, clipping_range=(0.01, 100.0))
        camera_cfg.func(camera_path, camera_cfg)
    origins = unwrapped.scene.env_origins[: unwrapped.num_envs].detach().cpu().numpy()
    center = origins.mean(axis=0)
    spread = float(np.linalg.norm(origins.max(axis=0) - origins.min(axis=0)))
    distance = max(2.8, spread * 1.4)
    target = center + np.array([0.0, 0.38, 0.52], dtype=np.float32)
    eye = _camera_eye_from_view(args_cli.camera_view, center, target, distance)
    unwrapped.sim.set_camera_view(eye=eye.tolist(), target=target.tolist(), camera_prim_path=camera_path)
    return eye, target


def main() -> None:
    _patch_syntheticdata_dependency_bug()
    env_device = args_cli.env_device or args_cli.device
    print(f"[RENDER] parsing task={args_cli.task} app_device={args_cli.device} env_device={env_device}", flush=True)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=env_device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.viewer.resolution = (1280, 720)
    env_cfg.viewer.cam_prim_path = args_cli.camera_path

    print("[RENDER] creating env", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    try:
        print("[RENDER] resetting env", flush=True)
        env.reset()
        eye, target = _set_camera_for_two_envs(env)
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            for _ in range(args_cli.steps):
                env.step(actions)
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        if args_cli.capture_backend == "viewport":
            _capture_viewport(env.unwrapped.cfg.viewer.cam_prim_path, args_cli.output)
            frame = np.asarray(Image.open(args_cli.output))
            frame_shape = tuple(frame.shape)
            frame_mean = float(frame.mean())
            frame_std = float(frame.std())
            capture_backend = "viewport"
        else:
            try:
                frame = _render_frame_rgb_array(env)
                Image.fromarray(frame).save(args_cli.output)
                frame_shape = tuple(frame.shape)
                frame_mean = float(frame.mean())
                frame_std = float(frame.std())
                capture_backend = "rgb_array"
            except Exception as err:
                if args_cli.capture_backend == "rgb_array":
                    raise
                print(f"[RENDER] rgb_array failed, trying viewport capture: {type(err).__name__}: {err}", flush=True)
                _capture_viewport(env.unwrapped.cfg.viewer.cam_prim_path, args_cli.output)
                frame = np.asarray(Image.open(args_cli.output))
                frame_shape = tuple(frame.shape)
                frame_mean = float(frame.mean())
                frame_std = float(frame.std())
                capture_backend = "viewport"
        print(f"[RENDER] saved={args_cli.output}", flush=True)
        print(f"[RENDER] capture_backend={capture_backend}", flush=True)
        print(f"[RENDER] camera_path={env.unwrapped.cfg.viewer.cam_prim_path}", flush=True)
        print(f"[RENDER] camera_view={args_cli.camera_view}", flush=True)
        print(f"[RENDER] camera_eye={eye.tolist()} camera_target={target.tolist()}", flush=True)
        print(f"[RENDER] frame_shape={frame_shape} frame_mean={frame_mean:.3f} frame_std={frame_std:.3f}", flush=True)
        if frame_std < 0.5:
            raise RuntimeError(f"Captured frame is flat after {capture_backend} capture.")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"[RENDER][ERROR] {type(err).__name__}: {err}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
