"""Export a short headless policy rollout video on cluster GPU nodes."""

# NOTE: torch must be imported AFTER isaacgym imports
# isort: off
from isaacgymenvs.tasks.simtoolreal.env import SimToolReal  # noqa: F401
import torch
# isort: on

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

from deployment.isaac.isaac_env import create_env
from deployment.rl_player import RlPlayer
from isaacgymenvs.utils.utils import get_repo_root_dir

N_OBS = 140
N_ACT = 29


@dataclass
class ExportVideoArgs:
    config_path: Path = Path("pretrained_policy/config.yaml")
    checkpoint_path: Path = Path("pretrained_policy/model.pth")
    object_category: str = "hammer"
    object_name: str = "claw_hammer"
    task_name: str = "swing_down"
    output_dir: Path = Path("outputs/videos")
    video_length: int = 60
    max_steps: int = 240
    headless: bool = True
    goal_z_offset: float = 0.1


def _load_traj(object_category: str, object_name: str, task_name: str) -> dict:
    trajectory_path = (
        get_repo_root_dir()
        / "dextoolbench/trajectories"
        / object_category
        / object_name
        / f"{task_name}.json"
    )
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")
    with trajectory_path.open() as f:
        return json.load(f)


def _latest_video(videos_dir: Path) -> Path:
    videos = sorted(videos_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not videos:
        raise FileNotFoundError(f"No video produced in {videos_dir}")
    return videos[-1]


def main() -> None:
    args = tyro.cli(ExportVideoArgs)

    if not args.config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config_path}")
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[video] device={device}")
    if device != "cuda":
        raise RuntimeError("CUDA is unavailable. Submit this script to a GPU debug job.")

    repo_root = get_repo_root_dir()
    args.output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else repo_root / args.output_dir
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.output_dir / "_raw"
    videos_dir.mkdir(parents=True, exist_ok=True)

    traj_data = _load_traj(args.object_category, args.object_name, args.task_name)
    traj_data["goals"] = [
        [x, y, z + args.goal_z_offset, qx, qy, qz, qw]
        for (x, y, z, qx, qy, qz, qw) in traj_data["goals"]
    ]

    print("[video] creating env")
    env = create_env(
        config_path=str(args.config_path),
        headless=args.headless,
        device=device,
        overrides={
            "task.env.resetPositionNoiseX": 0.0,
            "task.env.resetPositionNoiseY": 0.0,
            "task.env.resetPositionNoiseZ": 0.0,
            "task.env.randomizeObjectRotation": False,
            "task.env.resetDofPosRandomIntervalFingers": 0.0,
            "task.env.resetDofPosRandomIntervalArm": 0.0,
            "task.env.resetDofVelRandomInterval": 0.0,
            "task.env.tableResetZRange": 0.0,
            "task.env.objectName": args.object_name,
            "task.env.numEnvs": 1,
            "task.env.envSpacing": 0.4,
            "task.env.capture_video": True,
            "task.env.capture_video_freq": 1000000,
            "task.env.capture_video_len": args.video_length,
            "task.env.useFixedGoalStates": True,
            "task.env.fixedGoalStates": traj_data["goals"],
            "task.env.useActionDelay": False,
            "task.env.useObsDelay": False,
            "task.env.useObjectStateDelayNoise": False,
            "task.env.objectScaleNoiseMultiplierRange": [1.0, 1.0],
            "task.env.resetWhenDropped": False,
            "task.env.armMovingAverage": 0.1,
            "task.env.evalSuccessTolerance": 0.01,
            "task.env.successSteps": 1,
            "task.env.fixedSizeKeypointReward": True,
            "task.env.useFixedInitObjectPose": True,
            "task.env.objectStartPose": traj_data["start_pose"],
            "task.env.startArmHigher": True,
            "task.env.forceScale": 0.0,
            "task.env.torqueScale": 0.0,
            "task.env.linVelImpulseScale": 0.0,
            "task.env.angVelImpulseScale": 0.0,
            "task.env.forceOnlyWhenLifted": True,
            "task.env.torqueOnlyWhenLifted": True,
            "task.env.linVelImpulseOnlyWhenLifted": True,
            "task.env.angVelImpulseOnlyWhenLifted": True,
            "task.env.forceProbRange": [0.0001, 0.0001],
            "task.env.torqueProbRange": [0.0001, 0.0001],
            "task.env.linVelImpulseProbRange": [0.0001, 0.0001],
            "task.env.angVelImpulseProbRange": [0.0001, 0.0001],
        },
    )

    print("[video] loading checkpoint and policy")
    checkpoint = torch.load(args.checkpoint_path)
    env.set_env_state(checkpoint[0]["env_state"])
    policy = RlPlayer(
        num_observations=N_OBS,
        num_actions=N_ACT,
        config_path=str(args.config_path),
        checkpoint_path=str(args.checkpoint_path),
        device=device,
        num_envs=env.num_envs,
    )

    obs_dict, _, _, _ = env.step(torch.zeros((env.num_envs, N_ACT), device=device))
    obs = obs_dict["obs"]

    print("[video] starting capture")
    env.video_frames = []
    env._capture_video(video_capture_in_progress=False)

    for step_idx in range(args.max_steps):
        action = policy.get_normalized_action(obs, deterministic_actions=True)
        obs_dict, reward, done, _ = env.step(action)
        obs = obs_dict["obs"]
        print(
            "[video] "
            f"step={step_idx + 1}/{args.max_steps} "
            f"reward={float(reward[0].item()):.4f} "
            f"done={bool(done[0].item())}"
        )
        if env.video_frames is None:
            break

    latest_video = _latest_video(videos_dir)
    output_path = args.output_dir / (
        f"{args.object_name}_{args.task_name}_{latest_video.name}"
    )
    shutil.copy2(latest_video, output_path)
    print(f"[video] saved_copy={output_path}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
