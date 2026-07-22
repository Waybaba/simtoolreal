"""Audit the official MountainCarContinuous-v0 dynamics and RGB renderer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np

from skill_discovery.generate_point_cup_dataset import _write_png


@dataclass(frozen=True)
class MountainCarEnvironmentAuditConfig:
    env_id: str = "MountainCarContinuous-v0"
    seeds: tuple[int, ...] = (7, 17, 29, 41, 53)
    random_horizon: int = 256
    scripted_horizon: int = 999

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("audit seeds must be non-empty and unique")
        if self.random_horizon != 256 or self.scripted_horizon != 999:
            raise ValueError("MountainCar audit horizons are frozen")


def frame_hash(frame: np.ndarray) -> str:
    return hashlib.blake2b(frame.tobytes(), digest_size=16).hexdigest()


def expected_transition(
    state: np.ndarray,
    action: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    position = float(state[0])
    velocity = float(state[1])
    force = float(np.clip(action[0], -1.0, 1.0))
    velocity += force * 0.0015 - 0.0025 * np.cos(3.0 * position)
    velocity = float(np.clip(velocity, -0.07, 0.07))
    position += velocity
    position = float(np.clip(position, -1.2, 0.6))
    if position == -1.2 and velocity < 0:
        velocity = 0.0
    terminated = bool(position >= 0.45 and velocity >= 0.0)
    reward = 100.0 * terminated - 0.1 * float(action[0]) ** 2
    return np.asarray([position, velocity], dtype=np.float32), reward, terminated


def ordered_stage(previous_stage: int, position: float, terminated: bool) -> int:
    if previous_stage == 0 and position <= -0.75:
        return 1
    if previous_stage == 1 and position >= 0.0:
        return 2
    if previous_stage == 2 and terminated:
        return 3
    return previous_stage


def _frame_record(frame: np.ndarray) -> dict[str, object]:
    return {
        "hash": frame_hash(frame),
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "pixel_std": float(frame.std()),
    }


def run_random_tape(
    config: MountainCarEnvironmentAuditConfig,
    seed: int,
    actions: np.ndarray,
) -> tuple[dict[str, object], list[np.ndarray]]:
    env = gym.make(config.env_id, render_mode="rgb_array")
    try:
        observation, _ = env.reset(seed=seed)
        reset_observation = observation.copy()
        frame = env.render().copy()
        frames = [frame]
        frame_records = [_frame_record(frame)]
        rows = []
        for step_index, action in enumerate(actions, start=1):
            previous = observation.copy()
            observation, reward, terminated, truncated, _ = env.step(action)
            expected_state, expected_reward, expected_terminated = expected_transition(
                previous, action
            )
            frame = env.render().copy()
            frames.append(frame)
            frame_records.append(_frame_record(frame))
            rows.append(
                {
                    "step": step_index,
                    "action": action.tolist(),
                    "observation": observation.tolist(),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "expected_observation": expected_state.tolist(),
                    "dynamics_max_error": float(
                        np.max(np.abs(observation - expected_state))
                    ),
                    "reward_error": abs(float(reward) - expected_reward),
                    "termination_matches": bool(terminated == expected_terminated),
                    "frame": frame_records[-1],
                }
            )
        return (
            {
                "seed": seed,
                "reset_observation": reset_observation.tolist(),
                "reset_frame": frame_records[0],
                "frame_records": frame_records,
                "rows": rows,
                "api": {
                    "observation_shape": list(env.observation_space.shape),
                    "observation_dtype": str(env.observation_space.dtype),
                    "action_shape": list(env.action_space.shape),
                    "action_dtype": str(env.action_space.dtype),
                    "action_low": env.action_space.low.tolist(),
                    "action_high": env.action_space.high.tolist(),
                },
            },
            frames,
        )
    finally:
        env.close()


def compare_random_tapes(
    primary: dict[str, object],
    replica: dict[str, object],
) -> dict[str, bool]:
    primary_rows = primary["rows"]
    replica_rows = replica["rows"]
    reset_equal = bool(
        np.array_equal(primary["reset_observation"], replica["reset_observation"])
        and primary["reset_frame"]["hash"] == replica["reset_frame"]["hash"]
    )
    states_equal = all(
        np.array_equal(left["observation"], right["observation"])
        for left, right in zip(primary_rows, replica_rows, strict=True)
    )
    rewards_flags_equal = all(
        left["reward"] == right["reward"]
        and left["terminated"] == right["terminated"]
        and left["truncated"] == right["truncated"]
        for left, right in zip(primary_rows, replica_rows, strict=True)
    )
    frame_hashes_equal = [row["hash"] for row in primary["frame_records"]] == [
        row["hash"] for row in replica["frame_records"]
    ]
    return {
        "reset_exact_equal": reset_equal,
        "states_exact_equal": states_equal,
        "rewards_and_flags_exact_equal": rewards_flags_equal,
        "frame_hashes_exact_equal": frame_hashes_equal,
        "passed": bool(
            reset_equal and states_equal and rewards_flags_equal and frame_hashes_equal
        ),
    }


def run_scripted_reachability(
    config: MountainCarEnvironmentAuditConfig,
    seed: int,
) -> tuple[dict[str, object], list[np.ndarray]]:
    env = gym.make(config.env_id, render_mode="rgb_array")
    try:
        observation, _ = env.reset(seed=seed)
        stage = 0
        stage_sequence = [0]
        stage_steps = {"reset": 0}
        stage_frames = [env.render().copy()]
        max_dynamics_error = 0.0
        max_reward_error = 0.0
        all_termination_matches = True
        terminated = truncated = False
        action_counts = {"left": 0, "right": 0}
        for step_index in range(1, config.scripted_horizon + 1):
            action_value = -1.0 if observation[1] <= 0.0 else 1.0
            action = np.asarray([action_value], dtype=np.float32)
            action_counts["left" if action_value < 0 else "right"] += 1
            previous = observation.copy()
            observation, reward, terminated, truncated, _ = env.step(action)
            expected_state, expected_reward, expected_terminated = expected_transition(
                previous, action
            )
            max_dynamics_error = max(
                max_dynamics_error,
                float(np.max(np.abs(observation - expected_state))),
            )
            max_reward_error = max(
                max_reward_error, abs(float(reward) - expected_reward)
            )
            all_termination_matches &= bool(terminated == expected_terminated)
            next_stage = ordered_stage(stage, float(observation[0]), bool(terminated))
            if next_stage != stage:
                stage = next_stage
                stage_sequence.append(stage)
                stage_name = ("left_momentum", "right_climb", "goal")[stage - 1]
                stage_steps[stage_name] = step_index
                stage_frames.append(env.render().copy())
            if terminated or truncated:
                break
        return (
            {
                "seed": seed,
                "steps": step_index,
                "final_observation": observation.tolist(),
                "native_terminated": bool(terminated),
                "truncated": bool(truncated),
                "stage_sequence": stage_sequence,
                "stage_steps": stage_steps,
                "action_counts": action_counts,
                "max_dynamics_error": max_dynamics_error,
                "max_reward_error": max_reward_error,
                "all_termination_matches": all_termination_matches,
                "passed": bool(
                    terminated
                    and not truncated
                    and stage_sequence == [0, 1, 2, 3]
                    and max_dynamics_error <= 1e-7
                    and max_reward_error == 0.0
                    and all_termination_matches
                    and len(stage_frames) == 4
                ),
            },
            stage_frames,
        )
    finally:
        env.close()


def _write_contact_sheet(path: Path, frames_by_seed: list[list[np.ndarray]]) -> None:
    frame_height, frame_width = frames_by_seed[0][0].shape[:2]
    marker_width = 10
    gap = 4
    row_gap = 6
    width = marker_width + 4 * frame_width + 3 * gap
    height = len(frames_by_seed) * frame_height + (len(frames_by_seed) - 1) * row_gap
    sheet = np.full((height, width, 3), 255, dtype=np.uint8)
    row_colors = np.asarray(
        ([34, 114, 157], [214, 93, 74], [64, 145, 108], [146, 92, 156], [194, 147, 45]),
        dtype=np.uint8,
    )
    for row_index, frames in enumerate(frames_by_seed):
        y = row_index * (frame_height + row_gap)
        sheet[y : y + frame_height, :marker_width] = row_colors[row_index]
        for column, frame in enumerate(frames):
            x = marker_width + column * (frame_width + gap)
            sheet[y : y + frame_height, x : x + frame_width] = frame
    _write_png(path, sheet)


def run_audit(
    config: MountainCarEnvironmentAuditConfig,
    output_dir: Path,
) -> dict[str, object]:
    import pygame

    output_dir.mkdir(parents=True, exist_ok=False)
    random_runs = []
    comparisons = []
    scripted_runs = []
    stage_frames_by_seed = []
    for seed in config.seeds:
        actions = np.random.default_rng(seed + 600_000).uniform(
            -1.0, 1.0, size=(config.random_horizon, 1)
        ).astype(np.float32)
        primary, _ = run_random_tape(config, seed, actions)
        replica, _ = run_random_tape(config, seed, actions)
        random_runs.append(primary)
        comparisons.append(compare_random_tapes(primary, replica))
        scripted, stage_frames = run_scripted_reachability(config, seed)
        scripted_runs.append(scripted)
        stage_frames_by_seed.append(stage_frames)

    all_rows = [row for run in random_runs for row in run["rows"]]
    all_frames = [row for run in random_runs for row in run["frame_records"]]
    api_dynamics_gate = bool(
        all(
            run["api"]["observation_shape"] == [2]
            and run["api"]["action_shape"] == [1]
            and run["api"]["action_low"] == [-1.0]
            and run["api"]["action_high"] == [1.0]
            and -0.6 <= run["reset_observation"][0] <= -0.4
            and run["reset_observation"][1] == 0.0
            for run in random_runs
        )
        and max(row["dynamics_max_error"] for row in all_rows) <= 1e-7
        and max(row["reward_error"] for row in all_rows) == 0.0
        and all(row["termination_matches"] for row in all_rows)
    )
    reproducibility_gate = bool(
        all(row["passed"] for row in comparisons)
        and len({run["reset_observation"][0] for run in random_runs}) >= 4
    )
    rgb_numeric_gate = bool(
        all(
            row["shape"] == [400, 600, 3]
            and row["dtype"] == "uint8"
            and row["pixel_std"] > 5.0
            for row in all_frames
        )
        and all(
            len({row["hash"] for row in run["frame_records"]}) >= 2
            for run in random_runs
        )
    )
    reachability_gate = all(run["passed"] for run in scripted_runs)
    contact_sheet_path = output_dir / "mountaincar_stage_contact_sheet.png"
    _write_contact_sheet(contact_sheet_path, stage_frames_by_seed)
    gates = {
        "api_dynamics_gate_passed": api_dynamics_gate,
        "reproducibility_gate_passed": reproducibility_gate,
        "rgb_numeric_gate_passed": rgb_numeric_gate,
        "scripted_reachability_gate_passed": reachability_gate,
    }
    output = {
        "config": asdict(config),
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "gymnasium": gym.__version__,
            "pygame": pygame.version.ver,
            "sdl_video_driver": os.environ.get("SDL_VIDEODRIVER"),
        },
        "random_runs": random_runs,
        "replica_comparisons": comparisons,
        "scripted_runs": scripted_runs,
        "contact_sheet": str(contact_sheet_path.resolve()),
        "contact_sheet_rows": [f"seed_{seed}" for seed in config.seeds],
        "contact_sheet_columns": ["reset", "left_momentum", "right_climb", "goal"],
        "max_random_dynamics_error": max(
            row["dynamics_max_error"] for row in all_rows
        ),
        "max_random_reward_error": max(row["reward_error"] for row in all_rows),
        "minimum_random_frame_pixel_std": min(
            row["pixel_std"] for row in all_frames
        ),
        "gates": gates,
        "numeric_gate_passed": all(gates.values()),
        "manual_contact_sheet_gate": "pending",
        "passed": False,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = MountainCarEnvironmentAuditConfig()
    run_id = f"environment_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/mountaincar_continuous"
    ) / run_id
    output = run_audit(config, output_dir)
    print(
        json.dumps(
            {
                "output": str((output_dir / "audit.json").resolve()),
                "versions": output["versions"],
                "gates": output["gates"],
                "scripted_steps": [row["steps"] for row in output["scripted_runs"]],
                "numeric_gate_passed": output["numeric_gate_passed"],
                "manual_contact_sheet_gate": output["manual_contact_sheet_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
