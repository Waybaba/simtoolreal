"""Audit the official Gymnasium Pusher-v5 API and headless RGB renderer."""

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
class PusherV5EnvironmentAuditConfig:
    env_id: str = "Pusher-v5"
    seeds: tuple[int, ...] = (7, 17, 29, 41, 53)
    horizon: int = 100
    width: int = 256
    height: int = 256

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("audit seeds must be non-empty and unique")
        if self.horizon != 100:
            raise ValueError("Pusher-v5 audit horizon is frozen at 100")
        if (self.width, self.height) != (256, 256):
            raise ValueError("Pusher-v5 audit resolution is frozen at 256x256")


def frame_hash(frame: np.ndarray) -> str:
    return hashlib.blake2b(frame.tobytes(), digest_size=16).hexdigest()


def reward_terms(
    observation: np.ndarray,
    action: np.ndarray,
) -> dict[str, float]:
    fingertip = observation[14:17]
    object_position = observation[17:20]
    goal = observation[20:23]
    return {
        "reward_dist": -float(np.linalg.norm(object_position - goal)),
        "reward_ctrl": -0.1 * float(np.square(action).sum()),
        "reward_near": -0.5 * float(np.linalg.norm(object_position - fingertip)),
    }


def _body_positions(base: object) -> dict[str, np.ndarray]:
    return {
        "fingertip": np.asarray(base.get_body_com("tips_arm"), dtype=np.float64),
        "object": np.asarray(base.get_body_com("object"), dtype=np.float64),
        "goal": np.asarray(base.get_body_com("goal"), dtype=np.float64),
    }


def _render_record(frame: np.ndarray) -> dict[str, object]:
    return {
        "hash": frame_hash(frame),
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "pixel_std": float(frame.std()),
    }


def run_rollout(
    config: PusherV5EnvironmentAuditConfig,
    seed: int,
    actions: np.ndarray,
) -> tuple[dict[str, object], list[np.ndarray]]:
    env = gym.make(
        config.env_id,
        render_mode="rgb_array",
        width=config.width,
        height=config.height,
    )
    try:
        observation, _ = env.reset(seed=seed)
        reset_observation = observation.copy()
        base = env.unwrapped
        reset_bodies = _body_positions(base)
        reset_tail = np.concatenate(
            [reset_bodies["fingertip"], reset_bodies["object"], reset_bodies["goal"]]
        )
        reset_frame = env.render().copy()
        render_records = [_render_record(reset_frame)]
        frames = [reset_frame]
        rows = []
        for step_index, action in enumerate(actions, start=1):
            observation, reward, terminated, truncated, info = env.step(action)
            bodies = _body_positions(base)
            frame = env.render().copy()
            render_records.append(_render_record(frame))
            frames.append(frame)

            recomputed = reward_terms(observation, action)
            info_terms = {
                name: float(info[name])
                for name in ("reward_dist", "reward_ctrl", "reward_near")
            }
            reward_sum = sum(info_terms.values())
            tail = np.concatenate(
                [bodies["fingertip"], bodies["object"], bodies["goal"]]
            )
            rows.append(
                {
                    "step": step_index,
                    "action": action.tolist(),
                    "observation": observation.tolist(),
                    "reward": float(reward),
                    "reward_terms": info_terms,
                    "recomputed_reward_terms": recomputed,
                    "reward_sum_error": abs(float(reward) - reward_sum),
                    "reward_term_max_error": max(
                        abs(info_terms[name] - recomputed[name]) for name in info_terms
                    ),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "fingertip": bodies["fingertip"].tolist(),
                    "object": bodies["object"].tolist(),
                    "goal": bodies["goal"].tolist(),
                    "observation_body_max_error": float(
                        np.max(np.abs(observation[14:23] - tail))
                    ),
                    "frame": render_records[-1],
                }
            )

        action_space = env.action_space
        output = {
            "seed": seed,
            "reset_observation": reset_observation.tolist(),
            "reset_body_positions": {
                name: value.tolist() for name, value in reset_bodies.items()
            },
            "reset_observation_body_max_error": float(
                np.max(np.abs(reset_observation[14:23] - reset_tail))
            ),
            "reset_object_goal_planar_distance": float(
                np.linalg.norm(reset_bodies["object"][:2] - reset_bodies["goal"][:2])
            ),
            "reset_render": render_records[0],
            "render_records": render_records,
            "rows": rows,
            "api": {
                "observation_shape": list(env.observation_space.shape),
                "action_shape": list(action_space.shape),
                "action_low": action_space.low.tolist(),
                "action_high": action_space.high.tolist(),
                "action_dtype": str(action_space.dtype),
            },
        }
        return output, frames
    finally:
        env.close()


def compare_rollouts(
    primary: dict[str, object],
    replica: dict[str, object],
) -> dict[str, object]:
    primary_rows = primary["rows"]
    replica_rows = replica["rows"]
    observation_equal = all(
        np.array_equal(left["observation"], right["observation"])
        for left, right in zip(primary_rows, replica_rows, strict=True)
    )
    scalar_equal = all(
        left["reward"] == right["reward"]
        and left["reward_terms"] == right["reward_terms"]
        and left["terminated"] == right["terminated"]
        and left["truncated"] == right["truncated"]
        for left, right in zip(primary_rows, replica_rows, strict=True)
    )
    frame_hashes_equal = [row["hash"] for row in primary["render_records"]] == [
        row["hash"] for row in replica["render_records"]
    ]
    reset_equal = bool(
        np.array_equal(primary["reset_observation"], replica["reset_observation"])
        and primary["reset_render"]["hash"] == replica["reset_render"]["hash"]
    )
    return {
        "reset_exact_equal": reset_equal,
        "observations_exact_equal": observation_equal,
        "rewards_and_flags_exact_equal": scalar_equal,
        "frame_hashes_exact_equal": frame_hashes_equal,
        "passed": bool(
            reset_equal and observation_equal and scalar_equal and frame_hashes_equal
        ),
    }


def pixel_difference_summary(
    primary_frames: list[np.ndarray],
    replica_frames: list[np.ndarray],
) -> dict[str, int | float]:
    if len(primary_frames) != len(replica_frames):
        raise ValueError("replica frame sequences must have the same length")
    mismatched_frames = 0
    different_channel_values = 0
    absolute_difference_sum = 0
    maximum_absolute_difference = 0
    value_count = 0
    for primary, replica in zip(primary_frames, replica_frames, strict=True):
        if primary.shape != replica.shape:
            raise ValueError("replica frames must have the same shapes")
        difference = np.abs(primary.astype(np.int16) - replica.astype(np.int16))
        mismatched_frames += int(np.any(difference))
        different_channel_values += int(np.count_nonzero(difference))
        absolute_difference_sum += int(difference.sum())
        maximum_absolute_difference = max(
            maximum_absolute_difference, int(difference.max())
        )
        value_count += int(difference.size)
    return {
        "mismatched_frame_count": mismatched_frames,
        "total_frame_count": len(primary_frames),
        "different_channel_value_count": different_channel_values,
        "maximum_absolute_pixel_difference": maximum_absolute_difference,
        "mean_absolute_pixel_difference": absolute_difference_sum / value_count,
    }


def _write_contact_sheet(
    path: Path,
    seeds: tuple[int, ...],
    frames_by_seed: list[list[np.ndarray]],
) -> None:
    marker_width = 10
    gap = 4
    row_gap = 6
    frame_size = frames_by_seed[0][0].shape[0]
    width = marker_width + 3 * frame_size + 2 * gap
    height = len(seeds) * frame_size + (len(seeds) - 1) * row_gap
    sheet = np.full((height, width, 3), 255, dtype=np.uint8)
    row_colors = np.asarray(
        ([34, 114, 157], [214, 93, 74], [64, 145, 108], [146, 92, 156], [194, 147, 45]),
        dtype=np.uint8,
    )
    for row_index, frames in enumerate(frames_by_seed):
        y = row_index * (frame_size + row_gap)
        sheet[y : y + frame_size, :marker_width] = row_colors[row_index]
        for column, frame in enumerate(frames):
            x = marker_width + column * (frame_size + gap)
            sheet[y : y + frame_size, x : x + frame_size] = frame
    _write_png(path, sheet)


def run_audit(
    config: PusherV5EnvironmentAuditConfig,
    output_dir: Path,
) -> dict[str, object]:
    import mujoco

    output_dir.mkdir(parents=True, exist_ok=False)
    rollout_rows = []
    comparisons = []
    frames_by_seed = []
    for seed in config.seeds:
        rng = np.random.default_rng(seed + 500_000)
        actions = rng.uniform(-2.0, 2.0, size=(config.horizon, 7)).astype(np.float32)
        primary, primary_frames = run_rollout(config, seed, actions)
        replica, replica_frames = run_rollout(config, seed, actions)
        rollout_rows.append(primary)
        comparison = compare_rollouts(primary, replica)
        comparison["pixel_difference_diagnostic"] = pixel_difference_summary(
            primary_frames, replica_frames
        )
        comparisons.append(comparison)
        frames_by_seed.append(
            [primary_frames[0], primary_frames[config.horizon // 2], primary_frames[-1]]
        )

    all_steps = [row for rollout in rollout_rows for row in rollout["rows"]]
    all_render_records = [
        row for rollout in rollout_rows for row in rollout["render_records"]
    ]
    api_gate = bool(
        all(
            rollout["api"]["observation_shape"] == [23]
            and rollout["api"]["action_shape"] == [7]
            and rollout["api"]["action_low"] == [-2.0] * 7
            and rollout["api"]["action_high"] == [2.0] * 7
            for rollout in rollout_rows
        )
        and all(
            not row["terminated"]
            and row["truncated"] == (row["step"] == config.horizon)
            and np.isfinite(row["observation"]).all()
            and np.isfinite(row["reward"])
            for row in all_steps
        )
    )
    reward_gate = bool(
        max(row["reward_sum_error"] for row in all_steps) <= 1e-10
        and max(row["reward_term_max_error"] for row in all_steps) <= 1e-10
    )
    reproducibility_gate = bool(
        all(row["passed"] for row in comparisons)
        and len(
            {
                tuple(np.round(rollout["reset_body_positions"]["object"][:2], 12))
                for rollout in rollout_rows
            }
        )
        >= 4
    )
    rgb_gate = bool(
        all(
            row["shape"] == [256, 256, 3]
            and row["dtype"] == "uint8"
            and row["pixel_std"] > 5.0
            for row in all_render_records
        )
        and all(
            len({row["hash"] for row in rollout["render_records"]}) >= 2
            for rollout in rollout_rows
        )
    )
    semantic_interface_gate = bool(
        all(
            rollout["reset_object_goal_planar_distance"] > 0.17
            and rollout["reset_observation_body_max_error"] <= 1e-12
            for rollout in rollout_rows
        )
        and max(row["observation_body_max_error"] for row in all_steps) <= 1e-12
    )
    contact_sheet_path = output_dir / "pusher_v5_reset_mid_final.png"
    _write_contact_sheet(contact_sheet_path, config.seeds, frames_by_seed)
    gates = {
        "api_gate_passed": api_gate,
        "reward_gate_passed": reward_gate,
        "reproducibility_gate_passed": reproducibility_gate,
        "rgb_numeric_gate_passed": rgb_gate,
        "semantic_interface_gate_passed": semantic_interface_gate,
    }
    output = {
        "config": asdict(config),
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "gymnasium": gym.__version__,
            "mujoco": mujoco.__version__,
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        },
        "contact_sheet": str(contact_sheet_path.resolve()),
        "contact_sheet_rows": [f"seed_{seed}" for seed in config.seeds],
        "contact_sheet_columns": ["reset", "step_50", "step_100"],
        "rollouts": rollout_rows,
        "replica_comparisons": comparisons,
        "max_reward_sum_error": max(row["reward_sum_error"] for row in all_steps),
        "max_reward_term_error": max(
            row["reward_term_max_error"] for row in all_steps
        ),
        "max_observation_body_error": max(
            row["observation_body_max_error"] for row in all_steps
        ),
        "minimum_frame_pixel_std": min(
            row["pixel_std"] for row in all_render_records
        ),
        "gates": gates,
        "numeric_gate_passed": all(gates.values()),
        "manual_contact_sheet_gate": "pending",
        "passed": False,
    }
    output_path = output_dir / "audit.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = PusherV5EnvironmentAuditConfig()
    run_id = f"environment_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/pusher_v5_environment"
    ) / run_id
    output = run_audit(config, output_dir)
    print(
        json.dumps(
            {
                "output": str((output_dir / "audit.json").resolve()),
                "versions": output["versions"],
                "gates": output["gates"],
                "numeric_gate_passed": output["numeric_gate_passed"],
                "manual_contact_sheet_gate": output["manual_contact_sheet_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
