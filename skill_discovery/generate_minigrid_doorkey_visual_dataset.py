"""Generate split-safe balanced DoorKey stage frames for visual metrics."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np

from skill_discovery.generate_point_cup_dataset import _write_png
from skill_discovery.minigrid_doorkey import DOORKEY_STAGES, solve_doorkey_episode


@dataclass(frozen=True)
class DoorKeyVisualDatasetConfig:
    env_id: str = "MiniGrid-DoorKey-5x5-v0"
    train_generation_groups: tuple[int, ...] = (7, 17, 27)
    audit_generation_groups: tuple[int, ...] = (37, 47)
    trajectories_per_group: int = 128
    seed_stride: int = 100_000

    def __post_init__(self) -> None:
        if not self.train_generation_groups or not self.audit_generation_groups:
            raise ValueError("train and audit groups must both be nonempty")
        if set(self.train_generation_groups) & set(self.audit_generation_groups):
            raise ValueError("train and audit generation groups must be disjoint")
        if self.trajectories_per_group <= 0 or self.seed_stride <= 0:
            raise ValueError("trajectory count and seed stride must be positive")


def _write_contact_sheet(
    path: Path,
    frames: np.ndarray,
    split: np.ndarray,
    generation_groups: np.ndarray,
    env_seeds: np.ndarray,
) -> list[dict[str, int | str]]:
    selected = []
    for split_id in (0, 1):
        candidates = np.flatnonzero(split == split_id)
        selected.extend(candidates[:3].tolist())
    frame_height, frame_width = frames.shape[2:4]
    gap = 6
    marker = 9
    sheet = np.full(
        (
            len(selected) * frame_height + (len(selected) - 1) * gap,
            marker + 4 * frame_width + 3 * gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    colors = ((40, 117, 164), (209, 112, 49))
    manifest = []
    for row, index in enumerate(selected):
        y = row * (frame_height + gap)
        sheet[y : y + frame_height, :marker] = colors[int(split[index])]
        for stage in range(4):
            x = marker + stage * (frame_width + gap)
            sheet[y : y + frame_height, x : x + frame_width] = frames[index, stage]
        manifest.append(
            {
                "row": row,
                "trajectory_index": int(index),
                "split": "train" if split[index] == 0 else "audit",
                "generation_group": int(generation_groups[index]),
                "env_seed": int(env_seeds[index]),
            }
        )
    _write_png(path, sheet)
    return manifest


def generate_dataset(
    config: DoorKeyVisualDatasetConfig,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    env = gym.make(config.env_id, render_mode="rgb_array")
    trajectories = []
    try:
        for split_id, groups in enumerate(
            (config.train_generation_groups, config.audit_generation_groups)
        ):
            for group in groups:
                for episode in range(config.trajectories_per_group):
                    env_seed = group * config.seed_stride + episode
                    result = solve_doorkey_episode(env, env_seed)
                    if (
                        len(result.frames) != len(DOORKEY_STAGES)
                        or not result.terminated
                        or result.truncated
                        or result.reward <= 0
                    ):
                        raise RuntimeError("scripted DoorKey trajectory failed audit")
                    trajectories.append(
                        {
                            "split": split_id,
                            "generation_group": group,
                            "env_seed": env_seed,
                            "frames": result.frames,
                            "actions": result.actions,
                            "action_count": len(result.actions),
                            "native_reward": result.reward,
                        }
                    )
    finally:
        env.close()

    frames = np.asarray([row["frames"] for row in trajectories], dtype=np.uint8)
    split = np.asarray([row["split"] for row in trajectories], dtype=np.int8)
    generation_groups = np.asarray(
        [row["generation_group"] for row in trajectories],
        dtype=np.int32,
    )
    env_seeds = np.asarray(
        [row["env_seed"] for row in trajectories],
        dtype=np.int64,
    )
    action_counts = np.asarray(
        [row["action_count"] for row in trajectories],
        dtype=np.int16,
    )
    maximum_actions = int(action_counts.max())
    actions = np.full((len(trajectories), maximum_actions), -1, dtype=np.int8)
    for index, row in enumerate(trajectories):
        values = np.asarray(row["actions"], dtype=np.int8)
        actions[index, : len(values)] = values
    stage_labels = np.tile(
        np.arange(len(DOORKEY_STAGES), dtype=np.int8),
        (len(trajectories), 1),
    )
    native_rewards = np.asarray(
        [row["native_reward"] for row in trajectories],
        dtype=np.float32,
    )
    dataset_path = output_dir / "doorkey_visual_dataset.npz"
    np.savez_compressed(
        dataset_path,
        frames=frames,
        stage_labels=stage_labels,
        split=split,
        generation_groups=generation_groups,
        env_seeds=env_seeds,
        actions=actions,
        action_counts=action_counts,
        native_rewards=native_rewards,
    )
    image_path = output_dir / "manual_stage_audit.png"
    contact_manifest = _write_contact_sheet(
        image_path,
        frames,
        split,
        generation_groups,
        env_seeds,
    )
    (output_dir / "manual_stage_audit.json").write_text(
        json.dumps(contact_manifest, indent=2),
        encoding="utf-8",
    )
    train_groups = sorted(set(generation_groups[split == 0].tolist()))
    audit_groups = sorted(set(generation_groups[split == 1].tolist()))
    output = {
        "config": asdict(config),
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("minigrid", "gymnasium", "pygame-ce")
        },
        "trajectory_count": len(trajectories),
        "frame_count": int(frames.shape[0] * frames.shape[1]),
        "frame_shape": list(frames.shape[2:]),
        "stage_names": DOORKEY_STAGES,
        "train_trajectory_count": int(np.count_nonzero(split == 0)),
        "audit_trajectory_count": int(np.count_nonzero(split == 1)),
        "train_generation_groups": train_groups,
        "audit_generation_groups": audit_groups,
        "split_groups_disjoint": not bool(set(train_groups) & set(audit_groups)),
        "stage_frame_counts": {
            stage: len(trajectories) for stage in DOORKEY_STAGES
        },
        "action_count_min": int(action_counts.min()),
        "action_count_max": maximum_actions,
        "all_native_success": bool(np.all(native_rewards > 0)),
        "dataset": str(dataset_path.resolve()),
        "manual_audit_image": str(image_path.resolve()),
        "manual_audit_manifest": contact_manifest,
        "data_gate_passed": bool(
            not set(train_groups) & set(audit_groups)
            and np.all(native_rewards > 0)
            and frames.shape[1] == len(DOORKEY_STAGES)
        ),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["metrics"] = str(metrics_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories-per-group", type=int, default=128)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = DoorKeyVisualDatasetConfig(
        trajectories_per_group=args.trajectories_per_group,
    )
    run_id = f"doorkey5_visual_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/minigrid_doorkey_visual"
    ) / run_id
    output = generate_dataset(config, output_dir)
    print(
        json.dumps(
            {
                "metrics": output["metrics"],
                "dataset": output["dataset"],
                "manual_audit_image": output["manual_audit_image"],
                "trajectory_count": output["trajectory_count"],
                "frame_count": output["frame_count"],
                "split_groups_disjoint": output["split_groups_disjoint"],
                "all_native_success": output["all_native_success"],
                "data_gate_passed": output["data_gate_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
