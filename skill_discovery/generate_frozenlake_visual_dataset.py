"""Generate a balanced cached RGB trajectory dataset from slippery FrozenLake."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from skill_discovery.frozenlake import FROZENLAKE_OUTCOMES, FrozenLakeConfig
from skill_discovery.frozenlake_slippery import (
    finite_horizon_outcome_policy,
    rollout_outcome_policy,
)
from skill_discovery.generate_point_cup_dataset import _write_png


@dataclass(frozen=True)
class FrozenLakeVisualDatasetConfig:
    seeds: tuple[int, ...] = (7, 17, 27, 37, 47)
    trajectories_per_outcome_per_seed: int = 128
    image_size: int = 64
    sample_trajectories_per_outcome: int = 10
    lake: FrozenLakeConfig = FrozenLakeConfig(is_slippery=True)

    def __post_init__(self) -> None:
        if len(self.seeds) < 2:
            raise ValueError("at least two generation seeds are required")
        if self.trajectories_per_outcome_per_seed <= 0:
            raise ValueError("trajectory count must be positive")
        if self.image_size <= 0:
            raise ValueError("image size must be positive")


def _resize_nearest(image: np.ndarray, size: int) -> np.ndarray:
    y = np.linspace(0, image.shape[0] - 1, size).astype(np.int64)
    x = np.linspace(0, image.shape[1] - 1, size).astype(np.int64)
    return image[y][:, x]


def _write_sample_sheet(
    path: Path,
    frames: np.ndarray,
    outcomes: np.ndarray,
    config: FrozenLakeVisualDatasetConfig,
) -> list[int]:
    rng = np.random.default_rng(202_607_22)
    samples_per_outcome = min(
        config.sample_trajectories_per_outcome,
        config.trajectories_per_outcome_per_seed * len(config.seeds),
    )
    columns = 5
    triptych_size = 72
    gap = 4
    marker_width = 9
    rows_per_outcome = int(np.ceil(samples_per_outcome / columns))
    row_count = rows_per_outcome * len(FROZENLAKE_OUTCOMES)
    triptych_width = 3 * triptych_size
    sheet = np.full(
        (
            row_count * triptych_size + (row_count - 1) * gap,
            marker_width + columns * triptych_width + (columns - 1) * gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    colors = ((48, 116, 173), (198, 72, 58), (42, 137, 94))
    selected = []
    for outcome in range(len(FROZENLAKE_OUTCOMES)):
        candidates = np.flatnonzero(outcomes == outcome)
        chosen = rng.choice(candidates, size=samples_per_outcome, replace=False)
        selected.extend(int(index) for index in chosen)
        for local_index, trajectory_index in enumerate(chosen):
            row = outcome * rows_per_outcome + local_index // columns
            column = local_index % columns
            y = row * (triptych_size + gap)
            x = marker_width + column * (triptych_width + gap)
            sheet[y : y + triptych_size, :marker_width] = colors[outcome]
            for frame_index in range(3):
                frame = _resize_nearest(
                    frames[int(trajectory_index), frame_index],
                    triptych_size,
                )
                frame_x = x + frame_index * triptych_size
                sheet[
                    y : y + triptych_size,
                    frame_x : frame_x + triptych_size,
                ] = frame
    _write_png(path, sheet)
    return selected


def generate_dataset(
    config: FrozenLakeVisualDatasetConfig,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    policies = [
        finite_horizon_outcome_policy(config.lake, outcome)
        for outcome in range(len(FROZENLAKE_OUTCOMES))
    ]
    frames = []
    outcomes = []
    generation_seeds = []
    rollout_seeds = []
    states = []
    actions = []
    lengths = []
    native_rewards = []
    attempts_by_group = {}
    manifest = []
    trajectory_index = 0

    for outcome, policy in enumerate(policies):
        for generation_seed in config.seeds:
            accepted = 0
            attempt = 0
            while accepted < config.trajectories_per_outcome_per_seed:
                rollout_seed = (
                    generation_seed * 10_000_000
                    + outcome * 1_000_000
                    + attempt
                )
                rollout = rollout_outcome_policy(
                    config.lake,
                    policy,
                    seed=rollout_seed,
                    render=True,
                )
                attempt += 1
                if int(rollout["outcome"]) != outcome:
                    continue
                rollout_frames = rollout["frames"]
                key_indices = (
                    0,
                    len(rollout_frames) // 2,
                    len(rollout_frames) - 1,
                )
                frames.append(
                    np.stack(
                        [
                            _resize_nearest(rollout_frames[index], config.image_size)
                            for index in key_indices
                        ]
                    )
                )
                outcomes.append(outcome)
                generation_seeds.append(generation_seed)
                rollout_seeds.append(rollout_seed)
                state_sequence = np.full(
                    config.lake.max_episode_steps + 1,
                    -1,
                    dtype=np.int16,
                )
                action_sequence = np.full(
                    config.lake.max_episode_steps,
                    -1,
                    dtype=np.int8,
                )
                rollout_states = np.asarray(rollout["states"], dtype=np.int16)
                rollout_actions = np.asarray(rollout["actions"], dtype=np.int8)
                state_sequence[: len(rollout_states)] = rollout_states
                action_sequence[: len(rollout_actions)] = rollout_actions
                states.append(state_sequence)
                actions.append(action_sequence)
                lengths.append(len(rollout_actions))
                native_rewards.append(float(rollout["native_reward"]))
                manifest.append(
                    {
                        "index": trajectory_index,
                        "outcome": FROZENLAKE_OUTCOMES[outcome],
                        "generation_seed": generation_seed,
                        "rollout_seed": rollout_seed,
                        "attempt_in_group": attempt - 1,
                        "length": len(rollout_actions),
                        "terminal_state": int(rollout["terminal_state"]),
                        "native_reward": float(rollout["native_reward"]),
                    }
                )
                trajectory_index += 1
                accepted += 1
            attempts_by_group[
                f"{FROZENLAKE_OUTCOMES[outcome]}_seed{generation_seed}"
            ] = attempt

    frame_array = np.stack(frames).astype(np.uint8)
    outcome_array = np.asarray(outcomes, dtype=np.int8)
    seed_array = np.asarray(generation_seeds, dtype=np.int32)
    train_seed_count = min(
        len(config.seeds) - 1,
        max(1, int(np.ceil(len(config.seeds) * 0.6))),
    )
    train_seeds = set(config.seeds[:train_seed_count])
    split = np.asarray(
        [0 if int(seed) in train_seeds else 1 for seed in seed_array],
        dtype=np.int8,
    )
    dataset_path = output_dir / "trajectories.npz"
    np.savez_compressed(
        dataset_path,
        frames=frame_array,
        outcomes=outcome_array,
        generation_seeds=seed_array,
        rollout_seeds=np.asarray(rollout_seeds, dtype=np.int64),
        split=split,
        states=np.stack(states),
        actions=np.stack(actions),
        lengths=np.asarray(lengths, dtype=np.int16),
        native_rewards=np.asarray(native_rewards, dtype=np.float32),
    )
    image_path = output_dir / "visual_sample_30.png"
    selected = _write_sample_sheet(
        image_path,
        frame_array,
        outcome_array,
        config,
    )
    (output_dir / "trajectory_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    summary = {
        "config": asdict(config),
        "trajectory_count": len(outcome_array),
        "outcome_counts": {
            name: int(np.count_nonzero(outcome_array == index))
            for index, name in enumerate(FROZENLAKE_OUTCOMES)
        },
        "generation_seed_counts": {
            str(seed): int(np.count_nonzero(seed_array == seed))
            for seed in config.seeds
        },
        "train_seeds": sorted(train_seeds),
        "audit_seeds": sorted(set(config.seeds) - train_seeds),
        "train_count": int(np.count_nonzero(split == 0)),
        "audit_count": int(np.count_nonzero(split == 1)),
        "attempts_by_group": attempts_by_group,
        "sample_indices": selected,
        "dataset": str(dataset_path.resolve()),
        "image": str(image_path.resolve()),
        "passed": bool(
            all(
                np.count_nonzero(outcome_array == outcome)
                == config.trajectories_per_outcome_per_seed * len(config.seeds)
                for outcome in range(len(FROZENLAKE_OUTCOMES))
            )
            and len(np.unique(seed_array[split == 0])) >= 1
            and len(np.unique(seed_array[split == 1])) >= 1
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 17, 27, 37, 47))
    parser.add_argument("--trajectories-per-outcome-per-seed", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--map-name", default="4x4")
    parser.add_argument("--max-episode-steps", type=int, default=32)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = FrozenLakeVisualDatasetConfig(
        seeds=tuple(args.seeds),
        trajectories_per_outcome_per_seed=args.trajectories_per_outcome_per_seed,
        image_size=args.image_size,
        lake=FrozenLakeConfig(
            map_name=args.map_name,
            is_slippery=True,
            max_episode_steps=args.max_episode_steps,
        ),
    )
    run_id = (
        f"frozenlake_{config.lake.map_name}_visual_dataset_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/frozenlake_visual"
    ) / run_id
    summary = generate_dataset(config, output_dir)
    print(
        json.dumps(
            {
                "summary": str((output_dir / "summary.json").resolve()),
                "dataset": summary["dataset"],
                "image": summary["image"],
                "trajectory_count": summary["trajectory_count"],
                "outcome_counts": summary["outcome_counts"],
                "train_seeds": summary["train_seeds"],
                "audit_seeds": summary["audit_seeds"],
                "passed": summary["passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
