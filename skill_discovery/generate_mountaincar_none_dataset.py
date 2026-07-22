"""Generate an explicit MountainCar none-class frame-pair shard."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np

from skill_discovery.audit_mountaincar_continuous_environment import (
    expected_transition,
)
from skill_discovery.audit_mountaincar_rollout_rejection import oracle_relation
from skill_discovery.generate_mountaincar_frame_pair_dataset import pair_hash
from skill_discovery.generate_point_cup_dataset import _write_png


@dataclass(frozen=True)
class NoneDatasetConfig:
    samples_per_split: int = 256
    reference_seed: int = 8_100_007
    audit_seed: int = 9_100_007
    max_attempt_multiplier: int = 100

    def __post_init__(self) -> None:
        if self.samples_per_split <= 0 or self.max_attempt_multiplier <= 0:
            raise ValueError("none dataset budgets must be positive")
        if self.reference_seed == self.audit_seed:
            raise ValueError("none reference and audit seeds must differ")


def existing_relation_hashes(dataset_dir: Path) -> set[str]:
    hashes: set[str] = set()
    shard_paths = sorted(dataset_dir.glob("class_*.npz"))
    if len(shard_paths) != 4:
        raise ValueError("expected four relation shards")
    for shard_path in shard_paths:
        with np.load(shard_path) as shard:
            hashes.update(str(value) for value in shard["pair_hashes"])
    return hashes


def _collect_split(
    env: object,
    config: NoneDatasetConfig,
    split: int,
    forbidden_hashes: set[str],
) -> tuple[dict[str, np.ndarray], set[str], int]:
    seed = config.reference_seed if split == 0 else config.audit_seed
    rng = np.random.default_rng(seed)
    frames = []
    states = []
    actions = []
    next_states = []
    terminateds = []
    hashes = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = config.samples_per_split * config.max_attempt_multiplier
    while len(frames) < config.samples_per_split:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"none split {split} collected {len(frames)}/"
                f"{config.samples_per_split} after {attempts} attempts"
            )
        attempts += 1
        state = np.asarray(
            [rng.uniform(-1.15, 0.58), rng.uniform(-0.07, 0.07)],
            dtype=np.float32,
        )
        action = np.asarray([rng.uniform(-1.0, 1.0)], dtype=np.float32)
        next_state, _, terminated = expected_transition(state, action)
        if oracle_relation(next_state, terminated) != -1:
            continue
        env.unwrapped.state = state.copy()
        before = env.render().copy()
        env.unwrapped.state = next_state.copy()
        after = env.render().copy()
        digest = pair_hash(before, after)
        if digest in seen or digest in forbidden_hashes:
            continue
        frames.append(np.stack((before, after)))
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        terminateds.append(terminated)
        hashes.append(digest)
        seen.add(digest)
    return (
        {
            "frames": np.stack(frames).astype(np.uint8),
            "states": np.stack(states).astype(np.float32),
            "actions": np.stack(actions).astype(np.float32),
            "next_states": np.stack(next_states).astype(np.float32),
            "terminateds": np.asarray(terminateds, dtype=np.bool_),
            "pair_hashes": np.asarray(hashes),
            "classes": np.full(len(frames), -1, dtype=np.int8),
            "splits": np.full(len(frames), split, dtype=np.int8),
        },
        seen,
        attempts,
    )


def _write_contact_sheet(path: Path, frames: np.ndarray, splits: np.ndarray) -> None:
    frame_height, frame_width = 200, 300
    marker = 10
    gap = 5
    row_gap = 6
    examples_per_split = 2
    columns = examples_per_split * 2
    sheet = np.full(
        (
            2 * frame_height + row_gap,
            marker + columns * frame_width + (columns - 1) * gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    colors = ((100, 110, 118), (188, 138, 45))
    for split in range(2):
        y = split * (frame_height + row_gap)
        sheet[y : y + frame_height, :marker] = colors[split]
        indices = np.flatnonzero(splits == split)[:examples_per_split]
        for example_offset, index in enumerate(indices):
            for frame_offset, frame in enumerate(frames[index]):
                column = example_offset * 2 + frame_offset
                x = marker + column * (frame_width + gap)
                sheet[y : y + frame_height, x : x + frame_width] = frame[::2, ::2]
    _write_png(path, sheet)


def generate_dataset(
    config: NoneDatasetConfig,
    relation_dataset_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    relation_hashes = existing_relation_hashes(relation_dataset_dir)
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    env.reset(seed=config.reference_seed)
    try:
        reference, reference_hashes, reference_attempts = _collect_split(
            env, config, 0, relation_hashes
        )
        audit, audit_hashes, audit_attempts = _collect_split(
            env, config, 1, relation_hashes | reference_hashes
        )
    finally:
        env.close()
    arrays = {
        name: np.concatenate((reference[name], audit[name]), axis=0)
        for name in reference
    }
    predicates_valid = all(
        oracle_relation(next_state, bool(terminated)) == -1
        for next_state, terminated in zip(
            arrays["next_states"], arrays["terminateds"], strict=True
        )
    )
    cross_split_overlap = reference_hashes.intersection(audit_hashes)
    relation_overlap = (reference_hashes | audit_hashes).intersection(relation_hashes)
    counts = np.bincount(arrays["splits"], minlength=2)
    data_gate = bool(
        np.all(counts == config.samples_per_split)
        and not cross_split_overlap
        and not relation_overlap
        and predicates_valid
        and arrays["frames"].shape == (config.samples_per_split * 2, 2, 400, 600, 3)
        and arrays["frames"].dtype == np.uint8
        and np.isfinite(arrays["states"]).all()
        and np.isfinite(arrays["actions"]).all()
        and np.isfinite(arrays["next_states"]).all()
    )
    shard_path = output_dir / "class_none.npz"
    np.savez_compressed(shard_path, **arrays)
    contact_sheet_path = output_dir / "none_class_contact_sheet.png"
    _write_contact_sheet(contact_sheet_path, arrays["frames"], arrays["splits"])
    output = {
        "config": asdict(config),
        "relation_dataset_dir": str(relation_dataset_dir.resolve()),
        "sdl_video_driver": os.environ.get("SDL_VIDEODRIVER"),
        "shard": str(shard_path.resolve()),
        "counts_by_split": counts.tolist(),
        "reference_attempts": reference_attempts,
        "audit_attempts": audit_attempts,
        "reference_unique_pair_hashes": len(reference_hashes),
        "audit_unique_pair_hashes": len(audit_hashes),
        "cross_split_pair_hash_overlap": len(cross_split_overlap),
        "relation_pair_hash_overlap": len(relation_overlap),
        "predicates_valid": predicates_valid,
        "contact_sheet": str(contact_sheet_path.resolve()),
        "data_gate_passed": data_gate,
        "manual_contact_sheet_gate": "pending",
        "passed": False,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["metrics"] = str(metrics_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("relation_dataset_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = NoneDatasetConfig()
    run_id = f"none_class_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/mountaincar_continuous"
    ) / run_id
    output = generate_dataset(config, args.relation_dataset_dir, output_dir)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
