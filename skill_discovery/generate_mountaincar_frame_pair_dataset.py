"""Generate balanced MountainCar frame-pair relation shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np

from skill_discovery.audit_mountaincar_continuous_environment import (
    expected_transition,
)
from skill_discovery.audit_mountaincar_frame_pair_capacity import (
    PAIR_CLASSES,
    PROPOSAL_RANGES,
    transition_class,
)
from skill_discovery.generate_point_cup_dataset import _write_png


SPLIT_NAMES = ("reference", "audit")


@dataclass(frozen=True)
class FramePairDatasetConfig:
    samples_per_class: int = 256
    reference_seed: int = 3_100_007
    audit_seed: int = 4_100_007
    max_attempt_multiplier: int = 100

    def __post_init__(self) -> None:
        if self.samples_per_class <= 0:
            raise ValueError("samples per class must be positive")
        if self.reference_seed == self.audit_seed:
            raise ValueError("reference and audit seeds must differ")
        if self.max_attempt_multiplier <= 0:
            raise ValueError("attempt multiplier must be positive")


def frame_hash(frame: np.ndarray) -> str:
    return hashlib.blake2b(frame.tobytes(), digest_size=16).hexdigest()


def pair_hash(before: np.ndarray, after: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(before.tobytes())
    digest.update(after.tobytes())
    return digest.hexdigest()


def _collect_split(
    env: object,
    config: FramePairDatasetConfig,
    class_index: int,
    split: int,
    forbidden_pair_hashes: set[str],
) -> tuple[dict[str, np.ndarray], set[str], int]:
    seed = config.reference_seed if split == 0 else config.audit_seed
    rng = np.random.default_rng(seed + class_index * 100_000)
    (position_low, position_high), (velocity_low, velocity_high) = PROPOSAL_RANGES[
        class_index
    ]
    frames = []
    states = []
    actions = []
    next_states = []
    pair_hashes = []
    before_hashes = []
    after_hashes = []
    seen_pairs: set[str] = set()
    attempts = 0
    max_attempts = config.samples_per_class * config.max_attempt_multiplier
    while len(frames) < config.samples_per_class:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"{PAIR_CLASSES[class_index]} {SPLIT_NAMES[split]} collected "
                f"{len(frames)}/{config.samples_per_class} after {attempts} attempts"
            )
        attempts += 1
        state = np.asarray(
            [
                rng.uniform(position_low, position_high),
                rng.uniform(velocity_low, velocity_high),
            ],
            dtype=np.float32,
        )
        action = np.asarray([rng.uniform(-1.0, 1.0)], dtype=np.float32)
        next_state, _, terminated = expected_transition(state, action)
        if not transition_class(class_index, next_state, terminated):
            continue
        env.unwrapped.state = state.copy()
        before = env.render().copy()
        env.unwrapped.state = next_state.copy()
        after = env.render().copy()
        digest = pair_hash(before, after)
        if digest in seen_pairs or digest in forbidden_pair_hashes:
            continue
        frames.append(np.stack((before, after)))
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        pair_hashes.append(digest)
        before_hashes.append(frame_hash(before))
        after_hashes.append(frame_hash(after))
        seen_pairs.add(digest)
    return (
        {
            "frames": np.stack(frames).astype(np.uint8),
            "states": np.stack(states).astype(np.float32),
            "actions": np.stack(actions).astype(np.float32),
            "next_states": np.stack(next_states).astype(np.float32),
            "pair_hashes": np.asarray(pair_hashes),
            "before_hashes": np.asarray(before_hashes),
            "after_hashes": np.asarray(after_hashes),
            "classes": np.full(len(frames), class_index, dtype=np.int8),
            "splits": np.full(len(frames), split, dtype=np.int8),
        },
        seen_pairs,
        attempts,
    )


def generate_class_shard(
    config: FramePairDatasetConfig,
    class_index: int,
    output_dir: Path,
) -> dict[str, object]:
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    env.reset(seed=config.reference_seed + class_index)
    try:
        reference, reference_pairs, reference_attempts = _collect_split(
            env, config, class_index, 0, set()
        )
        audit, audit_pairs, audit_attempts = _collect_split(
            env, config, class_index, 1, reference_pairs
        )
    finally:
        env.close()
    arrays = {
        name: np.concatenate((reference[name], audit[name]), axis=0)
        for name in reference
    }
    shard_path = output_dir / f"class_{class_index}_{PAIR_CLASSES[class_index]}.npz"
    np.savez_compressed(shard_path, **arrays)
    examples = []
    for split in range(2):
        offset = split * config.samples_per_class
        for sample_offset in range(2):
            index = offset + sample_offset
            examples.append(
                {
                    "split": split,
                    "state": arrays["states"][index].tolist(),
                    "next_state": arrays["next_states"][index].tolist(),
                    "pair_hash": str(arrays["pair_hashes"][index]),
                    "frames": arrays["frames"][index],
                }
            )
    return {
        "class_index": class_index,
        "class_name": PAIR_CLASSES[class_index],
        "shard": str(shard_path.resolve()),
        "reference_attempts": reference_attempts,
        "audit_attempts": audit_attempts,
        "reference_pair_hashes": sorted(reference_pairs),
        "audit_pair_hashes": sorted(audit_pairs),
        "reference_before_unique": len(set(reference["before_hashes"].tolist())),
        "reference_after_unique": len(set(reference["after_hashes"].tolist())),
        "audit_before_unique": len(set(audit["before_hashes"].tolist())),
        "audit_after_unique": len(set(audit["after_hashes"].tolist())),
        "examples": examples,
    }


def _write_contact_sheet(path: Path, class_rows: list[dict[str, object]]) -> None:
    frame_height, frame_width = 200, 300
    marker = 10
    gap = 5
    row_gap = 6
    columns = 8
    sheet = np.full(
        (
            len(PAIR_CLASSES) * frame_height + (len(PAIR_CLASSES) - 1) * row_gap,
            marker + columns * frame_width + (columns - 1) * gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    colors = np.asarray(
        ([34, 114, 157], [214, 93, 74], [64, 145, 108], [146, 92, 156]),
        dtype=np.uint8,
    )
    for row_index, row in enumerate(class_rows):
        y = row_index * (frame_height + row_gap)
        sheet[y : y + frame_height, :marker] = colors[row_index]
        for example_index, example in enumerate(row["examples"]):
            for pair_offset, frame in enumerate(example["frames"]):
                column = example_index * 2 + pair_offset
                x = marker + column * (frame_width + gap)
                sheet[y : y + frame_height, x : x + frame_width] = frame[::2, ::2]
    _write_png(path, sheet)


def generate_dataset(
    config: FramePairDatasetConfig,
    output_dir: Path,
    *,
    workers: int = 4,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        class_rows = list(
            pool.map(
                generate_class_shard,
                [config] * len(PAIR_CLASSES),
                range(len(PAIR_CLASSES)),
                [output_dir] * len(PAIR_CLASSES),
            )
        )
    class_rows.sort(key=lambda row: int(row["class_index"]))
    reference_hashes = {
        digest for row in class_rows for digest in row["reference_pair_hashes"]
    }
    audit_hashes = {
        digest for row in class_rows for digest in row["audit_pair_hashes"]
    }
    expected_per_split = config.samples_per_class * len(PAIR_CLASSES)
    all_pair_counts_exact = bool(
        len(reference_hashes) == expected_per_split
        and len(audit_hashes) == expected_per_split
    )
    cross_split_overlap = reference_hashes.intersection(audit_hashes)
    predicates_valid = True
    finite = True
    frame_shape_valid = True
    counts = np.zeros((2, len(PAIR_CLASSES)), dtype=np.int64)
    for row in class_rows:
        with np.load(row["shard"]) as shard:
            frames = shard["frames"]
            states = shard["states"]
            actions = shard["actions"]
            next_states = shard["next_states"]
            classes = shard["classes"]
            splits = shard["splits"]
            np.add.at(counts, (splits, classes), 1)
            frame_shape_valid &= frames.shape[1:] == (2, 400, 600, 3)
            frame_shape_valid &= frames.dtype == np.uint8
            finite &= bool(
                np.isfinite(states).all()
                and np.isfinite(actions).all()
                and np.isfinite(next_states).all()
            )
            predicates_valid &= all(
                transition_class(int(class_index), next_state, int(class_index) == 3)
                for class_index, next_state in zip(classes, next_states, strict=True)
            )
    contact_sheet_path = output_dir / "frame_pair_dataset_contact_sheet.png"
    _write_contact_sheet(contact_sheet_path, class_rows)
    data_gate = bool(
        all_pair_counts_exact
        and not cross_split_overlap
        and np.all(counts == config.samples_per_class)
        and predicates_valid
        and finite
        and frame_shape_valid
    )
    public_rows = []
    for row in class_rows:
        public_rows.append(
            {key: value for key, value in row.items() if key != "examples"}
        )
    output = {
        "config": asdict(config),
        "workers": workers,
        "sdl_video_driver": os.environ.get("SDL_VIDEODRIVER"),
        "class_shards": public_rows,
        "counts_by_split_class": counts.tolist(),
        "reference_unique_pair_hashes": len(reference_hashes),
        "audit_unique_pair_hashes": len(audit_hashes),
        "cross_split_pair_hash_overlap": len(cross_split_overlap),
        "predicates_valid": predicates_valid,
        "all_values_finite": finite,
        "frame_shape_valid": frame_shape_valid,
        "contact_sheet": str(contact_sheet_path.resolve()),
        "contact_sheet_columns": [
            "reference_0_before",
            "reference_0_after",
            "reference_1_before",
            "reference_1_after",
            "audit_0_before",
            "audit_0_after",
            "audit_1_before",
            "audit_1_after",
        ],
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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = FramePairDatasetConfig()
    run_id = f"frame_pair_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/mountaincar_continuous"
    ) / run_id
    output = generate_dataset(config, output_dir, workers=args.workers)
    print(
        json.dumps(
            {
                "output": str((output_dir / "metrics.json").resolve()),
                "counts": output["counts_by_split_class"],
                "cross_split_overlap": output["cross_split_pair_hash_overlap"],
                "data_gate_passed": output["data_gate_passed"],
                "manual_contact_sheet_gate": output["manual_contact_sheet_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
