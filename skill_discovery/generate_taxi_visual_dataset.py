"""Generate the complete reachable-state Taxi-v4 RGB orientation dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np

from skill_discovery.audit_taxi_environment import (
    TAXI_STAGES,
    reachable_state_ids,
    taxi_semantic_stage,
)
from skill_discovery.generate_point_cup_dataset import _write_png


TAXI_SPLITS = (
    "train_reference",
    "joint_audit",
    "destination_only_audit",
    "orientation_only_audit",
)


@dataclass(frozen=True)
class TaxiVisualDatasetConfig:
    env_id: str = "Taxi-v4"
    train_destinations: tuple[int, ...] = (0, 2)
    audit_destinations: tuple[int, ...] = (1, 3)
    train_orientations: tuple[int, ...] = (0, 2)
    audit_orientations: tuple[int, ...] = (1, 3)

    def __post_init__(self) -> None:
        if sorted((*self.train_destinations, *self.audit_destinations)) != list(
            range(4)
        ):
            raise ValueError("destination split must partition four destinations")
        if sorted((*self.train_orientations, *self.audit_orientations)) != list(
            range(4)
        ):
            raise ValueError("orientation split must partition four orientations")


def taxi_visual_split(
    destination: int,
    orientation: int,
    config: TaxiVisualDatasetConfig,
) -> int:
    destination_train = destination in config.train_destinations
    orientation_train = orientation in config.train_orientations
    if destination_train and orientation_train:
        return 0
    if not destination_train and not orientation_train:
        return 1
    if not destination_train and orientation_train:
        return 2
    return 3


def _frame_hash(frame: np.ndarray) -> str:
    return hashlib.blake2b(frame.tobytes(), digest_size=16).hexdigest()


def split_counts_without_render(
    base: object,
    states: tuple[int, ...],
    config: TaxiVisualDatasetConfig,
) -> np.ndarray:
    counts = np.zeros((len(TAXI_SPLITS), len(TAXI_STAGES)), dtype=np.int64)
    for state in states:
        destination = int(base.decode(state)[3])
        stage = taxi_semantic_stage(base, state)
        for orientation in range(4):
            split = taxi_visual_split(destination, orientation, config)
            counts[split, stage] += 1
    return counts


def _write_contact_sheet(
    path: Path,
    frames: np.ndarray,
    stages: np.ndarray,
    splits: np.ndarray,
    states: np.ndarray,
    destinations: np.ndarray,
    orientations: np.ndarray,
) -> list[dict[str, object]]:
    gap = 6
    marker = 10
    sample = frames[0][::2, ::2]
    height, width = sample.shape[:2]
    sheet = np.full(
        (3 * height + 2 * gap, marker + 4 * width + 3 * gap, 3),
        255,
        dtype=np.uint8,
    )
    colors = ((104, 117, 125), (40, 117, 164), (43, 137, 95))
    manifest = []
    for stage, stage_name in enumerate(TAXI_STAGES):
        y = stage * (height + gap)
        sheet[y : y + height, :marker] = colors[stage]
        for split, split_name in enumerate(TAXI_SPLITS):
            candidates = np.flatnonzero((stages == stage) & (splits == split))
            if not len(candidates):
                raise RuntimeError("Taxi contact sheet split is missing a stage")
            index = int(candidates[0])
            x = marker + split * (width + gap)
            sheet[y : y + height, x : x + width] = frames[index][::2, ::2]
            manifest.append(
                {
                    "stage": stage_name,
                    "split": split_name,
                    "index": index,
                    "state": int(states[index]),
                    "destination": int(destinations[index]),
                    "orientation": int(orientations[index]),
                }
            )
    _write_png(path, sheet)
    return manifest


def generate_taxi_visual_dataset(
    config: TaxiVisualDatasetConfig,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    env = gym.make(config.env_id, render_mode="rgb_array")
    try:
        base = env.unwrapped
        _, reachable, terminal = reachable_state_ids(base)
        env.reset(seed=0)
        sample = env.render()
        row_count = len(reachable) * 4
        frames = np.empty((row_count, *sample.shape), dtype=np.uint8)
        states = np.empty(row_count, dtype=np.int16)
        decoded = np.empty((row_count, 4), dtype=np.int8)
        stages = np.empty(row_count, dtype=np.int8)
        orientations = np.empty(row_count, dtype=np.int8)
        splits = np.empty(row_count, dtype=np.int8)
        hashes = []
        all_orientation_unique = True
        cursor = 0
        for state in reachable:
            state_hashes = []
            row, col, passenger, destination = base.decode(state)
            stage = taxi_semantic_stage(base, state)
            for orientation in range(4):
                base.s = int(state)
                base.lastaction = orientation
                frame = base.render()
                digest = _frame_hash(frame)
                frames[cursor] = frame
                states[cursor] = state
                decoded[cursor] = (row, col, passenger, destination)
                stages[cursor] = stage
                orientations[cursor] = orientation
                splits[cursor] = taxi_visual_split(
                    destination,
                    orientation,
                    config,
                )
                hashes.append(digest)
                state_hashes.append(digest)
                cursor += 1
            all_orientation_unique &= len(set(state_hashes)) == 4
    finally:
        env.close()
    dataset_path = output_dir / "taxi_full_visual_dataset.npz"
    np.savez_compressed(
        dataset_path,
        frames=frames,
        states=states,
        decoded_states=decoded,
        stages=stages,
        destinations=decoded[:, 3],
        orientations=orientations,
        splits=splits,
        frame_hashes=np.asarray(hashes),
    )
    image_path = output_dir / "taxi_full_visual_contact_sheet.png"
    manifest = _write_contact_sheet(
        image_path,
        frames,
        stages,
        splits,
        states,
        decoded[:, 3],
        orientations,
    )
    manifest_path = output_dir / "taxi_full_visual_contact_sheet.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    split_stage_counts = np.zeros((4, 3), dtype=np.int64)
    for split in range(4):
        split_stage_counts[split] = np.bincount(
            stages[splits == split],
            minlength=3,
        )
    split_destinations = {
        TAXI_SPLITS[split]: sorted(set(decoded[splits == split, 3].tolist()))
        for split in range(4)
    }
    split_orientations = {
        TAXI_SPLITS[split]: sorted(set(orientations[splits == split].tolist()))
        for split in range(4)
    }
    expected_counts = np.asarray([[300, 100, 4]] * 4)
    data_gate = bool(
        len(reachable) == 404
        and len(terminal) == 4
        and cursor == 1_616
        and frames.shape == (1_616, 350, 550, 3)
        and all_orientation_unique
        and np.array_equal(split_stage_counts, expected_counts)
        and split_destinations["train_reference"] == [0, 2]
        and split_destinations["joint_audit"] == [1, 3]
        and split_orientations["train_reference"] == [0, 2]
        and split_orientations["joint_audit"] == [1, 3]
    )
    output = {
        "config": asdict(config),
        "stage_names": TAXI_STAGES,
        "split_names": TAXI_SPLITS,
        "reachable_state_count": len(reachable),
        "terminal_state_count": len(terminal),
        "frame_count": cursor,
        "frame_shape": list(frames.shape[1:]),
        "all_state_orientation_hashes_unique": bool(all_orientation_unique),
        "split_stage_counts": {
            TAXI_SPLITS[index]: split_stage_counts[index].tolist()
            for index in range(4)
        },
        "split_destinations": split_destinations,
        "split_orientations": split_orientations,
        "dataset": str(dataset_path.resolve()),
        "manual_audit_image": str(image_path.resolve()),
        "manual_audit_manifest": str(manifest_path.resolve()),
        "data_gate_passed": data_gate,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["metrics"] = str(metrics_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = TaxiVisualDatasetConfig()
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/taxi"
    ) / f"taxi_full_visual_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = generate_taxi_visual_dataset(config, output_dir)
    print(
        json.dumps(
            {
                "metrics": output["metrics"],
                "dataset": output["dataset"],
                "manual_audit_image": output["manual_audit_image"],
                "frame_count": output["frame_count"],
                "split_stage_counts": output["split_stage_counts"],
                "data_gate_passed": output["data_gate_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
