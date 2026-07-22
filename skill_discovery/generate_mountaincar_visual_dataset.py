"""Generate balanced MountainCarContinuous RGB position-region splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np

from skill_discovery.generate_point_cup_dataset import _write_png


MOUNTAINCAR_POSITION_STAGES = (
    "left_slope",
    "valley",
    "right_slope",
    "goal_region",
)
POSITION_INTERVALS = (
    (-1.15, -0.75),
    (-0.75, 0.0),
    (0.0, 0.45),
    (0.45, 0.58),
)
SPLIT_NAMES = ("reference", "audit")


@dataclass(frozen=True)
class MountainCarVisualDatasetConfig:
    samples_per_stage: int = 256
    reference_seed: int = 700_007
    audit_seed: int = 1_700_007
    max_attempts_per_stage: int = 100_000

    def __post_init__(self) -> None:
        if self.samples_per_stage <= 0:
            raise ValueError("samples per stage must be positive")
        if self.reference_seed == self.audit_seed:
            raise ValueError("reference and audit seeds must differ")
        if self.max_attempts_per_stage < self.samples_per_stage:
            raise ValueError("attempt budget must cover each stage quota")


def frame_hash(frame: np.ndarray) -> str:
    return hashlib.blake2b(frame.tobytes(), digest_size=16).hexdigest()


def position_stage(position: float) -> int:
    for stage, (low, high) in enumerate(POSITION_INTERVALS):
        if stage == 0 and low <= position <= high:
            return stage
        if low < position < high or (stage >= 2 and low <= position < high):
            return stage
    if POSITION_INTERVALS[-1][0] <= position <= POSITION_INTERVALS[-1][1]:
        return len(POSITION_INTERVALS) - 1
    raise ValueError("position is outside the frozen visual stage intervals")


def collect_split(
    config: MountainCarVisualDatasetConfig,
    split: int,
    forbidden_hashes: set[str],
) -> tuple[list[dict[str, object]], set[str], list[int]]:
    rng_seed = config.reference_seed if split == 0 else config.audit_seed
    rng = np.random.default_rng(rng_seed)
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    samples = []
    split_hashes: set[str] = set()
    attempts_by_stage = []
    try:
        env.reset(seed=rng_seed)
        for stage, (low, high) in enumerate(POSITION_INTERVALS):
            stage_samples = []
            attempts = 0
            while len(stage_samples) < config.samples_per_stage:
                if attempts >= config.max_attempts_per_stage:
                    raise RuntimeError(
                        f"stage {stage} collected {len(stage_samples)}/"
                        f"{config.samples_per_stage} unique frames after {attempts} attempts"
                    )
                attempts += 1
                position = float(rng.uniform(np.nextafter(low, high), high))
                velocity = float(rng.uniform(-0.07, 0.07))
                env.unwrapped.state = np.asarray(
                    [position, velocity], dtype=np.float32
                )
                frame = env.render().copy()
                digest = frame_hash(frame)
                if digest in split_hashes or digest in forbidden_hashes:
                    continue
                observed_position = float(env.unwrapped.state[0])
                if position_stage(observed_position) != stage:
                    raise RuntimeError("sample escaped its frozen position interval")
                sample = {
                    "frame": frame,
                    "hash": digest,
                    "stage": stage,
                    "split": split,
                    "position": observed_position,
                    "velocity": float(env.unwrapped.state[1]),
                    "sample_index": len(stage_samples),
                }
                stage_samples.append(sample)
                split_hashes.add(digest)
            samples.extend(stage_samples)
            attempts_by_stage.append(attempts)
    finally:
        env.close()
    return samples, split_hashes, attempts_by_stage


def _write_contact_sheet(
    path: Path,
    samples: list[dict[str, object]],
) -> list[dict[str, object]]:
    frame_height, frame_width = 200, 300
    gap = 5
    marker = 10
    examples_per_split = 4
    columns = len(SPLIT_NAMES) * examples_per_split
    sheet = np.full(
        (
            len(MOUNTAINCAR_POSITION_STAGES) * frame_height
            + (len(MOUNTAINCAR_POSITION_STAGES) - 1) * gap,
            marker + columns * frame_width + (columns - 1) * gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    colors = ((34, 114, 157), (214, 93, 74), (64, 145, 108), (146, 92, 156))
    manifest = []
    for stage, name in enumerate(MOUNTAINCAR_POSITION_STAGES):
        y = stage * (frame_height + gap)
        sheet[y : y + frame_height, :marker] = colors[stage]
        for split, split_name in enumerate(SPLIT_NAMES):
            candidates = [
                sample
                for sample in samples
                if sample["stage"] == stage and sample["split"] == split
            ]
            indices = np.linspace(
                0, len(candidates) - 1, examples_per_split, dtype=np.int64
            )
            for offset, index in enumerate(indices):
                sample = candidates[int(index)]
                column = split * examples_per_split + offset
                x = marker + column * (frame_width + gap)
                sheet[y : y + frame_height, x : x + frame_width] = sample["frame"][
                    ::2, ::2
                ]
                manifest.append(
                    {
                        "stage": name,
                        "split": split_name,
                        "position": sample["position"],
                        "velocity": sample["velocity"],
                        "frame_hash": sample["hash"],
                    }
                )
    _write_png(path, sheet)
    return manifest


def generate_dataset(
    config: MountainCarVisualDatasetConfig,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    reference, reference_hashes, reference_attempts = collect_split(
        config, 0, set()
    )
    audit, audit_hashes, audit_attempts = collect_split(
        config, 1, reference_hashes
    )
    samples = [*reference, *audit]
    frames = np.stack([sample["frame"] for sample in samples]).astype(np.uint8)
    stages = np.asarray([sample["stage"] for sample in samples], dtype=np.int8)
    splits = np.asarray([sample["split"] for sample in samples], dtype=np.int8)
    positions = np.asarray(
        [sample["position"] for sample in samples], dtype=np.float32
    )
    velocities = np.asarray(
        [sample["velocity"] for sample in samples], dtype=np.float32
    )
    hashes = np.asarray([sample["hash"] for sample in samples])
    counts = np.zeros((2, 4), dtype=np.int64)
    np.add.at(counts, (splits, stages), 1)
    expected_count = config.samples_per_stage * len(POSITION_INTERVALS) * 2
    positions_unique = len(np.unique(positions)) == len(positions)
    data_gate = bool(
        len(frames) == expected_count
        and frames.shape[1:] == (400, 600, 3)
        and frames.dtype == np.uint8
        and np.all(counts == config.samples_per_stage)
        and len(reference_hashes) == len(reference)
        and len(audit_hashes) == len(audit)
        and not reference_hashes.intersection(audit_hashes)
        and positions_unique
    )
    dataset_path = output_dir / "mountaincar_visual_dataset.npz"
    np.savez_compressed(
        dataset_path,
        frames=frames,
        stages=stages,
        splits=splits,
        positions=positions,
        velocities=velocities,
        frame_hashes=hashes,
    )
    contact_sheet_path = output_dir / "mountaincar_visual_contact_sheet.png"
    contact_manifest = _write_contact_sheet(contact_sheet_path, samples)
    output = {
        "config": asdict(config),
        "sdl_video_driver": os.environ.get("SDL_VIDEODRIVER"),
        "dataset": str(dataset_path.resolve()),
        "frame_count": len(frames),
        "frame_shape": list(frames.shape[1:]),
        "counts_by_split_stage": counts.tolist(),
        "attempts_by_split_stage": [reference_attempts, audit_attempts],
        "reference_unique_hashes": len(reference_hashes),
        "audit_unique_hashes": len(audit_hashes),
        "cross_split_hash_overlap": len(reference_hashes.intersection(audit_hashes)),
        "positions_unique": positions_unique,
        "contact_sheet": str(contact_sheet_path.resolve()),
        "contact_manifest": contact_manifest,
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
    config = MountainCarVisualDatasetConfig()
    run_id = f"visual_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/mountaincar_continuous"
    ) / run_id
    output = generate_dataset(config, output_dir)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
