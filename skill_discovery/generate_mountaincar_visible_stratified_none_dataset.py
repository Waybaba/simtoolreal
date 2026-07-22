"""Generate the frozen visible-stratified MountainCar none shard."""

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
from skill_discovery.audit_mountaincar_frame_pair_capacity import pair_hash
from skill_discovery.audit_mountaincar_stratified_none_capacity import (
    VISIBLE_STRATA,
    sample_candidate,
    stratum_matches,
)
from skill_discovery.generate_mountaincar_none_dataset import (
    existing_relation_hashes,
)
from skill_discovery.generate_point_cup_dataset import _write_png


@dataclass(frozen=True)
class VisibleStratifiedNoneDatasetConfig:
    samples_per_stratum: int = 32
    reference_seed: int = 8_300_007
    audit_seed: int = 9_300_007
    max_attempt_multiplier: int = 100

    def __post_init__(self) -> None:
        if self.samples_per_stratum <= 0 or self.max_attempt_multiplier <= 0:
            raise ValueError("dataset budgets must be positive")
        if self.reference_seed == self.audit_seed:
            raise ValueError("reference and audit seeds must differ")


def _collect_stratum(
    env: object,
    config: VisibleStratifiedNoneDatasetConfig,
    split: int,
    stratum_index: int,
    forbidden_hashes: set[str],
) -> tuple[dict[str, np.ndarray], set[str], int]:
    base_seed = config.reference_seed if split == 0 else config.audit_seed
    rng = np.random.default_rng(base_seed + 100_000 * stratum_index)
    frames = []
    states = []
    actions = []
    next_states = []
    terminateds = []
    hashes = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = config.samples_per_stratum * config.max_attempt_multiplier
    while len(frames) < config.samples_per_stratum:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"split {split} stratum {stratum_index} collected {len(frames)}/"
                f"{config.samples_per_stratum} after {attempts} attempts"
            )
        attempts += 1
        state, action = sample_candidate(rng, stratum_index, VISIBLE_STRATA)
        next_state, _, terminated = expected_transition(state, action)
        if not stratum_matches(
            stratum_index, next_state, terminated, VISIBLE_STRATA
        ):
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
    size = len(frames)
    return (
        {
            "frames": np.stack(frames).astype(np.uint8),
            "states": np.stack(states).astype(np.float32),
            "actions": np.stack(actions).astype(np.float32),
            "next_states": np.stack(next_states).astype(np.float32),
            "terminateds": np.asarray(terminateds, dtype=np.bool_),
            "pair_hashes": np.asarray(hashes),
            "classes": np.full(size, -1, dtype=np.int8),
            "splits": np.full(size, split, dtype=np.int8),
            "strata": np.full(size, stratum_index, dtype=np.int8),
        },
        seen,
        attempts,
    )


def _write_contact_sheet(
    path: Path,
    frames: np.ndarray,
    splits: np.ndarray,
    strata: np.ndarray,
) -> None:
    frame_height, frame_width = 200, 300
    marker, gap, row_gap = 10, 5, 6
    columns = 4
    sheet = np.full(
        (
            len(VISIBLE_STRATA) * frame_height
            + (len(VISIBLE_STRATA) - 1) * row_gap,
            marker + columns * frame_width + (columns - 1) * gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    for stratum_index in range(len(VISIBLE_STRATA)):
        y = stratum_index * (frame_height + row_gap)
        color = np.asarray(
            [
                60 + 20 * (stratum_index % 4),
                95 + 16 * stratum_index,
                155 - 10 * stratum_index,
            ],
            dtype=np.uint8,
        )
        sheet[y : y + frame_height, :marker] = color
        for split in range(2):
            index = np.flatnonzero((strata == stratum_index) & (splits == split))[0]
            for frame_index, frame in enumerate(frames[index]):
                column = 2 * split + frame_index
                x = marker + column * (frame_width + gap)
                sheet[y : y + frame_height, x : x + frame_width] = frame[::2, ::2]
    _write_png(path, sheet)


def generate_dataset(
    config: VisibleStratifiedNoneDatasetConfig,
    relation_dataset_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    relation_hashes = existing_relation_hashes(relation_dataset_dir)
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    env.reset(seed=config.reference_seed)
    rows = []
    attempts = np.zeros((2, len(VISIBLE_STRATA)), dtype=np.int64)
    split_hashes = [set(), set()]
    all_new_hashes: set[str] = set()
    try:
        for split in range(2):
            for stratum_index in range(len(VISIBLE_STRATA)):
                row, hashes, attempt_count = _collect_stratum(
                    env,
                    config,
                    split,
                    stratum_index,
                    relation_hashes | all_new_hashes,
                )
                rows.append(row)
                attempts[split, stratum_index] = attempt_count
                split_hashes[split].update(hashes)
                all_new_hashes.update(hashes)
    finally:
        env.close()
    arrays = {
        name: np.concatenate([row[name] for row in rows], axis=0)
        for name in rows[0]
    }
    counts = np.zeros((2, len(VISIBLE_STRATA)), dtype=np.int64)
    np.add.at(counts, (arrays["splits"], arrays["strata"]), 1)
    predicates_valid = all(
        stratum_matches(
            int(stratum_index),
            next_state,
            bool(terminated),
            VISIBLE_STRATA,
        )
        for stratum_index, next_state, terminated in zip(
            arrays["strata"],
            arrays["next_states"],
            arrays["terminateds"],
            strict=True,
        )
    )
    cross_split_overlap = split_hashes[0].intersection(split_hashes[1])
    relation_overlap = all_new_hashes.intersection(relation_hashes)
    expected_samples = 2 * len(VISIBLE_STRATA) * config.samples_per_stratum
    data_gate = bool(
        np.all(counts == config.samples_per_stratum)
        and len(all_new_hashes) == expected_samples
        and not cross_split_overlap
        and not relation_overlap
        and predicates_valid
        and arrays["frames"].shape == (expected_samples, 2, 400, 600, 3)
        and arrays["frames"].dtype == np.uint8
        and np.isfinite(arrays["states"]).all()
        and np.isfinite(arrays["actions"]).all()
        and np.isfinite(arrays["next_states"]).all()
    )
    shard_path = output_dir / "class_none.npz"
    np.savez_compressed(shard_path, **arrays)
    contact_sheet_path = output_dir / "visible_stratified_none_contact_sheet.png"
    _write_contact_sheet(
        contact_sheet_path,
        arrays["frames"],
        arrays["splits"],
        arrays["strata"],
    )
    output = {
        "config": asdict(config),
        "strata": [asdict(spec) for spec in VISIBLE_STRATA],
        "relation_dataset_dir": str(relation_dataset_dir.resolve()),
        "sdl_video_driver": os.environ.get("SDL_VIDEODRIVER"),
        "shard": str(shard_path.resolve()),
        "counts_by_split_stratum": counts.tolist(),
        "attempts_by_split_stratum": attempts.tolist(),
        "reference_unique_pair_hashes": len(split_hashes[0]),
        "audit_unique_pair_hashes": len(split_hashes[1]),
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
    config = VisibleStratifiedNoneDatasetConfig()
    run_id = (
        "visible_stratified_none_dataset_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/mountaincar_continuous"
    ) / run_id
    output = generate_dataset(config, args.relation_dataset_dir, output_dir)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
