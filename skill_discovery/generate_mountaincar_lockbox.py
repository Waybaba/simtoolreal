"""Generate the fresh hash-isolated MountainCar balanced lockbox."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

from skill_discovery.audit_mountaincar_continuous_environment import (
    expected_transition,
)
from skill_discovery.audit_mountaincar_frame_pair_capacity import (
    PAIR_CLASSES,
    PROPOSAL_RANGES,
    pair_hash,
    transition_class,
)
from skill_discovery.audit_mountaincar_stratified_none_capacity import (
    VISIBLE_STRATA,
    sample_candidate,
    stratum_matches,
)
from skill_discovery.evaluate_mountaincar_frame_pair_metric import (
    car_x_pair_features,
)
from skill_discovery.generate_point_cup_dataset import _write_png


@dataclass(frozen=True)
class MountainCarLockboxConfig:
    samples_per_relation: int = 256
    samples_per_none_stratum: int = 32
    relation_seed: int = 12_100_007
    none_seed: int = 13_100_007
    max_attempt_multiplier: int = 100

    def __post_init__(self) -> None:
        if self.samples_per_relation <= 0 or self.samples_per_none_stratum <= 0:
            raise ValueError("lockbox sample budgets must be positive")
        if self.max_attempt_multiplier <= 0:
            raise ValueError("attempt multiplier must be positive")


def load_forbidden_hashes(
    relation_dataset_dir: Path,
    prior_none_shards: tuple[Path, ...],
) -> set[str]:
    shard_paths = sorted(relation_dataset_dir.glob("class_*.npz"))
    if len(shard_paths) != 4:
        raise ValueError("expected four prior relation shards")
    hashes: set[str] = set()
    for shard_path in (*shard_paths, *prior_none_shards):
        with np.load(shard_path) as shard:
            hashes.update(str(value) for value in shard["pair_hashes"])
    return hashes


def _render_pair(
    env: object,
    state: np.ndarray,
    next_state: np.ndarray,
) -> np.ndarray:
    env.unwrapped.state = state.copy()
    before = env.render().copy()
    env.unwrapped.state = next_state.copy()
    after = env.render().copy()
    return np.stack((before, after))


def _pack_rows(
    frames: list[np.ndarray],
    states: list[np.ndarray],
    actions: list[np.ndarray],
    next_states: list[np.ndarray],
    terminateds: list[bool],
    hashes: list[str],
    class_index: int,
    stratum_index: int,
) -> dict[str, np.ndarray]:
    size = len(frames)
    return {
        "frames": np.stack(frames).astype(np.uint8),
        "states": np.stack(states).astype(np.float32),
        "actions": np.stack(actions).astype(np.float32),
        "next_states": np.stack(next_states).astype(np.float32),
        "terminateds": np.asarray(terminateds, dtype=np.bool_),
        "pair_hashes": np.asarray(hashes),
        "classes": np.full(size, class_index, dtype=np.int8),
        "strata": np.full(size, stratum_index, dtype=np.int8),
    }


def _collect_relation(
    env: object,
    config: MountainCarLockboxConfig,
    class_index: int,
    forbidden_hashes: set[str],
) -> tuple[dict[str, np.ndarray], set[str], int]:
    seed = config.relation_seed + 100_000 * class_index
    rng = np.random.default_rng(seed)
    position_range, velocity_range = PROPOSAL_RANGES[class_index]
    frames = []
    states = []
    actions = []
    next_states = []
    terminateds = []
    hashes = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = config.samples_per_relation * config.max_attempt_multiplier
    while len(frames) < config.samples_per_relation:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"{PAIR_CLASSES[class_index]} lockbox collected {len(frames)}/"
                f"{config.samples_per_relation} after {attempts} attempts"
            )
        attempts += 1
        state = np.asarray(
            [rng.uniform(*position_range), rng.uniform(*velocity_range)],
            dtype=np.float32,
        )
        action = np.asarray([rng.uniform(-1.0, 1.0)], dtype=np.float32)
        next_state, _, terminated = expected_transition(state, action)
        if not transition_class(class_index, next_state, terminated):
            continue
        pair = _render_pair(env, state, next_state)
        digest = pair_hash(pair[0], pair[1])
        if digest in seen or digest in forbidden_hashes:
            continue
        frames.append(pair)
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        terminateds.append(terminated)
        hashes.append(digest)
        seen.add(digest)
    return (
        _pack_rows(
            frames,
            states,
            actions,
            next_states,
            terminateds,
            hashes,
            class_index,
            -1,
        ),
        seen,
        attempts,
    )


def _collect_none_stratum(
    env: object,
    config: MountainCarLockboxConfig,
    stratum_index: int,
    forbidden_hashes: set[str],
) -> tuple[dict[str, np.ndarray], set[str], int]:
    seed = config.none_seed + 100_000 * stratum_index
    rng = np.random.default_rng(seed)
    frames = []
    states = []
    actions = []
    next_states = []
    terminateds = []
    hashes = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = config.samples_per_none_stratum * config.max_attempt_multiplier
    while len(frames) < config.samples_per_none_stratum:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"none stratum {stratum_index} collected {len(frames)}/"
                f"{config.samples_per_none_stratum} after {attempts} attempts"
            )
        attempts += 1
        state, action = sample_candidate(rng, stratum_index, VISIBLE_STRATA)
        next_state, _, terminated = expected_transition(state, action)
        if not stratum_matches(
            stratum_index, next_state, terminated, VISIBLE_STRATA
        ):
            continue
        pair = _render_pair(env, state, next_state)
        digest = pair_hash(pair[0], pair[1])
        if digest in seen or digest in forbidden_hashes:
            continue
        frames.append(pair)
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        terminateds.append(terminated)
        hashes.append(digest)
        seen.add(digest)
    return (
        _pack_rows(
            frames,
            states,
            actions,
            next_states,
            terminateds,
            hashes,
            -1,
            stratum_index,
        ),
        seen,
        attempts,
    )


def _write_rows_contact_sheet(
    path: Path,
    rows: list[np.ndarray],
) -> None:
    frame_height, frame_width = 200, 300
    marker, gap, row_gap = 10, 5, 6
    columns = 4
    sheet = np.full(
        (
            len(rows) * frame_height + (len(rows) - 1) * row_gap,
            marker + columns * frame_width + (columns - 1) * gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    for row_index, frames in enumerate(rows):
        y = row_index * (frame_height + row_gap)
        color = np.asarray(
            [70 + 18 * (row_index % 5), 90 + 20 * row_index, 160 - 12 * row_index],
            dtype=np.uint8,
        )
        sheet[y : y + frame_height, :marker] = color
        for example_index in range(2):
            for frame_index, frame in enumerate(frames[example_index]):
                column = 2 * example_index + frame_index
                x = marker + column * (frame_width + gap)
                sheet[y : y + frame_height, x : x + frame_width] = frame[::2, ::2]
    _write_png(path, sheet)


def generate_lockbox(
    config: MountainCarLockboxConfig,
    relation_dataset_dir: Path,
    prior_none_shards: tuple[Path, ...],
    background_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    prior_hashes = load_forbidden_hashes(relation_dataset_dir, prior_none_shards)
    background = np.asarray(Image.open(background_path).convert("RGB"), dtype=np.uint8)
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    env.reset(seed=config.relation_seed)
    all_new_hashes: set[str] = set()
    feature_rows = []
    car_x_rows = []
    metadata_rows: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "states",
            "actions",
            "next_states",
            "terminateds",
            "pair_hashes",
            "classes",
            "strata",
        )
    }
    class_contact_rows = []
    none_contact_rows = []
    relation_attempts = []
    none_attempts = []
    predicates_valid = True
    frame_gate = True
    shard_paths = []
    try:
        for class_index in range(len(PAIR_CLASSES)):
            arrays, hashes, attempts = _collect_relation(
                env,
                config,
                class_index,
                prior_hashes | all_new_hashes,
            )
            relation_attempts.append(attempts)
            all_new_hashes.update(hashes)
            predicates_valid &= all(
                transition_class(class_index, next_state, bool(terminated))
                for next_state, terminated in zip(
                    arrays["next_states"], arrays["terminateds"], strict=True
                )
            )
            frame_gate &= bool(
                arrays["frames"].shape
                == (config.samples_per_relation, 2, 400, 600, 3)
                and arrays["frames"].dtype == np.uint8
            )
            features, car_x = car_x_pair_features(arrays["frames"], background)
            feature_rows.append(features)
            car_x_rows.append(car_x)
            class_contact_rows.append(arrays["frames"][:2].copy())
            shard_path = output_dir / f"class_{class_index}_{PAIR_CLASSES[class_index]}.npz"
            np.savez_compressed(shard_path, **arrays)
            shard_paths.append(shard_path)
            for name in metadata_rows:
                metadata_rows[name].append(arrays[name])
            del arrays
        none_rows = []
        for stratum_index in range(len(VISIBLE_STRATA)):
            arrays, hashes, attempts = _collect_none_stratum(
                env,
                config,
                stratum_index,
                prior_hashes | all_new_hashes,
            )
            none_attempts.append(attempts)
            all_new_hashes.update(hashes)
            predicates_valid &= all(
                stratum_matches(
                    stratum_index,
                    next_state,
                    bool(terminated),
                    VISIBLE_STRATA,
                )
                for next_state, terminated in zip(
                    arrays["next_states"], arrays["terminateds"], strict=True
                )
            )
            frame_gate &= bool(
                arrays["frames"].shape
                == (config.samples_per_none_stratum, 2, 400, 600, 3)
                and arrays["frames"].dtype == np.uint8
            )
            features, car_x = car_x_pair_features(arrays["frames"], background)
            feature_rows.append(features)
            car_x_rows.append(car_x)
            none_contact_rows.append(arrays["frames"][:2].copy())
            none_rows.append(arrays)
        none_arrays = {
            name: np.concatenate([row[name] for row in none_rows], axis=0)
            for name in none_rows[0]
        }
        none_shard_path = output_dir / "class_none.npz"
        np.savez_compressed(none_shard_path, **none_arrays)
        shard_paths.append(none_shard_path)
        for name in metadata_rows:
            metadata_rows[name].append(none_arrays[name])
        class_contact_rows.append(none_arrays["frames"][:2].copy())
        del none_rows, none_arrays
    finally:
        env.close()
    combined = {
        name: np.concatenate(rows, axis=0) for name, rows in metadata_rows.items()
    }
    features = np.concatenate(feature_rows)
    car_x = np.concatenate(car_x_rows)
    expected_count = len(PAIR_CLASSES) * config.samples_per_relation + len(
        VISIBLE_STRATA
    ) * config.samples_per_none_stratum
    class_counts = {
        "none": int(np.sum(combined["classes"] == -1)),
        **{
            name: int(np.sum(combined["classes"] == class_index))
            for class_index, name in enumerate(PAIR_CLASSES)
        },
    }
    expected_class_counts = {
        "none": len(VISIBLE_STRATA) * config.samples_per_none_stratum,
        **{
            name: config.samples_per_relation
            for name in PAIR_CLASSES
        },
    }
    prior_overlap = all_new_hashes.intersection(prior_hashes)
    finite_gate = bool(
        np.isfinite(features).all()
        and np.isfinite(car_x).all()
        and np.isfinite(combined["states"]).all()
        and np.isfinite(combined["actions"]).all()
        and np.isfinite(combined["next_states"]).all()
    )
    data_gate = bool(
        len(all_new_hashes) == expected_count
        and not prior_overlap
        and class_counts == expected_class_counts
        and predicates_valid
        and frame_gate
        and finite_gate
        and features.shape == (expected_count, 3)
    )
    feature_path = output_dir / "lockbox_car_motion_features.npz"
    np.savez_compressed(
        feature_path,
        car_features=features,
        car_x=car_x,
        **combined,
    )
    class_contact_path = output_dir / "lockbox_class_contact_sheet.png"
    none_contact_path = output_dir / "lockbox_none_strata_contact_sheet.png"
    _write_rows_contact_sheet(class_contact_path, class_contact_rows)
    _write_rows_contact_sheet(none_contact_path, none_contact_rows)
    output = {
        "config": asdict(config),
        "sdl_video_driver": os.environ.get("SDL_VIDEODRIVER"),
        "relation_dataset_dir": str(relation_dataset_dir.resolve()),
        "prior_none_shards": [str(path.resolve()) for path in prior_none_shards],
        "background_path": str(background_path.resolve()),
        "prior_forbidden_pair_hashes": len(prior_hashes),
        "lockbox_unique_pair_hashes": len(all_new_hashes),
        "prior_pair_hash_overlap": len(prior_overlap),
        "class_counts": class_counts,
        "relation_attempts": relation_attempts,
        "none_stratum_attempts": none_attempts,
        "predicates_valid": predicates_valid,
        "frame_gate_passed": frame_gate,
        "finite_gate_passed": finite_gate,
        "shards": [str(path.resolve()) for path in shard_paths],
        "features": str(feature_path.resolve()),
        "class_contact_sheet": str(class_contact_path.resolve()),
        "none_strata_contact_sheet": str(none_contact_path.resolve()),
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
    parser.add_argument("uniform_none_shard", type=Path)
    parser.add_argument("stratified_none_shard", type=Path)
    parser.add_argument("background", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = MountainCarLockboxConfig()
    run_id = f"lockbox_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/mountaincar_continuous"
    ) / run_id
    output = generate_lockbox(
        config,
        args.relation_dataset_dir,
        (args.uniform_none_shard, args.stratified_none_shard),
        args.background,
        output_dir,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
