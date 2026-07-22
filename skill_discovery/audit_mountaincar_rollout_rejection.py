"""Audit frozen MountainCar relation rejection on natural rollouts."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

from skill_discovery.audit_mountaincar_frame_pair_capacity import (
    PAIR_CLASSES,
    transition_class,
)
from skill_discovery.evaluate_mountaincar_frame_pair_metric import (
    car_x_pair_features,
)
from skill_discovery.generate_point_cup_dataset import _write_png


@dataclass(frozen=True)
class RolloutRejectionConfig:
    energy_episodes: int = 128
    random_episodes: int = 256
    random_horizon: int = 200
    energy_seed_start: int = 5_100_000
    random_seed_start: int = 6_100_000
    action_seed: int = 7_100_007
    batch_size: int = 32
    workers: int = 4

    def __post_init__(self) -> None:
        if self.energy_episodes <= 0 or self.random_episodes <= 0:
            raise ValueError("rollout episode budgets must be positive")
        if self.random_horizon <= 0 or self.batch_size <= 0 or self.workers <= 0:
            raise ValueError("rollout execution budgets must be positive")


def leave_one_out_thresholds(
    features: np.ndarray,
    classes: np.ndarray,
    splits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = splits == 0
    reference_features = np.asarray(features[reference], dtype=np.float32)
    reference_classes = np.asarray(classes[reference], dtype=np.int8)
    thresholds = np.empty(len(PAIR_CLASSES), dtype=np.float64)
    for class_index in range(len(PAIR_CLASSES)):
        rows = reference_features[reference_classes == class_index].astype(np.float64)
        if len(rows) < 2:
            raise ValueError("each reference class needs at least two samples")
        distances = np.square(rows[:, None] - rows[None]).sum(axis=2)
        np.fill_diagonal(distances, np.inf)
        thresholds[class_index] = np.min(distances, axis=1).max()
    return thresholds, reference_features, reference_classes


def predict_with_rejection(
    features: np.ndarray,
    reference_features: np.ndarray,
    reference_classes: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(features, dtype=np.float64)
    reference = np.asarray(reference_features, dtype=np.float64)
    distances = np.square(rows[:, None] - reference[None]).sum(axis=2)
    nearest = np.argmin(distances, axis=1)
    predicted = reference_classes[nearest].astype(np.int8)
    nearest_distances = distances[np.arange(len(rows)), nearest]
    accepted = nearest_distances <= thresholds[predicted]
    predicted = np.where(accepted, predicted, -1).astype(np.int8)
    return predicted, nearest_distances


def predict_nearest(
    features: np.ndarray,
    reference_features: np.ndarray,
    reference_classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(features, dtype=np.float64)
    reference = np.asarray(reference_features, dtype=np.float64)
    distances = np.square(rows[:, None] - reference[None]).sum(axis=2)
    nearest = np.argmin(distances, axis=1)
    predicted = reference_classes[nearest].astype(np.int8)
    nearest_distances = distances[np.arange(len(rows)), nearest]
    return predicted, nearest_distances


def oracle_relation(next_state: np.ndarray, terminated: bool) -> int:
    matches = [
        class_index
        for class_index in range(len(PAIR_CLASSES))
        if transition_class(class_index, next_state, terminated)
    ]
    if len(matches) > 1:
        raise RuntimeError("MountainCar relation predicates overlap")
    return matches[0] if matches else -1


def _flush_batch(
    frames: list[np.ndarray],
    oracle_labels: list[int],
    sources: list[int],
    background: np.ndarray,
    reference_features: np.ndarray,
    reference_classes: np.ndarray,
    thresholds: np.ndarray | None,
    predictions: list[np.ndarray],
    distances: list[np.ndarray],
    all_oracle: list[np.ndarray],
    all_sources: list[np.ndarray],
    examples: dict[tuple[int, int], dict[str, object]],
) -> None:
    if not frames:
        return
    batch = np.stack(frames).astype(np.uint8)
    features, _ = car_x_pair_features(batch, background)
    if thresholds is None:
        predicted, nearest_distances = predict_nearest(
            features, reference_features, reference_classes
        )
    else:
        predicted, nearest_distances = predict_with_rejection(
            features, reference_features, reference_classes, thresholds
        )
    oracle = np.asarray(oracle_labels, dtype=np.int8)
    source = np.asarray(sources, dtype=np.int8)
    predictions.append(predicted)
    distances.append(nearest_distances)
    all_oracle.append(oracle)
    all_sources.append(source)
    for index, (expected, observed) in enumerate(zip(oracle, predicted, strict=True)):
        key = (int(expected), int(observed))
        if key not in examples:
            examples[key] = {
                "oracle": int(expected),
                "predicted": int(observed),
                "nearest_squared_distance": float(nearest_distances[index]),
                "frames": batch[index],
            }
    frames.clear()
    oracle_labels.clear()
    sources.clear()


def rollout_worker(
    config: RolloutRejectionConfig,
    worker_index: int,
    background: np.ndarray,
    reference_features: np.ndarray,
    reference_classes: np.ndarray,
    thresholds: np.ndarray | None,
) -> dict[str, object]:
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    action_rng = np.random.default_rng(config.action_seed + worker_index)
    frame_batch: list[np.ndarray] = []
    oracle_batch: list[int] = []
    source_batch: list[int] = []
    predictions: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    oracle_rows: list[np.ndarray] = []
    source_rows: list[np.ndarray] = []
    examples: dict[tuple[int, int], dict[str, object]] = {}

    def append_transition(
        before: np.ndarray,
        after: np.ndarray,
        relation: int,
        source: int,
    ) -> None:
        frame_batch.append(np.stack((before, after)))
        oracle_batch.append(relation)
        source_batch.append(source)
        if len(frame_batch) >= config.batch_size:
            _flush_batch(
                frame_batch,
                oracle_batch,
                source_batch,
                background,
                reference_features,
                reference_classes,
                thresholds,
                predictions,
                distances,
                oracle_rows,
                source_rows,
                examples,
            )

    try:
        for episode_index in range(worker_index, config.energy_episodes, config.workers):
            observation, _ = env.reset(seed=config.energy_seed_start + episode_index)
            before = env.render().copy()
            for _ in range(999):
                action_value = -1.0 if observation[1] <= 0.0 else 1.0
                action = np.asarray([action_value], dtype=np.float32)
                observation, _, terminated, truncated, _ = env.step(action)
                after = env.render().copy()
                append_transition(
                    before,
                    after,
                    oracle_relation(observation, bool(terminated)),
                    0,
                )
                before = after
                if terminated or truncated:
                    break
        for episode_index in range(worker_index, config.random_episodes, config.workers):
            observation, _ = env.reset(seed=config.random_seed_start + episode_index)
            before = env.render().copy()
            for _ in range(config.random_horizon):
                action = np.asarray(
                    [action_rng.uniform(-1.0, 1.0)], dtype=np.float32
                )
                observation, _, terminated, truncated, _ = env.step(action)
                after = env.render().copy()
                append_transition(
                    before,
                    after,
                    oracle_relation(observation, bool(terminated)),
                    1,
                )
                before = after
                if terminated or truncated:
                    break
        _flush_batch(
            frame_batch,
            oracle_batch,
            source_batch,
            background,
            reference_features,
            reference_classes,
            thresholds,
            predictions,
            distances,
            oracle_rows,
            source_rows,
            examples,
        )
    finally:
        env.close()
    return {
        "predictions": np.concatenate(predictions),
        "distances": np.concatenate(distances),
        "oracle": np.concatenate(oracle_rows),
        "sources": np.concatenate(source_rows),
        "examples": examples,
    }


def rejection_metrics(
    oracle: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, object]:
    confusion = np.zeros((5, 5), dtype=np.int64)
    np.add.at(confusion, (oracle + 1, predictions + 1), 1)
    oracle_counts = np.bincount(oracle + 1, minlength=5)
    predicted_counts = np.bincount(predictions + 1, minlength=5)
    recalls = {
        name: float(
            np.mean(predictions[oracle == class_index] == class_index)
        )
        if np.any(oracle == class_index)
        else 0.0
        for class_index, name in enumerate(PAIR_CLASSES)
    }
    none = oracle == -1
    none_false_positive = float(np.mean(predictions[none] >= 0)) if np.any(none) else 0.0
    none_to_goal = float(np.mean(predictions[none] == 3)) if np.any(none) else 0.0
    return {
        "sample_count": len(oracle),
        "oracle_counts": {
            "none": int(oracle_counts[0]),
            **{
                name: int(oracle_counts[index + 1])
                for index, name in enumerate(PAIR_CLASSES)
            },
        },
        "predicted_counts": {
            "none": int(predicted_counts[0]),
            **{
                name: int(predicted_counts[index + 1])
                for index, name in enumerate(PAIR_CLASSES)
            },
        },
        "relation_recall": recalls,
        "relation_macro_recall": float(np.mean(list(recalls.values()))),
        "none_false_positive_rate": none_false_positive,
        "none_to_goal_rate": none_to_goal,
        "confusion_none_left_valley_right_goal": confusion.tolist(),
    }


def _select_examples(
    worker_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    combined: dict[tuple[int, int], dict[str, object]] = {}
    for row in worker_rows:
        for key, example in row["examples"].items():
            combined.setdefault(key, example)
    selected = []
    selected_keys = set()
    for relation in (-1, 0, 1, 2, 3):
        preferred = (relation, relation)
        candidates = [key for key in combined if key[0] == relation]
        key = preferred if preferred in combined else (candidates[0] if candidates else None)
        if key is not None:
            selected.append(combined[key])
            selected_keys.add(key)
    if (-1, 3) in combined and (-1, 3) not in selected_keys:
        selected.append(combined[(-1, 3)])
    return selected


def _write_contact_sheet(
    path: Path,
    examples: list[dict[str, object]],
) -> None:
    frame_height, frame_width = 200, 300
    marker = 10
    gap = 5
    row_gap = 6
    sheet = np.full(
        (
            len(examples) * frame_height + max(len(examples) - 1, 0) * row_gap,
            marker + 2 * frame_width + gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    colors = np.asarray(
        (
            [100, 110, 118],
            [34, 114, 157],
            [214, 93, 74],
            [64, 145, 108],
            [146, 92, 156],
            [190, 70, 55],
        ),
        dtype=np.uint8,
    )
    for row_index, example in enumerate(examples):
        y = row_index * (frame_height + row_gap)
        sheet[y : y + frame_height, :marker] = colors[row_index]
        for frame_index, frame in enumerate(example["frames"]):
            x = marker + frame_index * (frame_width + gap)
            sheet[y : y + frame_height, x : x + frame_width] = frame[::2, ::2]
    _write_png(path, sheet)


def run_audit(
    config: RolloutRejectionConfig,
    feature_path: Path,
    background_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    with np.load(feature_path) as feature_data:
        features = feature_data["car_features"].astype(np.float32)
        classes = feature_data["classes"].astype(np.int8)
        splits = feature_data["splits"].astype(np.int8)
    thresholds, reference_features, reference_classes = leave_one_out_thresholds(
        features, classes, splits
    )
    background = np.asarray(Image.open(background_path).convert("RGB"), dtype=np.uint8)
    output_dir.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        worker_rows = list(
            pool.map(
                rollout_worker,
                [config] * config.workers,
                range(config.workers),
                [background] * config.workers,
                [reference_features] * config.workers,
                [reference_classes] * config.workers,
                [thresholds] * config.workers,
            )
        )
    predictions = np.concatenate([row["predictions"] for row in worker_rows])
    distances = np.concatenate([row["distances"] for row in worker_rows])
    oracle = np.concatenate([row["oracle"] for row in worker_rows])
    sources = np.concatenate([row["sources"] for row in worker_rows])
    combined = rejection_metrics(oracle, predictions)
    energy = rejection_metrics(oracle[sources == 0], predictions[sources == 0])
    random = rejection_metrics(oracle[sources == 1], predictions[sources == 1])
    coverage_gate = bool(
        min(combined["oracle_counts"][name] for name in PAIR_CLASSES) >= 100
        and combined["oracle_counts"]["none"] >= 10_000
    )
    relation_gate = bool(
        combined["relation_macro_recall"] >= 0.95
        and min(combined["relation_recall"].values()) >= 0.90
        and combined["relation_recall"]["native_goal"] >= 0.95
        and all(combined["predicted_counts"][name] > 0 for name in PAIR_CLASSES)
    )
    false_positive_gate = bool(
        combined["none_false_positive_rate"] <= 0.10
        and combined["none_to_goal_rate"] <= 0.01
    )
    examples = _select_examples(worker_rows)
    contact_sheet_path = output_dir / "rollout_rejection_contact_sheet.png"
    _write_contact_sheet(contact_sheet_path, examples)
    public_examples = [
        {key: value for key, value in example.items() if key != "frames"}
        for example in examples
    ]
    output = {
        "config": asdict(config),
        "feature_path": str(feature_path.resolve()),
        "background_path": str(background_path.resolve()),
        "rejection_thresholds_squared": {
            name: float(thresholds[index])
            for index, name in enumerate(PAIR_CLASSES)
        },
        "distance_summary": {
            "minimum": float(distances.min()),
            "median": float(np.median(distances)),
            "maximum": float(distances.max()),
        },
        "combined": combined,
        "energy": energy,
        "random": random,
        "coverage_gate_passed": coverage_gate,
        "relation_recall_gate_passed": relation_gate,
        "false_positive_gate_passed": false_positive_gate,
        "numeric_gate_passed": bool(
            coverage_gate and relation_gate and false_positive_gate
        ),
        "contact_sheet": str(contact_sheet_path.resolve()),
        "contact_examples": public_examples,
        "manual_contact_sheet_gate": "pending",
        "passed": False,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("background", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = RolloutRejectionConfig()
    run_id = f"rollout_rejection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/mountaincar_continuous"
    ) / run_id
    output = run_audit(config, args.features, args.background, output_dir)
    print(
        json.dumps(
            {
                "output": str((output_dir / "audit.json").resolve()),
                "thresholds": output["rejection_thresholds_squared"],
                "combined": output["combined"],
                "coverage_gate_passed": output["coverage_gate_passed"],
                "relation_recall_gate_passed": output["relation_recall_gate_passed"],
                "false_positive_gate_passed": output["false_positive_gate_passed"],
                "numeric_gate_passed": output["numeric_gate_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
