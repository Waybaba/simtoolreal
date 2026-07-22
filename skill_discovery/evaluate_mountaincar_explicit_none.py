"""Evaluate an explicit none class on balanced and natural MountainCar pairs."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from skill_discovery.audit_mountaincar_frame_pair_capacity import PAIR_CLASSES
from skill_discovery.audit_mountaincar_rollout_rejection import (
    RolloutRejectionConfig,
    _select_examples,
    _write_contact_sheet,
    predict_nearest,
    rejection_metrics,
    rollout_worker,
)
from skill_discovery.evaluate_mountaincar_frame_pair_metric import (
    car_x_pair_features,
)


CLASS_NAMES = ("none", *PAIR_CLASSES)


def five_class_metrics(
    oracle: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, object]:
    oracle = np.asarray(oracle, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    if not np.all((-1 <= oracle) & (oracle < len(PAIR_CLASSES))):
        raise ValueError("oracle labels must be in [-1, 3]")
    if not np.all((-1 <= predictions) & (predictions < len(PAIR_CLASSES))):
        raise ValueError("predicted labels must be in [-1, 3]")
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(confusion, (oracle + 1, predictions + 1), 1)
    recalls = {
        name: float(confusion[index, index] / max(confusion[index].sum(), 1))
        for index, name in enumerate(CLASS_NAMES)
    }
    return {
        "sample_count": len(oracle),
        "accuracy": float(np.mean(predictions == oracle)),
        "macro_recall": float(np.mean(list(recalls.values()))),
        "recall_by_class": recalls,
        "confusion_none_left_valley_right_goal": confusion.tolist(),
        "predicted_class_sizes": {
            name: int(np.sum(predictions == index - 1))
            for index, name in enumerate(CLASS_NAMES)
        },
    }


def _class_split_counts(classes: np.ndarray, splits: np.ndarray) -> dict[str, object]:
    return {
        split_name: {
            name: int(np.sum((classes == class_index - 1) & (splits == split)))
            for class_index, name in enumerate(CLASS_NAMES)
        }
        for split, split_name in enumerate(("reference", "audit"))
    }


def _load_five_class_features(
    feature_path: Path,
    background_path: Path,
    none_shard_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(feature_path) as relation_data:
        relation_features = relation_data["car_features"].astype(np.float32)
        relation_classes = relation_data["classes"].astype(np.int8)
        relation_splits = relation_data["splits"].astype(np.int8)
        relation_hashes = relation_data["pair_hashes"]
    background = np.asarray(Image.open(background_path).convert("RGB"), dtype=np.uint8)
    with np.load(none_shard_path) as none_data:
        none_frames = none_data["frames"].astype(np.uint8)
        none_classes = none_data["classes"].astype(np.int8)
        none_splits = none_data["splits"].astype(np.int8)
        none_hashes = none_data["pair_hashes"]
    none_features, _ = car_x_pair_features(none_frames, background)
    del none_frames
    return (
        np.concatenate((relation_features, none_features)),
        np.concatenate((relation_classes, none_classes)),
        np.concatenate((relation_splits, none_splits)),
        np.concatenate((relation_hashes, none_hashes)),
        background,
    )


def _paired_protocol_gate(
    config: RolloutRejectionConfig,
    baseline_audit_path: Path,
) -> tuple[bool, dict[str, object]]:
    baseline = json.loads(baseline_audit_path.read_text(encoding="utf-8"))
    baseline_config = baseline["config"]
    return baseline_config == asdict(config), baseline_config


def _run_natural_audit(
    config: RolloutRejectionConfig,
    background: np.ndarray,
    reference_features: np.ndarray,
    reference_classes: np.ndarray,
    output_dir: Path,
) -> dict[str, object]:
    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        worker_rows = list(
            pool.map(
                rollout_worker,
                [config] * config.workers,
                range(config.workers),
                [background] * config.workers,
                [reference_features] * config.workers,
                [reference_classes] * config.workers,
                [None] * config.workers,
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
    )
    none_gate = bool(
        combined["none_false_positive_rate"] <= 0.10
        and combined["none_to_goal_rate"] <= 0.01
    )
    predicted_coverage_gate = all(
        combined["predicted_counts"][name] > 0 for name in CLASS_NAMES
    )
    examples = _select_examples(worker_rows)
    contact_sheet_path = output_dir / "explicit_none_rollout_contact_sheet.png"
    _write_contact_sheet(contact_sheet_path, examples)
    public_examples = [
        {key: value for key, value in example.items() if key != "frames"}
        for example in examples
    ]
    return {
        "combined": combined,
        "energy": energy,
        "random": random,
        "distance_summary": {
            "minimum": float(distances.min()),
            "median": float(np.median(distances)),
            "maximum": float(distances.max()),
        },
        "coverage_gate_passed": coverage_gate,
        "relation_recall_gate_passed": relation_gate,
        "none_gate_passed": none_gate,
        "predicted_coverage_gate_passed": predicted_coverage_gate,
        "numeric_gate_passed": bool(
            coverage_gate and relation_gate and none_gate and predicted_coverage_gate
        ),
        "contact_sheet": str(contact_sheet_path.resolve()),
        "contact_examples": public_examples,
        "manual_contact_sheet_gate": "pending",
    }


def evaluate_explicit_none(
    config: RolloutRejectionConfig,
    feature_path: Path,
    background_path: Path,
    none_shard_path: Path,
    baseline_audit_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol_gate, baseline_config = _paired_protocol_gate(config, baseline_audit_path)
    if not protocol_gate:
        raise RuntimeError("rollout config does not match the Phase 5ZW baseline")
    features, classes, splits, pair_hashes, background = _load_five_class_features(
        feature_path, background_path, none_shard_path
    )
    counts = _class_split_counts(classes, splits)
    reference = splits == 0
    audit = splits == 1
    reference_hashes = set(str(value) for value in pair_hashes[reference])
    audit_hashes = set(str(value) for value in pair_hashes[audit])
    data_gate = bool(
        all(count == 256 for split in counts.values() for count in split.values())
        and len(reference_hashes) == 5 * 256
        and len(audit_hashes) == 5 * 256
        and not reference_hashes.intersection(audit_hashes)
        and np.isfinite(features).all()
    )
    balanced_predictions, balanced_distances = predict_nearest(
        features[audit], features[reference], classes[reference]
    )
    balanced = five_class_metrics(classes[audit], balanced_predictions)
    balanced_gate = bool(
        data_gate
        and balanced["accuracy"] >= 0.98
        and balanced["macro_recall"] >= 0.98
        and min(balanced["recall_by_class"].values()) >= 0.95
    )
    combined_feature_path = output_dir / "five_class_car_motion_features.npz"
    np.savez_compressed(
        combined_feature_path,
        car_features=features,
        classes=classes,
        splits=splits,
        pair_hashes=pair_hashes,
    )
    natural = None
    if balanced_gate:
        natural = _run_natural_audit(
            config,
            background,
            features[reference],
            classes[reference],
            output_dir,
        )
    output = {
        "config": asdict(config),
        "baseline_config": baseline_config,
        "paired_protocol_gate_passed": protocol_gate,
        "feature_path": str(feature_path.resolve()),
        "background_path": str(background_path.resolve()),
        "none_shard_path": str(none_shard_path.resolve()),
        "baseline_audit_path": str(baseline_audit_path.resolve()),
        "combined_features": str(combined_feature_path.resolve()),
        "class_split_counts": counts,
        "data_gate_passed": data_gate,
        "balanced_audit": balanced,
        "balanced_distance_summary": {
            "minimum": float(balanced_distances.min()),
            "median": float(np.median(balanced_distances)),
            "maximum": float(balanced_distances.max()),
        },
        "balanced_gate_passed": balanced_gate,
        "natural_audit": natural,
        "natural_audit_status": "completed" if natural is not None else "not_run",
        "numeric_gate_passed": bool(
            balanced_gate and natural is not None and natural["numeric_gate_passed"]
        ),
        "passed": False,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["metrics"] = str(metrics_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("background", type=Path)
    parser.add_argument("none_shard", type=Path)
    parser.add_argument("baseline_audit", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_id = f"explicit_none_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/mountaincar_continuous"
    ) / run_id
    output = evaluate_explicit_none(
        RolloutRejectionConfig(),
        args.features,
        args.background,
        args.none_shard,
        args.baseline_audit,
        output_dir,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
