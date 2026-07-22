"""Fit and audit the frozen MountainCar isotonic visual decoder."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.isotonic import IsotonicRegression

from skill_discovery.evaluate_mountaincar_explicit_none import (
    CLASS_NAMES,
    five_class_metrics,
)
from skill_discovery.evaluate_mountaincar_visual_state_decoder import (
    _error_summary,
    _load_targets,
    decoded_relations,
)


def build_decoder() -> IsotonicRegression:
    return IsotonicRegression(
        increasing=True,
        y_min=-1.2,
        y_max=0.6,
        out_of_bounds="clip",
    )


def decode_visual_pairs(
    decoder: IsotonicRegression,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted_before = decoder.predict(features[:, 0])
    predicted_after = decoder.predict(features[:, 1])
    predicted_velocity = predicted_after - predicted_before
    left_wall = (predicted_after <= -1.199) & (predicted_velocity < 0.0)
    predicted_velocity[left_wall] = 0.0
    return predicted_before, predicted_after, predicted_velocity


def run_balanced_audit(
    feature_path: Path,
    relation_dataset_dir: Path,
    none_shard_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(feature_path) as feature_data:
        features = feature_data["car_features"].astype(np.float32)
        classes = feature_data["classes"].astype(np.int8)
        splits = feature_data["splits"].astype(np.int8)
        pair_hashes = feature_data["pair_hashes"]
    targets = _load_targets(relation_dataset_dir, none_shard_path)
    alignment_gate = bool(
        np.array_equal(pair_hashes, targets["pair_hashes"])
        and np.array_equal(classes, targets["classes"])
        and np.array_equal(splits, targets["splits"])
    )
    if not alignment_gate:
        raise RuntimeError("decoder targets do not align with frozen visual features")
    reference = splits == 0
    audit = splits == 1
    training_visual_x = np.concatenate(
        (features[reference, 0], features[reference, 1])
    )
    training_position = np.concatenate(
        (
            targets["states"][reference, 0],
            targets["next_states"][reference, 0],
        )
    )
    decoder = build_decoder()
    decoder.fit(training_visual_x, training_position)
    predicted_before, predicted_after, predicted_velocity = decode_visual_pairs(
        decoder, features
    )
    audit_position_errors = np.concatenate(
        (
            np.abs(predicted_before[audit] - targets["states"][audit, 0]),
            np.abs(predicted_after[audit] - targets["next_states"][audit, 0]),
        )
    )
    audit_velocity_errors = np.abs(
        predicted_velocity[audit] - targets["next_states"][audit, 1]
    )
    position_metrics = _error_summary(audit_position_errors)
    velocity_metrics = _error_summary(audit_velocity_errors)
    regression_gate = bool(
        position_metrics["mae"] <= 0.0015
        and position_metrics["p99_absolute_error"] <= 0.0040
        and velocity_metrics["mae"] <= 0.00075
        and velocity_metrics["p99_absolute_error"] <= 0.0020
    )
    predictions = decoded_relations(predicted_after[audit], predicted_velocity[audit])
    balanced = five_class_metrics(classes[audit], predictions)
    predicted_coverage_gate = all(
        balanced["predicted_class_sizes"][name] > 0 for name in CLASS_NAMES
    )
    classification_gate = bool(
        balanced["accuracy"] >= 0.98
        and balanced["macro_recall"] >= 0.98
        and min(balanced["recall_by_class"].values()) >= 0.95
        and predicted_coverage_gate
    )
    model_path = output_dir / "isotonic_position_decoder.joblib"
    joblib.dump(decoder, model_path)
    output = {
        "sklearn_version": sklearn.__version__,
        "model_config": {
            "increasing": True,
            "y_min": -1.2,
            "y_max": 0.6,
            "out_of_bounds": "clip",
            "left_wall_threshold": -1.199,
        },
        "feature_path": str(feature_path.resolve()),
        "relation_dataset_dir": str(relation_dataset_dir.resolve()),
        "none_shard_path": str(none_shard_path.resolve()),
        "alignment_gate_passed": alignment_gate,
        "training_scalar_rows": len(training_position),
        "decoder": {
            "model": str(model_path.resolve()),
            "threshold_count": len(decoder.X_thresholds_),
            "minimum_visual_x": float(decoder.X_min_),
            "maximum_visual_x": float(decoder.X_max_),
        },
        "audit_regression": {
            "position": position_metrics,
            "next_velocity": velocity_metrics,
            "gate_passed": regression_gate,
        },
        "balanced_audit": balanced,
        "predicted_coverage_gate_passed": predicted_coverage_gate,
        "classification_gate_passed": classification_gate,
        "balanced_gate_passed": bool(regression_gate and classification_gate),
        "passed": bool(regression_gate and classification_gate),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["metrics"] = str(metrics_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("relation_dataset_dir", type=Path)
    parser.add_argument("none_shard", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_id = f"isotonic_state_decoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/mountaincar_continuous"
    ) / run_id
    output = run_balanced_audit(
        args.features,
        args.relation_dataset_dir,
        args.none_shard,
        output_dir,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
