"""Fit and audit the frozen MountainCar visual position decoder."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from skill_discovery.evaluate_mountaincar_explicit_none import (
    CLASS_NAMES,
    five_class_metrics,
)


def build_decoder() -> Pipeline:
    return Pipeline(
        (
            (
                "polynomial",
                PolynomialFeatures(degree=3, include_bias=False),
            ),
            (
                "ridge",
                Ridge(alpha=1.0e-8, fit_intercept=True, solver="svd"),
            ),
        )
    )


def decoded_relations(
    next_position: np.ndarray,
    next_velocity: np.ndarray,
) -> np.ndarray:
    position = np.asarray(next_position, dtype=np.float64)
    velocity = np.asarray(next_velocity, dtype=np.float64)
    predictions = np.full(len(position), -1, dtype=np.int8)
    predictions[
        (-1.15 <= position) & (position <= -0.75) & (velocity <= -0.005)
    ] = 0
    predictions[
        (-0.75 < position) & (position < 0.0) & (velocity >= 0.005)
    ] = 1
    predictions[
        (0.0 <= position) & (position < 0.45) & (velocity >= 0.005)
    ] = 2
    predictions[(position >= 0.45) & (velocity >= 0.0)] = 3
    return predictions


def _load_targets(
    relation_dataset_dir: Path,
    none_shard_path: Path,
) -> dict[str, np.ndarray]:
    shard_paths = sorted(relation_dataset_dir.glob("class_*.npz"))
    if len(shard_paths) != 4:
        raise ValueError("expected four relation shards")
    rows = []
    for shard_path in (*shard_paths, none_shard_path):
        with np.load(shard_path) as shard:
            rows.append(
                {
                    "states": shard["states"].astype(np.float32),
                    "next_states": shard["next_states"].astype(np.float32),
                    "pair_hashes": shard["pair_hashes"],
                    "classes": shard["classes"].astype(np.int8),
                    "splits": shard["splits"].astype(np.int8),
                }
            )
    return {
        name: np.concatenate([row[name] for row in rows], axis=0)
        for name in rows[0]
    }


def _error_summary(errors: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.mean(errors)),
        "p99_absolute_error": float(np.quantile(errors, 0.99)),
        "maximum_absolute_error": float(np.max(errors)),
    }


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
    decoder = build_decoder()
    training_visual_x = np.concatenate(
        (features[reference, 0], features[reference, 1])
    ).reshape(-1, 1)
    training_position = np.concatenate(
        (
            targets["states"][reference, 0],
            targets["next_states"][reference, 0],
        )
    )
    decoder.fit(training_visual_x, training_position)
    predicted_before = decoder.predict(features[:, 0].reshape(-1, 1))
    predicted_after = decoder.predict(features[:, 1].reshape(-1, 1))
    predicted_velocity = predicted_after - predicted_before
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
    model_path = output_dir / "visual_position_decoder.joblib"
    joblib.dump(decoder, model_path)
    ridge = decoder.named_steps["ridge"]
    output = {
        "sklearn_version": sklearn.__version__,
        "model_config": {
            "polynomial_degree": 3,
            "include_bias": False,
            "ridge_alpha": 1.0e-8,
            "ridge_fit_intercept": True,
            "ridge_solver": "svd",
        },
        "feature_path": str(feature_path.resolve()),
        "relation_dataset_dir": str(relation_dataset_dir.resolve()),
        "none_shard_path": str(none_shard_path.resolve()),
        "alignment_gate_passed": alignment_gate,
        "training_scalar_rows": len(training_position),
        "decoder": {
            "model": str(model_path.resolve()),
            "coefficients": [float(value) for value in ridge.coef_],
            "intercept": float(ridge.intercept_),
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
    run_id = f"visual_state_decoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
