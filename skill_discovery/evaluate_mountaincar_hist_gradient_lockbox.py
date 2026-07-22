"""Fit the frozen development model and evaluate the fresh lockbox once."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from skill_discovery.evaluate_mountaincar_explicit_none import (
    CLASS_NAMES,
    five_class_metrics,
)


def build_classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=15,
        max_depth=6,
        min_samples_leaf=12,
        l2_regularization=0.1,
        early_stopping=False,
        class_weight=None,
        random_state=18_100_007,
    )


def _class_counts(classes: np.ndarray) -> dict[str, int]:
    return {
        name: int(np.sum(classes == class_index - 1))
        for class_index, name in enumerate(CLASS_NAMES)
    }


def run_lockbox_evaluation(
    development_feature_path: Path,
    lockbox_feature_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(development_feature_path) as development:
        development_features = development["car_features"].astype(np.float32)
        development_classes = development["classes"].astype(np.int8)
        development_hashes = development["pair_hashes"]
    with np.load(lockbox_feature_path) as lockbox:
        lockbox_features = lockbox["car_features"].astype(np.float32)
        lockbox_classes = lockbox["classes"].astype(np.int8)
        lockbox_hashes = lockbox["pair_hashes"]
    development_counts = _class_counts(development_classes)
    lockbox_counts = _class_counts(lockbox_classes)
    development_hash_set = set(str(value) for value in development_hashes)
    lockbox_hash_set = set(str(value) for value in lockbox_hashes)
    data_gate = bool(
        development_features.shape == (2_560, 3)
        and lockbox_features.shape == (1_280, 3)
        and all(count == 512 for count in development_counts.values())
        and all(count == 256 for count in lockbox_counts.values())
        and len(development_hash_set) == 2_560
        and len(lockbox_hash_set) == 1_280
        and not development_hash_set.intersection(lockbox_hash_set)
        and np.isfinite(development_features).all()
        and np.isfinite(lockbox_features).all()
    )
    if not data_gate:
        raise RuntimeError("development/lockbox data failed frozen integrity checks")
    classifier = build_classifier()
    classifier.fit(development_features, development_classes)
    probabilities = classifier.predict_proba(lockbox_features)
    predictions = classifier.classes_[np.argmax(probabilities, axis=1)].astype(np.int8)
    confidence = np.max(probabilities, axis=1)
    metrics = five_class_metrics(lockbox_classes, predictions)
    predicted_coverage_gate = all(
        metrics["predicted_class_sizes"][name] > 0 for name in CLASS_NAMES
    )
    lockbox_gate = bool(
        metrics["accuracy"] >= 0.98
        and metrics["macro_recall"] >= 0.98
        and min(metrics["recall_by_class"].values()) >= 0.95
        and predicted_coverage_gate
    )
    model_path = output_dir / "hist_gradient_lockbox_model.joblib"
    joblib.dump(classifier, model_path)
    output = {
        "sklearn_version": sklearn.__version__,
        "model_config": {
            "loss": "log_loss",
            "learning_rate": 0.08,
            "max_iter": 200,
            "max_leaf_nodes": 15,
            "max_depth": 6,
            "min_samples_leaf": 12,
            "l2_regularization": 0.1,
            "early_stopping": False,
            "class_weight": None,
            "random_state": 18_100_007,
        },
        "development_feature_path": str(development_feature_path.resolve()),
        "lockbox_feature_path": str(lockbox_feature_path.resolve()),
        "development_counts": development_counts,
        "lockbox_counts": lockbox_counts,
        "development_unique_pair_hashes": len(development_hash_set),
        "lockbox_unique_pair_hashes": len(lockbox_hash_set),
        "cross_dataset_pair_hash_overlap": len(
            development_hash_set.intersection(lockbox_hash_set)
        ),
        "data_gate_passed": data_gate,
        "model": str(model_path.resolve()),
        "iterations": int(classifier.n_iter_),
        "lockbox_audit": metrics,
        "confidence_summary": {
            "minimum": float(confidence.min()),
            "median": float(np.median(confidence)),
            "maximum": float(confidence.max()),
        },
        "predicted_coverage_gate_passed": predicted_coverage_gate,
        "lockbox_gate_passed": lockbox_gate,
        "passed": lockbox_gate,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["metrics"] = str(metrics_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development_features", type=Path)
    parser.add_argument("lockbox_features", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_id = f"hist_gradient_lockbox_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/mountaincar_continuous"
    ) / run_id
    output = run_lockbox_evaluation(
        args.development_features,
        args.lockbox_features,
        output_dir,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
