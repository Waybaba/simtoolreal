"""Fit and audit the frozen MountainCar visual decision tree."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier, export_text

from skill_discovery.evaluate_mountaincar_explicit_none import (
    CLASS_NAMES,
    five_class_metrics,
)


FEATURE_NAMES = ("x_before", "x_after", "delta_x")


@dataclass(frozen=True)
class VisualDecisionTreeConfig:
    criterion: str = "entropy"
    splitter: str = "best"
    max_depth: int = 8
    min_samples_split: int = 2
    min_samples_leaf: int = 8
    random_state: int = 10_100_007

    def __post_init__(self) -> None:
        if self.max_depth <= 0:
            raise ValueError("tree depth must be positive")
        if self.min_samples_split < 2 or self.min_samples_leaf <= 0:
            raise ValueError("tree sample budgets are invalid")


def build_classifier(config: VisualDecisionTreeConfig) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        criterion=config.criterion,
        splitter=config.splitter,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        max_features=None,
        class_weight=None,
        ccp_alpha=0.0,
        random_state=config.random_state,
    )


def run_balanced_audit(
    config: VisualDecisionTreeConfig,
    feature_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(feature_path) as data:
        features = data["car_features"].astype(np.float32)
        classes = data["classes"].astype(np.int8)
        splits = data["splits"].astype(np.int8)
        pair_hashes = data["pair_hashes"]
    reference = splits == 0
    audit = splits == 1
    counts = {
        split_name: {
            name: int(np.sum((classes == class_index - 1) & (splits == split)))
            for class_index, name in enumerate(CLASS_NAMES)
        }
        for split, split_name in enumerate(("reference", "audit"))
    }
    reference_hashes = set(str(value) for value in pair_hashes[reference])
    audit_hashes = set(str(value) for value in pair_hashes[audit])
    data_gate = bool(
        features.shape == (2_560, 3)
        and np.isfinite(features).all()
        and all(count == 256 for split in counts.values() for count in split.values())
        and len(reference_hashes) == 1_280
        and len(audit_hashes) == 1_280
        and not reference_hashes.intersection(audit_hashes)
    )
    if not data_gate:
        raise RuntimeError("frozen five-class feature data failed integrity checks")
    classifier = build_classifier(config)
    classifier.fit(features[reference], classes[reference])
    predictions = classifier.predict(features[audit]).astype(np.int8)
    metrics = five_class_metrics(classes[audit], predictions)
    predicted_coverage_gate = all(
        metrics["predicted_class_sizes"][name] > 0 for name in CLASS_NAMES
    )
    balanced_gate = bool(
        metrics["accuracy"] >= 0.98
        and metrics["macro_recall"] >= 0.98
        and min(metrics["recall_by_class"].values()) >= 0.95
        and predicted_coverage_gate
    )
    model_path = output_dir / "visual_decision_tree.joblib"
    joblib.dump(classifier, model_path)
    rules = export_text(
        classifier,
        feature_names=list(FEATURE_NAMES),
        decimals=8,
    )
    rules_path = output_dir / "visual_decision_tree.txt"
    rules_path.write_text(rules, encoding="utf-8")
    output = {
        "config": asdict(config),
        "sklearn_version": sklearn.__version__,
        "feature_path": str(feature_path.resolve()),
        "feature_names": list(FEATURE_NAMES),
        "class_split_counts": counts,
        "data_gate_passed": data_gate,
        "tree": {
            "model": str(model_path.resolve()),
            "rules": str(rules_path.resolve()),
            "node_count": int(classifier.tree_.node_count),
            "actual_depth": int(classifier.tree_.max_depth),
            "feature_importances": {
                name: float(value)
                for name, value in zip(
                    FEATURE_NAMES,
                    classifier.feature_importances_,
                    strict=True,
                )
            },
        },
        "balanced_audit": metrics,
        "predicted_coverage_gate_passed": predicted_coverage_gate,
        "balanced_gate_passed": balanced_gate,
        "passed": balanced_gate,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["metrics"] = str(metrics_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = VisualDecisionTreeConfig()
    run_id = f"visual_decision_tree_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/mountaincar_continuous"
    ) / run_id
    output = run_balanced_audit(config, args.features, output_dir)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
