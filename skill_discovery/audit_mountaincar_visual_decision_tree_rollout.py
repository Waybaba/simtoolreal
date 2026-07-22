"""Audit the frozen MountainCar visual decision tree on natural rollouts."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from skill_discovery.audit_mountaincar_frame_pair_capacity import PAIR_CLASSES
from skill_discovery.audit_mountaincar_rollout_rejection import (
    RolloutRejectionConfig,
    _select_examples,
    _write_contact_sheet,
    rejection_metrics,
    rollout_worker,
)
from skill_discovery.evaluate_mountaincar_explicit_none import CLASS_NAMES


def natural_gate_results(metrics: dict[str, object]) -> dict[str, bool]:
    coverage = bool(
        min(metrics["oracle_counts"][name] for name in PAIR_CLASSES) >= 100
        and metrics["oracle_counts"]["none"] >= 10_000
    )
    relations = bool(
        metrics["relation_macro_recall"] >= 0.95
        and min(metrics["relation_recall"].values()) >= 0.90
        and metrics["relation_recall"]["native_goal"] >= 0.95
    )
    none = bool(
        metrics["none_false_positive_rate"] <= 0.10
        and metrics["none_to_goal_rate"] <= 0.01
    )
    predicted_coverage = all(
        metrics["predicted_counts"][name] > 0 for name in CLASS_NAMES
    )
    return {
        "coverage_gate_passed": coverage,
        "relation_recall_gate_passed": relations,
        "none_gate_passed": none,
        "predicted_coverage_gate_passed": predicted_coverage,
        "numeric_gate_passed": bool(
            coverage and relations and none and predicted_coverage
        ),
    }


def run_natural_audit(
    config: RolloutRejectionConfig,
    model_path: Path,
    balanced_metrics_path: Path,
    background_path: Path,
    baseline_audit_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    balanced = json.loads(balanced_metrics_path.read_text(encoding="utf-8"))
    if not balanced["balanced_gate_passed"]:
        raise RuntimeError("balanced decision-tree gate did not pass")
    baseline = json.loads(baseline_audit_path.read_text(encoding="utf-8"))
    protocol_gate = baseline["config"] == asdict(config)
    if not protocol_gate:
        raise RuntimeError("rollout config does not match Phase 5ZW")
    classifier = joblib.load(model_path)
    background = np.asarray(Image.open(background_path).convert("RGB"), dtype=np.uint8)
    output_dir.mkdir(parents=True, exist_ok=False)
    unused_features = np.empty((0, 3), dtype=np.float32)
    unused_classes = np.empty(0, dtype=np.int8)
    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        worker_rows = list(
            pool.map(
                rollout_worker,
                [config] * config.workers,
                range(config.workers),
                [background] * config.workers,
                [unused_features] * config.workers,
                [unused_classes] * config.workers,
                [None] * config.workers,
                [classifier] * config.workers,
            )
        )
    predictions = np.concatenate([row["predictions"] for row in worker_rows])
    confidence = np.concatenate([row["distances"] for row in worker_rows])
    oracle = np.concatenate([row["oracle"] for row in worker_rows])
    sources = np.concatenate([row["sources"] for row in worker_rows])
    combined = rejection_metrics(oracle, predictions)
    energy = rejection_metrics(oracle[sources == 0], predictions[sources == 0])
    random = rejection_metrics(oracle[sources == 1], predictions[sources == 1])
    gates = natural_gate_results(combined)
    examples = _select_examples(worker_rows)
    contact_sheet_path = output_dir / "visual_tree_rollout_contact_sheet.png"
    _write_contact_sheet(contact_sheet_path, examples)
    public_examples = [
        {key: value for key, value in example.items() if key != "frames"}
        for example in examples
    ]
    output = {
        "config": asdict(config),
        "paired_protocol_gate_passed": protocol_gate,
        "model_path": str(model_path.resolve()),
        "balanced_metrics_path": str(balanced_metrics_path.resolve()),
        "background_path": str(background_path.resolve()),
        "baseline_audit_path": str(baseline_audit_path.resolve()),
        "combined": combined,
        "energy": energy,
        "random": random,
        "confidence_summary": {
            "minimum": float(confidence.min()),
            "median": float(np.median(confidence)),
            "maximum": float(confidence.max()),
        },
        **gates,
        "contact_sheet": str(contact_sheet_path.resolve()),
        "contact_examples": public_examples,
        "manual_contact_sheet_gate": "pending",
        "passed": False,
    }
    metrics_path = output_dir / "audit.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["metrics"] = str(metrics_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("balanced_metrics", type=Path)
    parser.add_argument("background", type=Path)
    parser.add_argument("baseline_audit", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = RolloutRejectionConfig()
    run_id = f"visual_tree_rollout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/mountaincar_continuous"
    ) / run_id
    output = run_natural_audit(
        config,
        args.model,
        args.balanced_metrics,
        args.background,
        args.baseline_audit,
        output_dir,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
