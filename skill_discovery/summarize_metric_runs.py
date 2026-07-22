"""Aggregate repeated Point-Cup metric probes and evaluate the Phase 1 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SUMMARY_FIELDS = (
    "representative_class_coverage",
    "representative_rare_recall",
    "representative_semantic_entropy",
    "representative_geometric_coverage",
    "inter_intra_distance_ratio",
    "cross_layout_knn_accuracy",
    "nuisance_triplet_accuracy",
    "nuisance_triplet_margin",
)


def _stats(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "values": [float(value) for value in array],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manual-audit", type=Path, default=None)
    args = parser.parse_args()

    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.metrics]
    methods = list(runs[0]["metrics"])
    summary = {
        method: {
            field: _stats([float(run["metrics"][method][field]) for run in runs])
            for field in SUMMARY_FIELDS
        }
        for method in methods
    }

    per_run_gate = []
    for path, run in zip(args.metrics, runs):
        raw = run["metrics"]["raw_l2"]
        oracle = run["metrics"]["semantic_oracle"]
        checks = {
            "audit_labels_match": run["audit_observed_matches_intent"] == 1.0,
            "nuisance_labels_match": run["nuisance_observed_matches_intent"] == 1.0,
            "oracle_rare_recall_better": oracle["representative_rare_recall"] > raw["representative_rare_recall"],
            "oracle_semantic_entropy_better": oracle["representative_semantic_entropy"] > raw["representative_semantic_entropy"],
            "oracle_geometric_coverage_retained": oracle["representative_geometric_coverage"]
            >= 0.7 * raw["representative_geometric_coverage"],
            "oracle_cross_layout_better": oracle["cross_layout_knn_accuracy"] > raw["cross_layout_knn_accuracy"],
            "oracle_nuisance_triplet_better": oracle["nuisance_triplet_accuracy"] > raw["nuisance_triplet_accuracy"],
        }
        per_run_gate.append(
            {
                "metrics_path": str(path.resolve()),
                "passed": all(checks.values()),
                "checks": checks,
            }
        )

    manual_audit = None
    if args.manual_audit is not None:
        manual_payload = json.loads(args.manual_audit.read_text(encoding="utf-8"))
        manual_audit = {
            "path": str(args.manual_audit.resolve()),
            "row_count": int(manual_payload["row_count"]),
            "all_labels_match": bool(manual_payload["all_labels_match"]),
            "passed": bool(manual_payload["row_count"] >= 30 and manual_payload["all_labels_match"]),
        }

    numeric_gate_passed = all(run["passed"] for run in per_run_gate)
    output = {
        "run_count": len(runs),
        "metrics_paths": [str(path.resolve()) for path in args.metrics],
        "summary": summary,
        "per_run_gate": per_run_gate,
        "phase_1_numeric_gate_passed": numeric_gate_passed,
        "manual_audit": manual_audit,
        "phase_1_gate_passed": bool(numeric_gate_passed and manual_audit is not None and manual_audit["passed"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
