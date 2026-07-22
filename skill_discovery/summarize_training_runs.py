"""Aggregate repeated Point-Cup tabular DIAYN runs by representation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _stats(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "values": array.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visual-audit", type=Path)
    args = parser.parse_args()

    grouped: dict[str, list[tuple[Path, dict]]] = {}
    for path in args.metrics:
        payload = json.loads(path.read_text(encoding="utf-8"))
        grouped.setdefault(payload["config"]["representation"], []).append((path, payload))

    summary = {}
    for representation, entries in sorted(grouped.items()):
        high_inside = []
        low_inside = []
        inside_gap = []
        semantic_mi = []
        endpoint_mi = []
        xy_coverage = []
        stability_fraction = []
        passed = []
        for _, payload in entries:
            evaluation = payload["evaluation"]
            rates = [float(value) for value in evaluation["inside_rates"]]
            high_inside.append(max(rates))
            low_inside.append(min(rates))
            inside_gap.append(float(evaluation["inside_rate_gap"]))
            semantic_mi.append(float(evaluation["semantic_mi_bits"]))
            endpoint_mi.append(float(evaluation["endpoint_grid_mi_bits"]))
            xy_coverage.append(float(evaluation["xy_coverage"]))
            stability_fraction.append(float(payload["training_stability"]["specialization_gate_fraction"]))
            passed.append(
                bool(evaluation["specialization_gate_passed"] and payload["training_stability"]["passed"])
            )
        summary[representation] = {
            "run_count": len(entries),
            "paths": [str(path.resolve()) for path, _ in entries],
            "inside_rate_high": _stats(high_inside),
            "inside_rate_low": _stats(low_inside),
            "inside_rate_gap": _stats(inside_gap),
            "semantic_mi_bits": _stats(semantic_mi),
            "endpoint_grid_mi_bits": _stats(endpoint_mi),
            "xy_coverage": _stats(xy_coverage),
            "last20_specialization_fraction": _stats(stability_fraction),
            "passed_count": int(sum(passed)),
            "passed_values": passed,
        }

    semantic = summary.get("semantic")
    raw = summary.get("raw")
    random = summary.get("random")
    checks = {
        "all_three_baselines_present": semantic is not None and raw is not None and random is not None,
        "five_semantic_seeds": semantic is not None and semantic["run_count"] >= 5,
        "semantic_passes_at_least_four": semantic is not None and semantic["passed_count"] >= 4,
        "semantic_high_inside_at_least_0p8": semantic is not None and semantic["inside_rate_high"]["mean"] >= 0.8,
        "semantic_low_inside_at_most_0p2": semantic is not None and semantic["inside_rate_low"]["mean"] <= 0.2,
        "semantic_mi_better_than_raw": semantic is not None and raw is not None and semantic["semantic_mi_bits"]["mean"] > raw["semantic_mi_bits"]["mean"],
        "semantic_mi_better_than_random": semantic is not None and random is not None and semantic["semantic_mi_bits"]["mean"] > random["semantic_mi_bits"]["mean"],
    }
    numeric_gate_passed = all(checks.values())
    visual_audit = None
    if args.visual_audit is not None:
        visual_audit = json.loads(args.visual_audit.read_text(encoding="utf-8"))
        checks["visual_policy_audit_passed"] = bool(visual_audit.get("passed"))
    output = {
        "summary": summary,
        "gate_checks": checks,
        "numeric_gate_passed": numeric_gate_passed,
        "phase_2_gate_passed": all(checks.values()),
        "visual_audit_pending": args.visual_audit is None,
        "visual_audit": visual_audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
