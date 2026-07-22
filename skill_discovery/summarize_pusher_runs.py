"""Aggregate repeated Pusher-Cup training runs by objective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from skill_discovery.pusher_cup import TRAJECTORY_CLASSES


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
        metrics_by_name: dict[str, list[float]] = {
            "matched_class_rate_mean": [],
            "matched_class_rate_min": [],
            "semantic_mi_bits": [],
            "terminal_graph_mi_bits": [],
            "pusher_terminal_coverage": [],
            "ball_terminal_coverage": [],
            "inside_rate_high": [],
            "inside_skill_ball_path_mean": [],
            "last20_specialization_fraction": [],
        }
        class_rates = {name: [] for name in TRAJECTORY_CLASSES}
        passed = []
        for _, payload in entries:
            evaluation = payload["evaluation"]
            assignment = [int(value) for value in evaluation["class_assignment"]]
            matched = [float(value) for value in evaluation["matched_class_rates"]]
            by_class = np.zeros(len(TRAJECTORY_CLASSES), dtype=np.float64)
            for skill, class_id in enumerate(assignment):
                by_class[class_id] = matched[skill]
            inside_skill = assignment.index(2)
            for class_id, class_name in enumerate(TRAJECTORY_CLASSES):
                class_rates[class_name].append(float(by_class[class_id]))
            metrics_by_name["matched_class_rate_mean"].append(float(np.mean(matched)))
            metrics_by_name["matched_class_rate_min"].append(float(np.min(matched)))
            metrics_by_name["semantic_mi_bits"].append(float(evaluation["semantic_mi_bits"]))
            metrics_by_name["terminal_graph_mi_bits"].append(
                float(evaluation["terminal_graph_mi_bits"])
            )
            metrics_by_name["pusher_terminal_coverage"].append(
                float(evaluation["pusher_terminal_coverage"])
            )
            metrics_by_name["ball_terminal_coverage"].append(
                float(evaluation["ball_terminal_coverage"])
            )
            metrics_by_name["inside_rate_high"].append(
                float(max(evaluation["inside_rates"]))
            )
            metrics_by_name["inside_skill_ball_path_mean"].append(
                float(evaluation["ball_path_length_means"][inside_skill])
            )
            metrics_by_name["last20_specialization_fraction"].append(
                float(payload["training_stability"]["specialization_gate_fraction"])
            )
            passed.append(
                bool(
                    evaluation["specialization_gate_passed"]
                    and payload["training_stability"]["passed"]
                )
            )
        summary[representation] = {
            "run_count": len(entries),
            "paths": [str(path.resolve()) for path, _ in entries],
            **{name: _stats(values) for name, values in metrics_by_name.items()},
            "matched_rate_by_class": {
                class_name: _stats(values) for class_name, values in class_rates.items()
            },
            "passed_count": int(sum(passed)),
            "passed_values": passed,
        }

    required = ("random", "raw", "semantic", "semantic_balanced")
    balanced = summary.get("semantic_balanced")
    oracle_checks = {
        "all_four_objectives_present": all(name in summary for name in required),
        "five_runs_per_objective": all(
            summary.get(name, {}).get("run_count", 0) >= 5 for name in required
        ),
        "balanced_oracle_passes_at_least_four": balanced is not None
        and balanced["passed_count"] >= 4,
        "balanced_oracle_all_class_means_at_least_0p7": balanced is not None
        and all(
            balanced["matched_rate_by_class"][name]["mean"] >= 0.70
            for name in TRAJECTORY_CLASSES
        ),
        "balanced_inside_ball_path_at_least_0p4": balanced is not None
        and balanced["inside_skill_ball_path_mean"]["mean"] >= 0.40,
    }
    spread = summary.get("semantic_spread")
    method_checks = {
        "five_semantic_spread_runs": spread is not None and spread["run_count"] >= 5,
        "semantic_spread_passes_at_least_four": spread is not None
        and spread["passed_count"] >= 4,
        "semantic_spread_all_class_means_at_least_0p7": spread is not None
        and all(
            spread["matched_rate_by_class"][name]["mean"] >= 0.70
            for name in TRAJECTORY_CLASSES
        ),
        "semantic_spread_inside_ball_path_at_least_0p4": spread is not None
        and spread["inside_skill_ball_path_mean"]["mean"] >= 0.40,
        "visual_policy_audit_passed": False,
    }
    visual_audit = None
    if args.visual_audit is not None:
        visual_audit = json.loads(args.visual_audit.read_text(encoding="utf-8"))
        method_checks["visual_policy_audit_passed"] = bool(visual_audit.get("passed"))
    method_gate_passed = all(method_checks.values())
    output = {
        "summary": summary,
        "oracle_gate_checks": oracle_checks,
        "oracle_diagnostic_gate_passed": all(oracle_checks.values()),
        "method_gate_checks": method_checks,
        "phase_3_method_gate_passed": method_gate_passed,
        "phase_3_method_gate_reason": None
        if method_gate_passed
        else "semantic spread multi-seed or visual gate is incomplete",
        "visual_audit_pending": args.visual_audit is None,
        "visual_audit": visual_audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
