"""Compare a visual-stage GoToObject run with its paired semantic source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare_q_tables(left_path: Path, right_path: Path) -> dict[str, object]:
    with np.load(left_path) as left, np.load(right_path) as right:
        keys_equal = np.array_equal(left["relation_keys"], right["relation_keys"])
        q_values_equal = np.array_equal(left["q_values"], right["q_values"])
        visits_equal = np.array_equal(left["visits"], right["visits"])
        max_q_abs_difference = (
            float(np.max(np.abs(left["q_values"] - right["q_values"])))
            if left["q_values"].shape == right["q_values"].shape
            else None
        )
    return {
        "relation_keys_exact_equal": bool(keys_equal),
        "q_values_exact_equal": bool(q_values_equal),
        "visits_exact_equal": bool(visits_equal),
        "max_q_abs_difference": max_q_abs_difference,
        "exact_equal": bool(keys_equal and q_values_equal and visits_equal),
    }


def paired_training_audit(
    visual_run_dir: Path,
    semantic_run_dir: Path,
    output_path: Path,
) -> dict[str, object]:
    visual = json.loads((visual_run_dir / "metrics.json").read_text(encoding="utf-8"))
    semantic = json.loads((semantic_run_dir / "metrics.json").read_text(encoding="utf-8"))
    visual_config = dict(visual["config"])
    semantic_config = dict(semantic["config"])
    visual_stage_source = visual_config.pop("stage_source", "oracle_state")
    semantic_stage_source = semantic_config.pop("stage_source", "oracle_state")
    config_exact_except_stage_source = visual_config == semantic_config
    q_comparison = compare_q_tables(
        visual_run_dir / "q_table.npz",
        semantic_run_dir / "q_table.npz",
    )
    observer = visual["stage_observer"]
    recent = visual.get("evaluations", [])[-3:]
    expected_recent = [13_000, 14_000, 15_000]
    recent_gate = bool(
        [row["policy_episodes"] for row in recent] == expected_recent
        and all(row["specialization_gate_passed"] for row in recent)
    )
    bootstrap_gate = bool(
        visual["bootstrap_gate_passed"]
        and visual["bootstrap_assigned_stages"] == [0, 2, 1]
    )
    paired_gate = bool(
        visual_stage_source == "rgb_template_object_graph"
        and semantic_stage_source == "oracle_state"
        and config_exact_except_stage_source
        and observer["query_count"] > 0
        and observer["mismatch_count"] == 0
        and observer["minimum_exact_tile_fraction"] == 1.0
        and bootstrap_gate
        and recent_gate
        and visual["signal_gate_passed"]
        and q_comparison["exact_equal"]
    )
    output = {
        "visual_run": str(visual_run_dir.resolve()),
        "semantic_run": str(semantic_run_dir.resolve()),
        "visual_stage_source": visual_stage_source,
        "semantic_stage_source": semantic_stage_source,
        "config_exact_except_stage_source": config_exact_except_stage_source,
        "stage_observer": observer,
        "bootstrap_gate_passed": bootstrap_gate,
        "recent_checkpoint_episodes": [row["policy_episodes"] for row in recent],
        "recent_checkpoint_gate_passed": recent_gate,
        "final_evaluation": visual.get("final_evaluation"),
        "visual_signal_gate_passed": visual["signal_gate_passed"],
        "q_table_comparison": q_comparison,
        "paired_gate_passed": paired_gate,
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("visual_run", type=Path)
    parser.add_argument("semantic_run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output_path = args.output or args.visual_run / "paired_semantic_audit.json"
    output = paired_training_audit(args.visual_run, args.semantic_run, output_path)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
