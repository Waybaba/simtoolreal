"""Summarize multi-seed GoToObject blockwise runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from skill_discovery.minigrid_gotoobject import GOTOOBJECT_STAGES


def _failure_reason(metrics: dict[str, object]) -> str | None:
    if not metrics["bootstrap_gate_passed"]:
        return "bootstrap_collision"
    final = metrics.get("final_evaluation")
    if not isinstance(final, dict) or not final["specialization_gate_passed"]:
        return "final_stage_gate"
    if not metrics["checkpoint_stability_passed"]:
        return "temporal_stability"
    return None


def summarize_runs(run_dirs: list[Path], output_path: Path) -> dict[str, object]:
    rows = []
    seen_seeds = set()
    for run_dir in run_dirs:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        config = metrics["config"]
        if not isinstance(config, dict):
            raise ValueError("run config must be a mapping")
        seed = int(config["seed"])
        if seed in seen_seeds:
            raise ValueError(f"duplicate training seed: {seed}")
        seen_seeds.add(seed)
        final = metrics.get("final_evaluation")
        rows.append(
            {
                "seed": seed,
                "run_dir": str(run_dir.resolve()),
                "bootstrap_gate_passed": metrics["bootstrap_gate_passed"],
                "bootstrap_top_stages": metrics["bootstrap_top_stages"],
                "policy_phase_ran": metrics["policy_phase_ran"],
                "final_matched_rate_by_stage": None
                if not isinstance(final, dict)
                else final["matched_rate_by_stage"],
                "final_gate_passed": False
                if not isinstance(final, dict)
                else final["specialization_gate_passed"],
                "checkpoint_stability_passed": metrics[
                    "checkpoint_stability_passed"
                ],
                "run_gate_passed": metrics["signal_gate_passed"],
                "failure_reason": _failure_reason(metrics),
            }
        )
    completed = [row for row in rows if row["final_matched_rate_by_stage"]]
    stage_summary = {}
    for stage in GOTOOBJECT_STAGES:
        values = np.asarray(
            [row["final_matched_rate_by_stage"][stage] for row in completed],
            dtype=np.float64,
        )
        stage_summary[stage] = None if not len(values) else {
            "completed_policy_seeds": len(values),
            "mean": float(values.mean()),
            "standard_deviation": float(values.std()),
            "worst": float(values.min()),
        }
    output = {
        "training_seeds": sorted(seen_seeds),
        "runs": sorted(rows, key=lambda row: row["seed"]),
        "seeds_passed": sum(bool(row["run_gate_passed"]) for row in rows),
        "seeds_total": len(rows),
        "multi_seed_gate_passed": all(
            bool(row["run_gate_passed"]) for row in rows
        ),
        "stage_summary": stage_summary,
    }
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = summarize_runs(args.run_dirs, args.output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
