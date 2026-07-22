"""Tests for GoToObject blockwise multi-seed summaries."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from skill_discovery.summarize_gotoobject_blockwise import summarize_runs


class BlockwiseSummaryTest(unittest.TestCase):
    def _write_run(self, root: Path, seed: int, *, bootstrap: bool) -> Path:
        run_dir = root / f"seed{seed}"
        run_dir.mkdir()
        final = {
            "matched_rate_by_stage": {
                "object_far": 0.95,
                "object_adjacent": 0.96,
                "object_carried": 0.94,
            },
            "specialization_gate_passed": True,
        }
        metrics = {
            "config": {"seed": seed},
            "bootstrap_gate_passed": bootstrap,
            "bootstrap_top_stages": [0, 2, 1] if bootstrap else [2, 2, 1],
            "policy_phase_ran": bootstrap,
            "final_evaluation": final if bootstrap else None,
            "checkpoint_stability_passed": bootstrap,
            "signal_gate_passed": bootstrap,
        }
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics),
            encoding="utf-8",
        )
        return run_dir

    def test_summary_preserves_failure_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = [
                self._write_run(root, 7, bootstrap=True),
                self._write_run(root, 29, bootstrap=False),
            ]
            output = summarize_runs(runs, root / "summary.json")
            self.assertEqual(output["seeds_passed"], 1)
            self.assertFalse(output["multi_seed_gate_passed"])
            self.assertEqual(
                output["runs"][1]["failure_reason"],
                "bootstrap_collision",
            )


if __name__ == "__main__":
    unittest.main()
