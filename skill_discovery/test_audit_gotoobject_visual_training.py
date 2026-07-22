"""Tests for paired visual/semantic GoToObject training audits."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from skill_discovery.audit_gotoobject_visual_training import compare_q_tables
from skill_discovery.summarize_gotoobject_visual_training import (
    summarize_paired_audits,
)


class GoToObjectVisualTrainingAuditTest(unittest.TestCase):
    def test_q_table_comparison_requires_keys_values_and_visits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            keys = np.asarray([[1, 2, 0, 3, 3, -1, -1, 1]], dtype=np.int16)
            values = np.ones((1, 3, 5), dtype=np.float64)
            visits = np.ones((1, 3, 5), dtype=np.float64)
            np.savez_compressed(root / "left.npz", relation_keys=keys, q_values=values, visits=visits)
            np.savez_compressed(root / "right.npz", relation_keys=keys, q_values=values, visits=visits)
            equal = compare_q_tables(root / "left.npz", root / "right.npz")
            np.savez_compressed(
                root / "changed.npz",
                relation_keys=keys,
                q_values=values + 0.1,
                visits=visits,
            )
            changed = compare_q_tables(root / "left.npz", root / "changed.npz")
        self.assertTrue(equal["exact_equal"])
        self.assertEqual(equal["max_q_abs_difference"], 0.0)
        self.assertFalse(changed["exact_equal"])
        self.assertAlmostEqual(changed["max_q_abs_difference"], 0.1)

    def test_summary_separates_representation_and_control_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for seed, control_passed in ((7, True), (17, False)):
                path = root / f"seed_{seed}.json"
                path.write_text(
                    json.dumps(
                        {
                            "seed": seed,
                            "stage_observer": {
                                "query_count": 100,
                                "parser_calls": 20,
                                "mismatch_count": 0,
                            },
                            "q_table_comparison": {"exact_equal": True},
                            "evaluations_exact_equal": True,
                            "final_evaluation": {
                                "matched_rate_by_stage": {
                                    "object_far": 0.98,
                                    "object_adjacent": 0.96,
                                    "object_carried": 0.93,
                                }
                            },
                            "visual_signal_gate_passed": control_passed,
                            "representation_equivalence_gate_passed": True,
                        }
                    )
                )
                paths.append(path)
            output = summarize_paired_audits(paths, root / "summary.json")
        self.assertTrue(output["representation_equivalence_gate_passed"])
        self.assertEqual(output["representation_equivalence_pass_count"], 2)
        self.assertFalse(output["temporal_control_gate_passed"])
        self.assertEqual(output["temporal_control_pass_count"], 1)


if __name__ == "__main__":
    unittest.main()
