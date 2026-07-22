"""Tests for MountainCar visual decision-tree natural gates."""

from __future__ import annotations

import unittest

from skill_discovery.audit_mountaincar_visual_decision_tree_rollout import (
    natural_gate_results,
)


class MountainCarVisualDecisionTreeRolloutTest(unittest.TestCase):
    def test_natural_gate_results_require_all_five_classes(self) -> None:
        metrics = {
            "oracle_counts": {
                "none": 10_000,
                "left_momentum": 100,
                "valley_return": 100,
                "right_climb": 100,
                "native_goal": 100,
            },
            "predicted_counts": {
                "none": 10_000,
                "left_momentum": 100,
                "valley_return": 100,
                "right_climb": 100,
                "native_goal": 100,
            },
            "relation_macro_recall": 0.96,
            "relation_recall": {
                "left_momentum": 0.95,
                "valley_return": 0.95,
                "right_climb": 0.95,
                "native_goal": 0.99,
            },
            "none_false_positive_rate": 0.08,
            "none_to_goal_rate": 0.0,
        }
        self.assertTrue(natural_gate_results(metrics)["numeric_gate_passed"])
        metrics["predicted_counts"]["none"] = 0
        self.assertFalse(natural_gate_results(metrics)["numeric_gate_passed"])


if __name__ == "__main__":
    unittest.main()
