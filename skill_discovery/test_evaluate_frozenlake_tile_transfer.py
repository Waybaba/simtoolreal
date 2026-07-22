"""Tests for scale-aware FrozenLake tile transfer gates."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.evaluate_frozenlake_tile_transfer import (
    style_transfer_metrics,
)


class FrozenLakeTileTransferTest(unittest.TestCase):
    def test_style_gate_requires_goal_and_per_style_coverage(self) -> None:
        outcomes = np.tile(np.repeat(np.arange(3), 4), 16)
        seeds = np.repeat(np.arange(16), 12)
        perfect = style_transfer_metrics(outcomes.copy(), outcomes, seeds)
        self.assertTrue(perfect["gate_passed"])
        self.assertEqual(perfect["styles_passing_0_80"], 16)

        predictions = outcomes.copy()
        predictions[outcomes == 2] = 1
        missing_goal = style_transfer_metrics(predictions, outcomes, seeds)
        self.assertFalse(missing_goal["gate_passed"])
        self.assertEqual(
            missing_goal["recall_by_outcome"]["goal_terminal"],
            0.0,
        )

    def test_style_gate_rejects_five_bad_styles(self) -> None:
        outcomes = np.tile(np.repeat(np.arange(3), 4), 16)
        seeds = np.repeat(np.arange(16), 12)
        predictions = outcomes.copy()
        predictions[seeds < 5] = 0
        metrics = style_transfer_metrics(predictions, outcomes, seeds)
        self.assertEqual(metrics["styles_passing_0_80"], 11)
        self.assertFalse(metrics["gate_passed"])


if __name__ == "__main__":
    unittest.main()
