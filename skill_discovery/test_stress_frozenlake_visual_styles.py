"""Tests for the multi-style FrozenLake visual stress audit."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.stress_frozenlake_visual_styles import (
    build_styled_trajectories,
    evaluate_predictions,
    run_style_stress,
    select_balanced_sources,
)


class FrozenLakeStyleStressTest(unittest.TestCase):
    def test_balanced_sources_are_deterministic_and_audit_only(self) -> None:
        outcomes = np.tile(np.arange(3), 20)
        split = np.tile([0, 1], 30)
        first = select_balanced_sources(
            outcomes,
            split,
            count_per_outcome=5,
            seed=7,
        )
        second = select_balanced_sources(
            outcomes,
            split,
            count_per_outcome=5,
            seed=7,
        )
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(split[first] == 1))
        self.assertEqual(np.bincount(outcomes[first], minlength=3).tolist(), [5] * 3)

    def test_style_batches_are_deterministic_and_distinct(self) -> None:
        frames = np.full((2, 3, 8, 8, 3), 100, dtype=np.uint8)
        first = build_styled_trajectories(frames, (107, 117))
        second = build_styled_trajectories(frames, (107, 117))
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first[0], first[1]))

    def test_prediction_gate_requires_aggregate_and_style_coverage(self) -> None:
        outcomes = np.tile(np.arange(3), 4)
        predictions = np.tile(outcomes, (16, 1))
        passing = evaluate_predictions(predictions, outcomes, tuple(range(16)))
        self.assertTrue(passing["gate_passed"])
        predictions[:3] = 2
        failing = evaluate_predictions(predictions, outcomes, tuple(range(16)))
        self.assertFalse(failing["gate_passed"])
        self.assertEqual(failing["styles_passing_0_84"], 13)

    def test_run_rejects_nonstandard_style_seed_count_before_io(self) -> None:
        with self.assertRaises(ValueError):
            run_style_stress(
                None,
                None,
                None,
                None,
                style_seeds=(1, 2),
            )


if __name__ == "__main__":
    unittest.main()
