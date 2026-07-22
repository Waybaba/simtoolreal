"""Tests for the MountainCarContinuous environment audit."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.audit_mountaincar_continuous_environment import (
    MountainCarEnvironmentAuditConfig,
    compare_random_tapes,
    expected_transition,
    ordered_stage,
)


class MountainCarEnvironmentAuditTest(unittest.TestCase):
    def test_protocol_horizons_are_frozen(self) -> None:
        with self.assertRaises(ValueError):
            MountainCarEnvironmentAuditConfig(random_horizon=255)
        with self.assertRaises(ValueError):
            MountainCarEnvironmentAuditConfig(scripted_horizon=998)

    def test_expected_transition_matches_frozen_equations(self) -> None:
        state = np.asarray([-0.5, 0.0], dtype=np.float32)
        action = np.asarray([-1.0], dtype=np.float32)
        next_state, reward, terminated = expected_transition(state, action)
        expected_velocity = -0.0015 - 0.0025 * np.cos(-1.5)
        self.assertAlmostEqual(float(next_state[1]), expected_velocity, places=7)
        self.assertAlmostEqual(float(next_state[0]), -0.5 + expected_velocity, places=7)
        self.assertEqual(reward, -0.1)
        self.assertFalse(terminated)

    def test_ordered_stages_cannot_skip(self) -> None:
        self.assertEqual(ordered_stage(0, 0.1, False), 0)
        self.assertEqual(ordered_stage(0, -0.8, False), 1)
        self.assertEqual(ordered_stage(1, 0.1, False), 2)
        self.assertEqual(ordered_stage(2, 0.46, True), 3)

    def test_comparison_requires_exact_rgb_hashes(self) -> None:
        row = {
            "observation": [0.0, 0.0],
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
        }
        rollout = {
            "reset_observation": [0.0, 0.0],
            "reset_frame": {"hash": "reset"},
            "frame_records": [{"hash": "reset"}, {"hash": "step"}],
            "rows": [row],
        }
        self.assertTrue(compare_random_tapes(rollout, rollout)["passed"])
        changed = dict(rollout)
        changed["frame_records"] = [{"hash": "reset"}, {"hash": "changed"}]
        comparison = compare_random_tapes(rollout, changed)
        self.assertTrue(comparison["states_exact_equal"])
        self.assertFalse(comparison["frame_hashes_exact_equal"])
        self.assertFalse(comparison["passed"])


if __name__ == "__main__":
    unittest.main()
