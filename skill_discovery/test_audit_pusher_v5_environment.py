"""Tests for the Pusher-v5 environment audit helpers."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.audit_pusher_v5_environment import (
    PusherV5EnvironmentAuditConfig,
    compare_rollouts,
    pixel_difference_summary,
    reward_terms,
)


class PusherV5EnvironmentAuditTest(unittest.TestCase):
    def test_frozen_config_rejects_protocol_changes(self) -> None:
        with self.assertRaises(ValueError):
            PusherV5EnvironmentAuditConfig(horizon=99)
        with self.assertRaises(ValueError):
            PusherV5EnvironmentAuditConfig(width=128)

    def test_reward_terms_use_post_step_relations(self) -> None:
        observation = np.zeros(23, dtype=np.float64)
        observation[14:17] = [1.0, 0.0, 0.0]
        observation[17:20] = [0.0, 0.0, 0.0]
        observation[20:23] = [0.0, 2.0, 0.0]
        action = np.asarray([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        terms = reward_terms(observation, action)
        self.assertEqual(terms["reward_dist"], -2.0)
        self.assertEqual(terms["reward_near"], -0.5)
        self.assertAlmostEqual(terms["reward_ctrl"], -0.2)

    def test_rollout_comparison_keeps_visual_gate_separate(self) -> None:
        row = {
            "observation": [1.0, 2.0],
            "reward": -1.0,
            "reward_terms": {"reward_dist": -1.0},
            "terminated": False,
            "truncated": False,
        }
        rollout = {
            "reset_observation": [0.0],
            "reset_render": {"hash": "reset"},
            "render_records": [{"hash": "reset"}, {"hash": "step"}],
            "rows": [row],
        }
        equal = compare_rollouts(rollout, rollout)
        self.assertTrue(equal["passed"])

        changed = dict(rollout)
        changed["render_records"] = [{"hash": "reset"}, {"hash": "changed"}]
        comparison = compare_rollouts(rollout, changed)
        self.assertTrue(comparison["observations_exact_equal"])
        self.assertFalse(comparison["frame_hashes_exact_equal"])
        self.assertFalse(comparison["passed"])

    def test_pixel_difference_diagnostic_does_not_relax_exact_gate(self) -> None:
        primary = np.zeros((2, 2, 3), dtype=np.uint8)
        replica = primary.copy()
        replica[0, 0, 0] = 1
        summary = pixel_difference_summary([primary], [replica])
        self.assertEqual(summary["mismatched_frame_count"], 1)
        self.assertEqual(summary["different_channel_value_count"], 1)
        self.assertEqual(summary["maximum_absolute_pixel_difference"], 1)
        self.assertAlmostEqual(summary["mean_absolute_pixel_difference"], 1 / 12)


if __name__ == "__main__":
    unittest.main()
