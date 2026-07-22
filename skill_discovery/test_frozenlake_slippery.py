"""Tests for FrozenLake finite-horizon stochastic outcome control."""

from __future__ import annotations

import unittest

from skill_discovery.audit_frozenlake_slippery import _environment_smoke
from skill_discovery.frozenlake import FrozenLakeConfig
from skill_discovery.frozenlake_slippery import (
    finite_horizon_outcome_policy,
    rollout_outcome_policy,
)


class FrozenLakeSlipperyTest(unittest.TestCase):
    def test_finite_horizon_policies_are_bounded_and_reproducible(self) -> None:
        config = FrozenLakeConfig(is_slippery=True)
        for target in range(3):
            policy = finite_horizon_outcome_policy(config, target)
            self.assertGreater(policy.probability_upper_bound, 0.0)
            self.assertLessEqual(policy.probability_upper_bound, 1.0)
            first = rollout_outcome_policy(config, policy, seed=700 + target)
            second = rollout_outcome_policy(config, policy, seed=700 + target)
            self.assertEqual(first["states"], second["states"])
            self.assertEqual(first["outcome"], second["outcome"])

    def test_non_slippery_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            finite_horizon_outcome_policy(FrozenLakeConfig(), 2)

    def test_official_8x8_environment_and_policy_are_generic(self) -> None:
        config = FrozenLakeConfig(
            map_name="8x8",
            is_slippery=True,
            max_episode_steps=128,
        )
        smoke = _environment_smoke(config, seed=7)
        self.assertEqual(smoke["state_count"], 64)
        self.assertEqual(smoke["map_shape"], [8, 8])
        self.assertEqual(smoke["frame_shape"], [512, 512, 3])
        self.assertTrue(smoke["seed_reproducible"])
        goal = finite_horizon_outcome_policy(config, 2)
        self.assertGreater(goal.probability_upper_bound, 0.70)
        self.assertLess(goal.probability_upper_bound, 0.80)


if __name__ == "__main__":
    unittest.main()
