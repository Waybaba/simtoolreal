"""Tests for FrozenLake finite-horizon stochastic outcome control."""

from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
