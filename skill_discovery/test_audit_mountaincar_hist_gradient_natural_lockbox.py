"""Tests for the fresh MountainCar natural lockbox protocol."""

from __future__ import annotations

import unittest

from skill_discovery.audit_mountaincar_hist_gradient_natural_lockbox import (
    fresh_natural_config,
)


class MountainCarHistGradientNaturalLockboxTest(unittest.TestCase):
    def test_fresh_natural_protocol_is_frozen(self) -> None:
        config = fresh_natural_config()
        self.assertEqual(config.energy_episodes, 128)
        self.assertEqual(config.random_episodes, 256)
        self.assertEqual(config.random_horizon, 200)
        self.assertEqual(config.energy_seed_start, 15_100_000)
        self.assertEqual(config.random_seed_start, 16_100_000)
        self.assertEqual(config.action_seed, 17_100_007)
        self.assertEqual(config.workers, 4)


if __name__ == "__main__":
    unittest.main()
