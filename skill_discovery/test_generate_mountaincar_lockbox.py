"""Tests for the fresh MountainCar balanced lockbox generator."""

from __future__ import annotations

import unittest

from skill_discovery.generate_mountaincar_lockbox import MountainCarLockboxConfig


class MountainCarLockboxGeneratorTest(unittest.TestCase):
    def test_frozen_lockbox_budgets(self) -> None:
        config = MountainCarLockboxConfig()
        self.assertEqual(config.samples_per_relation, 256)
        self.assertEqual(config.samples_per_none_stratum, 32)
        self.assertEqual(config.relation_seed, 12_100_007)
        self.assertEqual(config.none_seed, 13_100_007)

    def test_lockbox_budget_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            MountainCarLockboxConfig(samples_per_relation=0)


if __name__ == "__main__":
    unittest.main()
