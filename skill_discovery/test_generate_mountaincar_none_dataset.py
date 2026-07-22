"""Tests for the explicit MountainCar none dataset."""

from __future__ import annotations

import unittest

from skill_discovery.generate_mountaincar_none_dataset import NoneDatasetConfig


class MountainCarNoneDatasetTest(unittest.TestCase):
    def test_none_split_seeds_must_differ(self) -> None:
        with self.assertRaises(ValueError):
            NoneDatasetConfig(reference_seed=1, audit_seed=1)

    def test_none_budget_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            NoneDatasetConfig(samples_per_split=0)


if __name__ == "__main__":
    unittest.main()
