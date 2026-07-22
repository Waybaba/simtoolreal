"""Tests for the visible-stratified MountainCar none dataset."""

from __future__ import annotations

import unittest

from skill_discovery.generate_mountaincar_visible_stratified_none_dataset import (
    VisibleStratifiedNoneDatasetConfig,
)


class MountainCarVisibleStratifiedNoneDatasetTest(unittest.TestCase):
    def test_split_seeds_must_differ(self) -> None:
        with self.assertRaises(ValueError):
            VisibleStratifiedNoneDatasetConfig(reference_seed=1, audit_seed=1)

    def test_budget_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            VisibleStratifiedNoneDatasetConfig(samples_per_stratum=0)


if __name__ == "__main__":
    unittest.main()
