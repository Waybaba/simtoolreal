"""Tests for complete-domain Taxi visual dataset splitting."""

from __future__ import annotations

import unittest

import gymnasium as gym

from skill_discovery.audit_taxi_environment import reachable_state_ids
from skill_discovery.generate_taxi_visual_dataset import (
    TaxiVisualDatasetConfig,
    split_counts_without_render,
    taxi_visual_split,
)


class TaxiVisualDatasetTest(unittest.TestCase):
    def test_destination_and_orientation_splits_are_disjoint(self) -> None:
        config = TaxiVisualDatasetConfig()
        self.assertEqual(taxi_visual_split(0, 0, config), 0)
        self.assertEqual(taxi_visual_split(1, 1, config), 1)
        self.assertEqual(taxi_visual_split(1, 0, config), 2)
        self.assertEqual(taxi_visual_split(0, 1, config), 3)

    def test_all_four_splits_have_fixed_natural_stage_counts(self) -> None:
        config = TaxiVisualDatasetConfig()
        env = gym.make(config.env_id)
        try:
            _, reachable, _ = reachable_state_ids(env.unwrapped)
            counts = split_counts_without_render(env.unwrapped, reachable, config)
        finally:
            env.close()
        self.assertEqual(counts.tolist(), [[300, 100, 4]] * 4)


if __name__ == "__main__":
    unittest.main()
