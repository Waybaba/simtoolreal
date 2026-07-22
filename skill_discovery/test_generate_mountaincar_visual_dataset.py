"""Tests for the MountainCar visual position-stage dataset."""

from __future__ import annotations

import unittest

from skill_discovery.generate_mountaincar_visual_dataset import (
    MOUNTAINCAR_POSITION_STAGES,
    MountainCarVisualDatasetConfig,
    position_stage,
)


class MountainCarVisualDatasetTest(unittest.TestCase):
    def test_position_regions_cover_frozen_intervals(self) -> None:
        self.assertEqual(position_stage(-1.0), 0)
        self.assertEqual(position_stage(-0.5), 1)
        self.assertEqual(position_stage(0.2), 2)
        self.assertEqual(position_stage(0.5), 3)
        self.assertEqual(len(MOUNTAINCAR_POSITION_STAGES), 4)

    def test_positions_outside_sweep_are_rejected(self) -> None:
        for position in (-1.2, 0.59):
            with self.assertRaises(ValueError):
                position_stage(position)

    def test_split_seeds_must_differ(self) -> None:
        with self.assertRaises(ValueError):
            MountainCarVisualDatasetConfig(reference_seed=1, audit_seed=1)


if __name__ == "__main__":
    unittest.main()
