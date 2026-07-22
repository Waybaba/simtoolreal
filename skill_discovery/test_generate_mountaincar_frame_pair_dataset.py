"""Tests for the balanced MountainCar frame-pair dataset."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.generate_mountaincar_frame_pair_dataset import (
    FramePairDatasetConfig,
    pair_hash,
)


class MountainCarFramePairDatasetTest(unittest.TestCase):
    def test_dataset_split_seeds_must_differ(self) -> None:
        with self.assertRaises(ValueError):
            FramePairDatasetConfig(reference_seed=1, audit_seed=1)

    def test_pair_hash_is_ordered(self) -> None:
        before = np.zeros((2, 2, 3), dtype=np.uint8)
        after = before.copy()
        after[0, 0] = 2
        self.assertNotEqual(pair_hash(before, after), pair_hash(after, before))


if __name__ == "__main__":
    unittest.main()
