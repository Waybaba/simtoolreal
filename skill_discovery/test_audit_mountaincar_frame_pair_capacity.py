"""Tests for the MountainCar frame-pair capacity survey."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.audit_mountaincar_frame_pair_capacity import (
    FramePairCapacityConfig,
    pair_hash,
    transition_class,
)


class MountainCarFramePairCapacityTest(unittest.TestCase):
    def test_transition_classes_match_frozen_relations(self) -> None:
        self.assertTrue(transition_class(0, np.asarray([-0.9, -0.02]), False))
        self.assertTrue(transition_class(1, np.asarray([-0.4, 0.02]), False))
        self.assertTrue(transition_class(2, np.asarray([0.2, 0.02]), False))
        self.assertTrue(transition_class(3, np.asarray([0.46, 0.02]), True))
        self.assertFalse(transition_class(3, np.asarray([0.46, -0.02]), True))

    def test_pair_hash_depends_on_order(self) -> None:
        left = np.zeros((2, 2, 3), dtype=np.uint8)
        right = left.copy()
        right[0, 0] = 1
        self.assertNotEqual(pair_hash(left, right), pair_hash(right, left))

    def test_capacity_gate_cannot_exceed_sample_budget(self) -> None:
        with self.assertRaises(ValueError):
            FramePairCapacityConfig(
                accepted_per_class=10,
                minimum_unique_pairs=11,
            )


if __name__ == "__main__":
    unittest.main()
