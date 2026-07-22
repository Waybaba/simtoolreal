"""Tests for MountainCar stratified none capacity helpers."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.audit_mountaincar_stratified_none_capacity import (
    VISIBLE_STRATA,
    StratifiedNoneCapacityConfig,
    stratum_matches,
)


class MountainCarStratifiedNoneCapacityTest(unittest.TestCase):
    def test_frozen_strata_boundaries(self) -> None:
        self.assertTrue(stratum_matches(0, np.asarray([-1.16, 0.0]), False))
        self.assertTrue(stratum_matches(1, np.asarray([-1.0, 0.0]), False))
        self.assertFalse(stratum_matches(1, np.asarray([-1.0, -0.005]), False))
        self.assertTrue(stratum_matches(2, np.asarray([-1.0, 0.005]), False))
        self.assertTrue(stratum_matches(3, np.asarray([-0.4, -0.005]), False))
        self.assertTrue(stratum_matches(4, np.asarray([-0.4, -0.006]), False))
        self.assertTrue(stratum_matches(5, np.asarray([0.2, 0.0]), False))
        self.assertTrue(stratum_matches(6, np.asarray([0.2, -0.006]), False))
        self.assertTrue(stratum_matches(7, np.asarray([0.4, -0.006]), False))

    def test_relation_samples_match_no_none_stratum(self) -> None:
        relation_samples = (
            (np.asarray([-1.0, -0.006]), False),
            (np.asarray([-0.4, 0.006]), False),
            (np.asarray([0.2, 0.006]), False),
            (np.asarray([0.46, 0.01]), True),
        )
        for next_state, terminated in relation_samples:
            self.assertFalse(
                any(
                    stratum_matches(index, next_state, terminated)
                    for index in range(8)
                )
            )

    def test_visible_valley_threshold_excludes_stationary_motion(self) -> None:
        self.assertFalse(
            stratum_matches(3, np.asarray([-0.4, 0.0]), False, VISIBLE_STRATA)
        )
        self.assertTrue(
            stratum_matches(3, np.asarray([-0.4, -0.001]), False, VISIBLE_STRATA)
        )
        self.assertTrue(
            stratum_matches(3, np.asarray([-0.4, 0.001]), False, VISIBLE_STRATA)
        )

    def test_capacity_budget_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            StratifiedNoneCapacityConfig(accepted_per_stratum=0)


if __name__ == "__main__":
    unittest.main()
