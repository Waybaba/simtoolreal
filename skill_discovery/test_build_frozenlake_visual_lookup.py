"""Tests for FrozenLake visual lookup trajectory keys."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.build_frozenlake_visual_lookup import (
    ACTIVE,
    TERMINAL_GOAL,
    TERMINAL_HOLE,
    audit_finite_keys,
    trajectory_frame_keys,
)


class FrozenLakeVisualLookupTest(unittest.TestCase):
    def test_trajectory_keys_preserve_temporal_state_and_terminal_type(self) -> None:
        states = np.asarray([0, 1, 2, 6, 10, 14, 15])
        goal = trajectory_frame_keys(states, 6)
        self.assertEqual(goal, ((ACTIVE, 0), (ACTIVE, 6), (TERMINAL_GOAL, 15)))
        hole = trajectory_frame_keys(np.asarray([0, 4, 5]), 2)
        self.assertEqual(hole, ((ACTIVE, 0), (ACTIVE, 4), (TERMINAL_HOLE, 5)))
        safe = trajectory_frame_keys(np.asarray([0, 1, 2, 3]), 3)
        self.assertEqual(safe, ((ACTIVE, 0), (ACTIVE, 2), (ACTIVE, 3)))

    def test_finite_key_audit_requires_distinct_singleton_clusters(self) -> None:
        lookup = np.zeros((3, 16, 2), dtype=np.float32)
        active_states = (0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14)
        lookup[ACTIVE, list(active_states)] = [1.0, 0.0]
        lookup[TERMINAL_HOLE, [5, 7, 11, 12]] = [0.0, 1.0]
        lookup[TERMINAL_GOAL, 15] = [-1.0, 0.0]
        centers = []
        for final in ([1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]):
            center = np.asarray([1.0, 0.0, 1.0, 0.0, *final])
            centers.append(center / np.linalg.norm(center))
        audit = audit_finite_keys(lookup, np.stack(centers))
        self.assertEqual(
            audit["clusters_by_outcome"],
            {"safe_timeout": [0], "hole_terminal": [1], "goal_terminal": [2]},
        )
        self.assertTrue(audit["one_to_one_partition_passed"])


if __name__ == "__main__":
    unittest.main()
