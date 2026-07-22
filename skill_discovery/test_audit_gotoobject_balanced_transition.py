"""Tests for balanced transition-aware GoToObject audits."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.audit_gotoobject_balanced_transition import (
    maximum_weight_assignment,
    supported_predecessors,
    transition_aware_matrix,
)


class BalancedTransitionAuditTest(unittest.TestCase):
    def test_global_assignment_resolves_local_collision(self) -> None:
        matrix = np.asarray(
            [
                [0.21, -2.67, 0.54],
                [-1.45, 0.43, 0.64],
                [0.04, 0.38, -3.94],
            ]
        )
        assignment, scores = maximum_weight_assignment(matrix)
        self.assertEqual(assignment, (0, 2, 1))
        self.assertEqual(len(scores), 6)

    def test_supported_predecessors_receive_zero_reward(self) -> None:
        counts = np.asarray(
            [
                [0, 100, 0],
                [80, 0, 120],
                [0, 0, 0],
            ]
        )
        predecessors = supported_predecessors(counts)
        self.assertEqual(predecessors, ((1,), (0,), (1,)))
        matrix = transition_aware_matrix((0, 2, 1), predecessors)
        self.assertEqual(matrix[1], (-1.0, 0.0, 1.0))
        self.assertEqual(matrix[2], (0.0, 1.0, -1.0))


if __name__ == "__main__":
    unittest.main()
