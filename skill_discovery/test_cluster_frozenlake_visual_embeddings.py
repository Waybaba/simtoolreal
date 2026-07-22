"""Tests for unsupervised visual cluster alignment helpers."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.cluster_frozenlake_visual_embeddings import (
    best_cluster_mapping,
)


class FrozenLakeVisualClusterTest(unittest.TestCase):
    def test_best_mapping_is_permutation_invariant(self) -> None:
        clusters = np.asarray([2, 2, 0, 0, 1, 1])
        outcomes = np.asarray([0, 0, 1, 1, 2, 2])
        mapping, accuracy = best_cluster_mapping(clusters, outcomes, 3)
        self.assertEqual(mapping, [1, 2, 0])
        self.assertEqual(accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
