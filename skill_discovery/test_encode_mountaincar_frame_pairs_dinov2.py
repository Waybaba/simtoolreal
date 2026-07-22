"""Tests for MountainCar DINO pair feature construction."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.encode_mountaincar_frame_pairs_dinov2 import (
    build_pair_features,
)


class MountainCarPairDinoTest(unittest.TestCase):
    def test_pair_feature_contains_ordered_delta(self) -> None:
        embeddings = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
        features = build_pair_features(embeddings)
        self.assertEqual(features.shape, (1, 6))
        self.assertAlmostEqual(float(np.linalg.norm(features[0])), 1.0)
        self.assertGreater(features[0, 3], 0.0)
        self.assertLess(features[0, 4], 0.0)

    def test_pair_feature_rejects_single_frames(self) -> None:
        with self.assertRaises(ValueError):
            build_pair_features(np.zeros((2, 3), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
