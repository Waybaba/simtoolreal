"""Tests for DoorKey DINO representation composition and clustering."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.cluster_minigrid_doorkey_visual_embeddings import (
    _cluster_method,
    build_raw_representations,
)
from skill_discovery.encode_minigrid_doorkey_dinov2 import (
    build_dinov2_representations,
)


class DoorKeyVisualEmbeddingTest(unittest.TestCase):
    def test_dinov2_representations_use_four_stage_self_reference(self) -> None:
        rng = np.random.default_rng(7)
        frame_embeddings = rng.normal(size=(5, 4, 8)).astype(np.float32)
        representations = build_dinov2_representations(frame_embeddings)
        self.assertEqual(representations["dinov2_current"].shape, (20, 8))
        self.assertEqual(
            representations["dinov2_start_current"].shape,
            (20, 16),
        )
        self.assertEqual(
            representations["dinov2_temporal_delta"].shape,
            (20, 8),
        )
        np.testing.assert_allclose(
            representations["dinov2_temporal_delta"][::4],
            0.0,
            atol=1.0e-7,
        )

    def test_raw_representations_include_current_and_delta(self) -> None:
        frames = np.zeros((2, 4, 160, 160, 3), dtype=np.uint8)
        frames[:, 1:, 40:80, 40:80] = 255
        representations = build_raw_representations(frames)
        self.assertEqual(representations["raw_current"].shape, (8, 3072))
        self.assertEqual(
            representations["raw_temporal_delta"].shape,
            (8, 3072),
        )
        np.testing.assert_allclose(
            representations["raw_temporal_delta"][::4],
            0.0,
        )

    def test_kmeans_alignment_is_fit_only_on_train_features(self) -> None:
        rng = np.random.default_rng(17)
        centers = np.eye(4, dtype=np.float32) * 10.0
        train_stages = np.repeat(np.arange(4), 12)
        audit_stages = np.repeat(np.arange(4), 8)
        train = centers[train_stages] + rng.normal(0, 0.01, (48, 4))
        audit = centers[audit_stages] + rng.normal(0, 0.01, (32, 4))
        metrics, train_clusters, audit_clusters, fitted = _cluster_method(
            train,
            audit,
            train_stages,
            audit_stages,
            seed=7,
        )
        self.assertEqual(train_clusters.shape, (48,))
        self.assertEqual(audit_clusters.shape, (32,))
        self.assertEqual(fitted.shape, (4, 4))
        self.assertEqual(metrics["train_aligned_accuracy"], 1.0)
        self.assertEqual(metrics["audit_aligned_accuracy"], 1.0)
        self.assertTrue(metrics["all_clusters_nonempty"])


if __name__ == "__main__":
    unittest.main()
