"""Tests for frozen DoorKey visual-center policy-state transfer."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.evaluate_minigrid_doorkey_visual_state_transfer import (
    build_pair_dinov2_representations,
    build_pair_raw_representations,
    nearest_centers,
    transfer_metrics,
)


class DoorKeyVisualStateTransferTest(unittest.TestCase):
    def test_pair_representations_use_reset_as_reference(self) -> None:
        rng = np.random.default_rng(7)
        embeddings = rng.normal(size=(5, 2, 8)).astype(np.float32)
        embeddings[:, 1] = embeddings[:, 0]
        representations = build_pair_dinov2_representations(embeddings)
        self.assertEqual(representations["dinov2_current"].shape, (5, 8))
        self.assertEqual(
            representations["dinov2_start_current"].shape,
            (5, 16),
        )
        np.testing.assert_allclose(
            representations["dinov2_temporal_delta"],
            0.0,
            atol=1.0e-7,
        )
        frames = np.zeros((5, 2, 160, 160, 3), dtype=np.uint8)
        raw = build_pair_raw_representations(frames)
        self.assertEqual(raw["raw_current"].shape, (5, 3072))
        np.testing.assert_allclose(raw["raw_temporal_delta"], 0.0)

    def test_nearest_centers_assigns_expected_cluster(self) -> None:
        centers = np.eye(4, dtype=np.float32)
        features = centers[[3, 1, 0, 2]]
        np.testing.assert_array_equal(
            nearest_centers(features, centers),
            [3, 1, 0, 2],
        )

    def test_transfer_metrics_reports_each_generation_group(self) -> None:
        clusters = np.asarray([0, 1, 2, 3] * 2)
        stages = np.asarray([3, 2, 1, 0] * 2)
        groups = np.asarray([57] * 4 + [67] * 4)
        mapping = {
            "0": "goal_reached",
            "1": "door_opened",
            "2": "key_acquired",
            "3": "navigation_only",
        }
        output = transfer_metrics(clusters, stages, groups, mapping)
        self.assertEqual(output["accuracy"], 1.0)
        self.assertEqual(output["by_generation_group"]["57"]["accuracy"], 1.0)
        self.assertEqual(output["by_generation_group"]["67"]["accuracy"], 1.0)
        self.assertTrue(output["all_clusters_nonempty"])


if __name__ == "__main__":
    unittest.main()
