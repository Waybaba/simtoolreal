"""Tests for DINOv2 trajectory embedding composition."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.encode_frozenlake_dinov2 import (
    compose_trajectory_embeddings,
)


class FrozenLakeDinov2Test(unittest.TestCase):
    def test_composition_normalizes_and_preserves_frame_order(self) -> None:
        frames = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 2.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [3.0, 0.0],
                [1.0, -1.0],
            ],
            dtype=np.float32,
        )
        trajectories = compose_trajectory_embeddings(frames, 2)
        np.testing.assert_allclose(np.linalg.norm(trajectories, axis=1), 1.0)
        reordered = compose_trajectory_embeddings(frames[[1, 0, 2, 3, 4, 5]], 2)
        self.assertFalse(np.allclose(trajectories[0], reordered[0]))

    def test_composition_requires_three_frames(self) -> None:
        with self.assertRaises(ValueError):
            compose_trajectory_embeddings(np.ones((5, 4), dtype=np.float32), 2)


if __name__ == "__main__":
    unittest.main()
