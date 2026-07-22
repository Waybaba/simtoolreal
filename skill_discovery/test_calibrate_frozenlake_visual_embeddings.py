"""Tests for trajectory self-reference visual representations."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.calibrate_frozenlake_visual_embeddings import (
    build_self_reference_representations,
)
from skill_discovery.encode_frozenlake_dinov2 import (
    compose_trajectory_embeddings,
)


class FrozenLakeSelfReferenceTest(unittest.TestCase):
    def test_representations_have_expected_shapes_and_norms(self) -> None:
        rng = np.random.default_rng(7)
        frames = rng.normal(size=(5, 3, 4)).astype(np.float32)
        representations = build_self_reference_representations(frames)
        self.assertEqual(representations["absolute_3frame"].shape, (5, 12))
        self.assertEqual(representations["middle_final"].shape, (5, 8))
        self.assertEqual(representations["temporal_delta"].shape, (5, 8))
        for features in representations.values():
            np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1.0)
        np.testing.assert_allclose(
            representations["absolute_3frame"],
            compose_trajectory_embeddings(frames.reshape(-1, 4), 5),
        )

    def test_positive_frame_scaling_does_not_change_representations(self) -> None:
        rng = np.random.default_rng(17)
        frames = rng.normal(size=(4, 3, 6)).astype(np.float32)
        scales = rng.uniform(0.2, 3.0, size=(4, 3, 1)).astype(np.float32)
        original = build_self_reference_representations(frames)
        scaled = build_self_reference_representations(frames * scales)
        for name in original:
            np.testing.assert_allclose(original[name], scaled[name], atol=1.0e-6)

    def test_invalid_frame_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_self_reference_representations(np.zeros((5, 8), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
