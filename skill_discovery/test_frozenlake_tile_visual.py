"""Tests for RGB-only FrozenLake tile selection and pooling."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.frozenlake_tile_visual import (
    compose_tile_delta_embeddings,
    extract_selected_tile_pairs,
    select_top_change_tiles,
    selection_metrics,
)


class FrozenLakeTileVisualTest(unittest.TestCase):
    def test_selection_finds_changed_tiles_and_extracts_pairs(self) -> None:
        frames = np.zeros((1, 3, 8, 8, 3), dtype=np.uint8)
        frames[0, 1, 0:2, 0:2] = 100
        frames[0, 2, 6:8, 6:8] = 200
        selected, scores, capture = select_top_change_tiles(frames, 4)
        self.assertEqual(int(selected[0, 0, 0]), 0)
        self.assertEqual(int(selected[0, 1, 0]), 15)
        self.assertEqual(float(capture[0, 0]), 1.0)
        self.assertEqual(float(capture[0, 1]), 1.0)
        pairs = extract_selected_tile_pairs(
            frames,
            selected,
            4,
            output_size=4,
        )
        self.assertEqual(pairs.shape, (1, 2, 4, 2, 4, 4, 3))
        self.assertEqual(int(pairs[0, 1, 0, 1].max()), 200)
        self.assertTrue(selection_metrics(scores, capture)["gate_passed"])

    def test_tile_delta_pooling_is_permutation_invariant(self) -> None:
        rng = np.random.default_rng(7)
        embeddings = rng.normal(size=(3, 2, 4, 2, 8)).astype(np.float32)
        original = compose_tile_delta_embeddings(embeddings)
        permuted = compose_tile_delta_embeddings(embeddings[:, :, [2, 0, 3, 1]])
        self.assertEqual(original.shape, (3, 48))
        np.testing.assert_allclose(original, permuted, atol=1.0e-6)
        np.testing.assert_allclose(np.linalg.norm(original, axis=1), 1.0)

    def test_invalid_grid_and_embedding_shapes_are_rejected(self) -> None:
        frames = np.zeros((1, 3, 7, 7, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            select_top_change_tiles(frames, 4)
        with self.assertRaises(ValueError):
            compose_tile_delta_embeddings(np.zeros((1, 2, 3, 2, 8)))


if __name__ == "__main__":
    unittest.main()
