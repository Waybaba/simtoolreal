"""Tests for FrozenLake tile-DINO input preparation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from skill_discovery.encode_frozenlake_tile_dinov2 import (
    prepare_tile_trajectories,
)


class FrozenLakeTileDinoPreparationTest(unittest.TestCase):
    def _dataset(self, path: Path) -> np.lib.npyio.NpzFile:
        frames = np.zeros((18, 3, 8, 8, 3), dtype=np.uint8)
        outcomes = np.repeat(np.arange(3), 6)
        split = np.tile((0, 0, 0, 1, 1, 1), 3)
        seeds = np.where(split == 0, 7, 37)
        np.savez_compressed(
            path,
            frames=frames,
            outcomes=outcomes,
            split=split,
            generation_seeds=seeds,
        )
        return np.load(path)

    def test_base_mode_preserves_dataset_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = self._dataset(Path(temporary) / "data.npz")
            frames, outcomes, split, seeds, metadata = prepare_tile_trajectories(
                data,
                audit_nuisance=False,
                style_seeds=None,
            )
            self.assertEqual(len(frames), 18)
            np.testing.assert_array_equal(outcomes, data["outcomes"])
            np.testing.assert_array_equal(split, data["split"])
            np.testing.assert_array_equal(seeds, data["generation_seeds"])
            self.assertIsNone(metadata["source_indices"])

    def test_style_mode_rejects_wrong_seed_count_before_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = self._dataset(Path(temporary) / "data.npz")
            with self.assertRaises(ValueError):
                prepare_tile_trajectories(
                    data,
                    audit_nuisance=False,
                    style_seeds=(1, 2),
                )


if __name__ == "__main__":
    unittest.main()
