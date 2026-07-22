"""Tests for cached FrozenLake visual trajectory generation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from skill_discovery.frozenlake import FrozenLakeConfig
from skill_discovery.generate_frozenlake_visual_dataset import (
    FrozenLakeVisualDatasetConfig,
    generate_dataset,
)


class FrozenLakeVisualDatasetTest(unittest.TestCase):
    def test_small_dataset_is_balanced_and_split_by_seed(self) -> None:
        config = FrozenLakeVisualDatasetConfig(
            seeds=(7, 17),
            trajectories_per_outcome_per_seed=2,
            image_size=16,
            sample_trajectories_per_outcome=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "dataset"
            summary = generate_dataset(config, output_dir)
            data = np.load(output_dir / "trajectories.npz")
            self.assertTrue(summary["passed"])
            self.assertEqual(data["frames"].shape, (12, 3, 16, 16, 3))
            self.assertEqual(np.bincount(data["outcomes"]).tolist(), [4, 4, 4])
            self.assertEqual(set(data["generation_seeds"][data["split"] == 0]), {7})
            self.assertEqual(set(data["generation_seeds"][data["split"] == 1]), {17})
            self.assertTrue((output_dir / "visual_sample_30.png").exists())

    def test_tiny_8x8_dataset_uses_generic_state_padding(self) -> None:
        config = FrozenLakeVisualDatasetConfig(
            seeds=(7, 17),
            trajectories_per_outcome_per_seed=1,
            image_size=16,
            sample_trajectories_per_outcome=1,
            lake=FrozenLakeConfig(
                map_name="8x8",
                is_slippery=True,
                max_episode_steps=128,
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "dataset"
            output = generate_dataset(config, output_dir)
            data = np.load(output["dataset"])
            self.assertEqual(data["frames"].shape, (6, 3, 16, 16, 3))
            self.assertEqual(data["states"].shape, (6, 129))
            self.assertEqual(data["actions"].shape, (6, 128))
            self.assertEqual(
                np.bincount(data["outcomes"], minlength=3).tolist(),
                [2] * 3,
            )


if __name__ == "__main__":
    unittest.main()
