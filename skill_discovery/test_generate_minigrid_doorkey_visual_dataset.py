"""Tests for balanced DoorKey visual dataset generation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from skill_discovery.generate_minigrid_doorkey_visual_dataset import (
    DoorKeyVisualDatasetConfig,
    generate_dataset,
)


class DoorKeyVisualDatasetTest(unittest.TestCase):
    def test_generation_groups_must_be_disjoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "disjoint"):
            DoorKeyVisualDatasetConfig(
                train_generation_groups=(7, 17),
                audit_generation_groups=(17, 27),
            )

    def test_tiny_dataset_is_balanced_and_split_safe(self) -> None:
        config = DoorKeyVisualDatasetConfig(
            train_generation_groups=(7,),
            audit_generation_groups=(37,),
            trajectories_per_group=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "dataset"
            output = generate_dataset(config, output_dir)
            self.assertTrue(output["data_gate_passed"])
            self.assertEqual(output["trajectory_count"], 4)
            self.assertEqual(output["frame_count"], 16)
            self.assertEqual(output["stage_frame_counts"], {
                "navigation_only": 4,
                "key_acquired": 4,
                "door_opened": 4,
                "goal_reached": 4,
            })
            with np.load(output["dataset"]) as data:
                self.assertEqual(data["frames"].shape, (4, 4, 160, 160, 3))
                np.testing.assert_array_equal(
                    data["stage_labels"],
                    np.tile(np.arange(4), (4, 1)),
                )
                self.assertEqual(set(data["split"].tolist()), {0, 1})
                self.assertTrue(np.all(data["native_rewards"] > 0))
            manifest = json.loads(
                (output_dir / "manual_stage_audit.json").read_text()
            )
            self.assertEqual(len(manifest), 4)
            self.assertTrue((output_dir / "manual_stage_audit.png").exists())


if __name__ == "__main__":
    unittest.main()
