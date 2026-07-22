"""Tests for the unique-state DoorKey policy sequence generator."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from skill_discovery.generate_minigrid_doorkey_policy_sequences import (
    PolicySequenceDatasetConfig,
    UniqueRenderedStateCache,
    _write_contact_sheet,
    source_tabular_config,
)
from skill_discovery.train_minigrid_doorkey_tabular import DoorKeyTabularConfig


class DoorKeyPolicySequenceDatasetTest(unittest.TestCase):
    def test_generation_groups_must_be_unique(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            PolicySequenceDatasetConfig(generation_groups=(97, 97))

    def test_legacy_metrics_use_top_level_reward_and_default_assignment(self) -> None:
        defaults = DoorKeyTabularConfig()
        metrics = {
            "config": {
                "env_id": defaults.env_id,
                "seed": defaults.seed,
                "episodes": defaults.episodes,
                "horizon": defaults.horizon,
                "evaluation_checkpoints": defaults.evaluation_checkpoints,
                "stage_rate_gate": defaults.stage_rate_gate,
                "valid_action_mask": True,
                "terminate_on_target": True,
            },
            "reward_matrix": defaults.reward_matrix,
        }
        config = source_tabular_config(metrics)
        self.assertEqual(config.target_assignment, (0, 1, 2, 3))
        self.assertEqual(config.reward_matrix, defaults.reward_matrix)

    def test_identical_compact_state_reuses_cached_frame(self) -> None:
        cache = UniqueRenderedStateCache()
        key = tuple(range(12))
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.assertEqual(cache.add(key, frame, 1), 0)
        self.assertEqual(cache.add(key, frame.copy(), 1), 0)
        self.assertEqual(len(cache.frames), 1)

    def test_rgb_alias_fails_closed(self) -> None:
        cache = UniqueRenderedStateCache()
        key = tuple(range(12))
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        cache.add(key, frame, 1)
        frame[0, 0] = 255
        with self.assertRaisesRegex(ValueError, "multiple RGB"):
            cache.add(key, frame, 1)

    def test_stage_alias_fails_closed(self) -> None:
        cache = UniqueRenderedStateCache()
        key = tuple(range(12))
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        cache.add(key, frame, 1)
        with self.assertRaisesRegex(ValueError, "multiple oracle"):
            cache.add(key, frame, 2)

    def test_contact_sheet_repeats_a_rare_stage_for_display_only(self) -> None:
        frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
        stages = np.arange(4, dtype=np.int8)
        keys = np.tile(np.arange(12, dtype=np.int16), (4, 1))
        with TemporaryDirectory() as directory:
            manifest = _write_contact_sheet(
                Path(directory) / "audit.png",
                frames,
                stages,
                keys,
            )
        self.assertEqual(len(manifest), 12)
        self.assertTrue(
            all(row["unique_states_in_stage"] == 1 for row in manifest)
        )


if __name__ == "__main__":
    unittest.main()
