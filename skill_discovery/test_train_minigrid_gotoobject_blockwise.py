"""Tests for block-wise GoToObject spread training."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from skill_discovery.train_minigrid_gotoobject_blockwise import (
    BlockwiseSpreadConfig,
    _is_evaluation_checkpoint,
    _replay_frozen_transitions,
    common_layout_seed,
    snapshot_spread_matrix,
    train_blockwise_run,
)
from skill_discovery.train_minigrid_gotoobject_skills import (
    GoToObjectReward,
    GoToObjectTrainConfig,
)


class BlockwiseSpreadTrainerTest(unittest.TestCase):
    def test_common_layout_seed_groups_three_skills(self) -> None:
        seeds = [common_layout_seed(7, 500_000, episode) for episode in range(6)]
        self.assertEqual(seeds[:3], [7_500_000] * 3)
        self.assertEqual(seeds[3:], [7_500_001] * 3)

    def test_explicit_evaluation_checkpoints_override_interval(self) -> None:
        config = BlockwiseSpreadConfig(
            bootstrap_episodes=3,
            policy_episodes=12,
            eval_interval=3,
            evaluation_checkpoints=(2, 10, 11, 12),
        )
        observed = [
            episode
            for episode in range(1, 13)
            if _is_evaluation_checkpoint(config, episode)
        ]
        self.assertEqual(observed, [2, 10, 11, 12])

    def test_snapshot_calibrates_unique_spread_rows(self) -> None:
        config = BlockwiseSpreadConfig(
            bootstrap_episodes=3,
            policy_episodes=3,
            eval_interval=1,
        )
        reward_config = GoToObjectTrainConfig(
            objective="semantic_spread",
            reward_timing="occupancy",
            episodes=3,
            eval_interval=1,
        )
        reward_model = GoToObjectReward(reward_config)
        reward_model.counts = np.asarray(
            [
                [1.0, 1.0, 100.0],
                [1.0, 100.0, 1.0],
                [100.0, 1.0, 1.0],
            ]
        )
        matrix = snapshot_spread_matrix(
            reward_model,
            config,
            calibration="runner_up_unit",
        )
        self.assertEqual(
            tuple(max(range(3), key=row.__getitem__) for row in matrix),
            (2, 1, 0),
        )
        for row in matrix:
            ordered = sorted(row)
            self.assertAlmostEqual(ordered[-1], 1.0)
            self.assertAlmostEqual(ordered[-2], 0.0)

    def test_reverse_replay_propagates_frozen_reward(self) -> None:
        matrix = (
            (0.0, -1.0, 1.0),
            (1.0, 0.0, -1.0),
            (-1.0, 1.0, 0.0),
        )
        policy_config = GoToObjectTrainConfig(
            objective="frozen_matrix",
            frozen_reward_matrix=matrix,
            reward_timing="occupancy",
            episodes=2,
            eval_interval=1,
        )
        first = (1, 1, 0, 2, 2, 3, 3, 0)
        second = (1, 2, 0, 2, 2, 3, 3, 0)
        replay_buffer = [
            [
                (first, 0, 0, 0, second, False),
                (second, 0, 1, 2, None, True),
            ]
        ]
        q_table = {}
        visits = {}
        summary = _replay_frozen_transitions(
            replay_buffer,
            policy_config,
            q_table,
            visits,
        )
        self.assertEqual(summary["transitions"], 2)
        self.assertAlmostEqual(q_table[second][0, 1], 1.0)
        self.assertAlmostEqual(q_table[first][0, 0], 0.99)

    def test_tiny_run_preserves_bootstrap_artifacts(self) -> None:
        config = BlockwiseSpreadConfig(
            seed=109,
            bootstrap_episodes=30,
            policy_episodes=30,
            horizon=8,
            eval_interval=15,
            eval_episodes_per_skill=2,
            stability_checkpoints=2,
            bootstrap_replay="reverse_once",
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "run"
            output = train_blockwise_run(config, output_dir)
            self.assertIn("bootstrap_gate_passed", output)
            self.assertEqual(len(output["bootstrap_top_stages"]), 3)
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "bootstrap_q_table.npz").exists())
            self.assertTrue((output_dir / "bootstrap_replay_buffer.npz").exists())
            if output["policy_phase_ran"]:
                self.assertTrue((output_dir / "q_table.npz").exists())
                self.assertTrue(
                    (output_dir / "relabelled_bootstrap_q_table.npz").exists()
                )
                self.assertEqual(len(output["evaluations"]), 2)


if __name__ == "__main__":
    unittest.main()
