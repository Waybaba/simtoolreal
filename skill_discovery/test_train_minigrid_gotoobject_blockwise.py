"""Tests for block-wise GoToObject spread training."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from skill_discovery.train_minigrid_gotoobject_blockwise import (
    BlockwiseSpreadConfig,
    StageObserver,
    _is_evaluation_checkpoint,
    _replay_frozen_transitions,
    common_layout_seed,
    resolve_bootstrap_matrix,
    snapshot_spread_matrix,
    train_blockwise_run,
)
from skill_discovery.minigrid_gotoobject import make_gotoobject
from skill_discovery.train_minigrid_gotoobject_skills import (
    GoToObjectReward,
    GoToObjectTrainConfig,
)


class BlockwiseSpreadTrainerTest(unittest.TestCase):
    def test_visual_stage_observer_is_cached_and_matches_oracle(self) -> None:
        env = make_gotoobject(render_mode="rgb_array")
        observer = StageObserver("rgb_template_object_graph")
        try:
            env.reset(seed=7)
            first = observer.stage(env)
            second = observer.stage(env)
        finally:
            env.close()
        self.assertEqual(first, second)
        self.assertEqual(observer.query_count, 2)
        self.assertEqual(observer.parser_calls, 1)
        self.assertEqual(observer.cache_hits, 1)
        self.assertEqual(observer.mismatch_count, 0)
        self.assertEqual(observer.minimum_exact_tile_fraction, 1.0)

    def test_unknown_stage_source_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "stage source"):
            BlockwiseSpreadConfig(stage_source="unknown")

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

    def test_balanced_transition_resolution_repairs_collision(self) -> None:
        config = BlockwiseSpreadConfig(
            bootstrap_episodes=3,
            policy_episodes=3,
            eval_interval=1,
            matrix_strategy="balanced_transition",
        )
        reward_config = GoToObjectTrainConfig(
            objective="semantic_spread",
            reward_timing="occupancy",
            episodes=3,
            eval_interval=1,
        )
        reward_model = GoToObjectReward(reward_config)
        raw_matrix = (
            (0.21, -2.67, 0.54),
            (-1.45, 0.43, 0.64),
            (0.04, 0.38, -3.94),
        )
        key = (1, 1, 0, 2, 2, 3, 3, 0)
        replay_buffer = [
            [(key, 0, 0, stage, key, False) for stage in [0, 1] * 30],
            [(key, 1, 0, stage, key, False) for stage in [1, 2] * 30],
        ]
        resolution = resolve_bootstrap_matrix(
            raw_matrix,
            reward_model,
            config,
            replay_buffer,
        )
        self.assertTrue(resolution["gate_passed"])
        self.assertEqual(resolution["independent_top_stages"], (2, 2, 1))
        self.assertEqual(resolution["assigned_stages"], (0, 2, 1))
        self.assertEqual(
            resolution["matrix"],
            ((1.0, 0.0, -1.0), (-1.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        )

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
            self.assertEqual(len(output["bootstrap_assigned_stages"]), 3)
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
