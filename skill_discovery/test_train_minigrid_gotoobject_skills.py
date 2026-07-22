"""Tests for mission-free GoToObject tabular skill training."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from skill_discovery.minigrid_gotoobject import make_gotoobject
from skill_discovery.train_minigrid_gotoobject_skills import (
    GoToObjectReward,
    GoToObjectTrainConfig,
    _learning_rate,
    _transition_reward,
    compact_relation_key,
    frozen_reward_matrix_from_metrics,
    train_run,
)


class GoToObjectSkillTrainerTest(unittest.TestCase):
    def test_frozen_reward_matrix_uses_source_formula(self) -> None:
        metrics = {
            "config": {"semantic_coverage_weight": 1.0},
            "reward_model": {
                "objective": "semantic_spread",
                "semantic_counts": [
                    [8.0, 1.0, 1.0],
                    [1.0, 4.0, 1.0],
                    [1.0, 1.0, 2.0],
                ],
            },
        }
        matrix = frozen_reward_matrix_from_metrics(metrics)
        expected = math.log(3 * 8 / 10) - math.log(3 * 10 / 20)
        self.assertAlmostEqual(matrix[0][0], expected)
        calibrated = frozen_reward_matrix_from_metrics(
            metrics,
            calibration="runner_up_unit",
        )
        for row in calibrated:
            ordered = sorted(row)
            self.assertAlmostEqual(ordered[-1], 1.0)
            self.assertAlmostEqual(ordered[-2], 0.0)
        config = GoToObjectTrainConfig(
            objective="frozen_matrix",
            frozen_reward_matrix=matrix,
            episodes=1,
        )
        reward = GoToObjectReward(config)
        value, _ = reward.reward(skill=0, stage=0)
        self.assertAlmostEqual(value, expected)

    def test_learning_rate_modes(self) -> None:
        visit_config = GoToObjectTrainConfig(episodes=1)
        self.assertAlmostEqual(_learning_rate(visit_config, 4.0), 4.0**-0.6)
        constant_config = GoToObjectTrainConfig(
            learning_rate_mode="constant",
            constant_learning_rate=0.17,
            episodes=1,
        )
        self.assertEqual(_learning_rate(constant_config, 1.0), 0.17)
        self.assertEqual(_learning_rate(constant_config, 10_000.0), 0.17)

    def test_reward_timing_preserves_terminal_default(self) -> None:
        terminal_config = GoToObjectTrainConfig(episodes=1)
        terminal_reward = GoToObjectReward(terminal_config)
        value, _ = _transition_reward(
            terminal_reward,
            terminal_config,
            skill=0,
            stage=int(terminal_reward.balanced_targets[0]),
            terminal=False,
        )
        self.assertEqual(value, 0.0)
        self.assertEqual(terminal_reward.reward_calls_by_skill.tolist(), [0, 0, 0])

        occupancy_config = GoToObjectTrainConfig(
            reward_timing="occupancy",
            episodes=1,
        )
        occupancy_reward = GoToObjectReward(occupancy_config)
        value, _ = _transition_reward(
            occupancy_reward,
            occupancy_config,
            skill=0,
            stage=int(occupancy_reward.balanced_targets[0]),
            terminal=False,
        )
        self.assertEqual(value, 1.0)
        self.assertEqual(
            occupancy_reward.reward_calls_by_skill.tolist(),
            [1, 0, 0],
        )

    def test_compact_key_ignores_mission_text(self) -> None:
        env = make_gotoobject()
        try:
            env.reset(seed=7)
            first = compact_relation_key(env)
            env.unwrapped.mission = "deliberately changed mission"
            env.unwrapped.targetType = "key"
            env.unwrapped.target_color = "red"
            env.unwrapped.target_pos = (1, 1)
            self.assertEqual(first, compact_relation_key(env))
        finally:
            env.close()

    def test_balanced_reward_targets_are_seeded(self) -> None:
        config = GoToObjectTrainConfig(seed=19, episodes=3, eval_interval=1)
        first = GoToObjectReward(config)
        second = GoToObjectReward(config)
        self.assertEqual(first.balanced_targets.tolist(), second.balanced_targets.tolist())
        self.assertEqual(sorted(first.balanced_targets.tolist()), [0, 1, 2])
        for skill, target in enumerate(first.balanced_targets):
            positive, _ = first.reward(skill, int(target))
            self.assertEqual(positive, 1.0)

    def test_tiny_training_writes_artifacts(self) -> None:
        config = GoToObjectTrainConfig(
            seed=107,
            episodes=300,
            horizon=16,
            eval_interval=100,
            eval_episodes_per_skill=4,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "run"
            output = train_run(config, output_dir)
            self.assertEqual(output["reward_model"]["episode_counts"], [100] * 3)
            self.assertEqual(len(output["evaluations"]), 3)
            self.assertGreater(output["q_state_count"], 0)
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "q_table.npz").exists())
            self.assertTrue((output_dir / "policy_rollout_audit.png").exists())


if __name__ == "__main__":
    unittest.main()
