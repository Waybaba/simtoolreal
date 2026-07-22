"""Tests for the FrozenLake tabular skill trainer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from skill_discovery.train_frozenlake_skills import (
    FrozenLakeIntrinsicReward,
    FrozenLakeTrainConfig,
    FrozenLakeVisualClusterLookup,
    _step_size,
    train_run,
)


class FrozenLakeTrainerTest(unittest.TestCase):
    def test_visit_decay_step_size_decreases(self) -> None:
        constant = FrozenLakeTrainConfig(learning_rate=0.15)
        self.assertEqual(_step_size(constant, 100), 0.15)
        decayed = FrozenLakeTrainConfig(learning_rate_schedule="visit_decay")
        self.assertEqual(_step_size(decayed, 1), 1.0)
        self.assertLess(_step_size(decayed, 100), _step_size(decayed, 10))

    def test_class_specific_outcome_gates_are_resolved(self) -> None:
        config = FrozenLakeTrainConfig(outcome_rate_gates=(0.9, 0.8, 0.3))
        self.assertEqual(config.resolved_outcome_rate_gates(), (0.9, 0.8, 0.3))
        default = FrozenLakeTrainConfig()
        self.assertEqual(default.resolved_outcome_rate_gates(), (0.95,) * 3)
        with self.assertRaises(ValueError):
            FrozenLakeTrainConfig(outcome_rate_gates=(0.9, 0.8))

    def test_balanced_targets_are_seeded_and_reward_outcomes(self) -> None:
        config = FrozenLakeTrainConfig(
            objective="semantic_balanced",
            seed=19,
        )
        first = FrozenLakeIntrinsicReward(config)
        second = FrozenLakeIntrinsicReward(config)
        np.testing.assert_array_equal(first.balanced_targets, second.balanced_targets)
        self.assertEqual(sorted(first.balanced_targets.tolist()), [0, 1, 2])
        for skill, target in enumerate(first.balanced_targets):
            positive, _ = first.reward(skill, int(target), terminal_state=0)
            negative, _ = first.reward(
                skill,
                int((target + 1) % 3),
                terminal_state=0,
            )
            self.assertEqual(positive, 1.0)
            self.assertEqual(negative, 0.0)

    def test_semantic_spread_rewards_rare_outcome(self) -> None:
        config = FrozenLakeTrainConfig(
            objective="semantic_spread",
            semantic_decay=1.0,
        )
        reward_model = FrozenLakeIntrinsicReward(config)
        common_rewards = []
        for _ in range(32):
            reward, _ = reward_model.reward(0, 1, terminal_state=5)
            common_rewards.append(reward)
        rare_reward, parts = reward_model.reward(1, 2, terminal_state=15)
        self.assertGreater(parts["coverage_reward"], 0.0)
        self.assertGreater(rare_reward, common_rewards[-1])

    def test_visual_reward_uses_cluster_instead_of_outcome(self) -> None:
        config = FrozenLakeTrainConfig(
            objective="visual_cluster",
            visual_lookup_path="unused.npz",
            semantic_decay=1.0,
        )
        reward_model = FrozenLakeIntrinsicReward(config)
        reward_model.reward(0, outcome=0, terminal_state=0, visual_cluster=2)
        self.assertEqual(reward_model.semantic_counts[0].tolist(), [2.0, 2.0, 3.0])
        self.assertEqual(reward_model.outcome_counts.tolist(), [1, 0, 0])
        with self.assertRaises(ValueError):
            reward_model.reward(0, outcome=0, terminal_state=0)

    def test_visual_lookup_predicts_from_state_trajectory(self) -> None:
        frame_lookup = np.zeros((3, 16, 2), dtype=np.float32)
        frame_counts = np.zeros((3, 16), dtype=np.int64)
        frame_lookup[0, 0] = [1.0, 0.0]
        frame_lookup[0, 4] = [0.0, 1.0]
        frame_lookup[1, 5] = [1.0, 1.0]
        frame_counts[0, 0] = frame_counts[0, 4] = frame_counts[1, 5] = 1
        expected = np.asarray([1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        expected /= np.linalg.norm(expected)
        centers = np.stack((np.zeros(6), expected, np.ones(6))).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lookup.npz"
            np.savez_compressed(
                path,
                frame_lookup=frame_lookup,
                frame_counts=frame_counts,
                cluster_centers=centers,
            )
            lookup = FrozenLakeVisualClusterLookup(path)
            self.assertEqual(lookup.predict([0, 4, 5]), 1)
            with self.assertRaises(ValueError):
                lookup.predict([0, 1, 5])

    def test_tiny_training_writes_complete_artifacts(self) -> None:
        config = FrozenLakeTrainConfig(
            objective="semantic_balanced",
            seed=107,
            episodes=900,
            eval_interval=300,
            eval_episodes_per_skill=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "run"
            output = train_run(config, output_dir)
            self.assertEqual(output["reward_model"]["episode_counts"], [300] * 3)
            self.assertEqual(len(output["evaluations"]), 3)
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "q_table.npz").exists())
            self.assertTrue((output_dir / "policy_rollout_audit.png").exists())


if __name__ == "__main__":
    unittest.main()
