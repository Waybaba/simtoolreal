"""Tests for the Pusher-Cup tabular DIAYN trainer."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.train_pusher_diayn import (
    PusherTrainConfig,
    _balanced_class_targets,
    _best_class_assignment,
    _graph_state_ids,
    evaluate,
    train,
)


class PusherTrainerTest(unittest.TestCase):
    def test_balanced_targets_are_a_seeded_permutation(self) -> None:
        first = _balanced_class_targets(7, 3)
        second = _balanced_class_targets(7, 3)
        self.assertEqual(sorted(first.tolist()), [0, 1, 2])
        np.testing.assert_array_equal(first, second)

    def test_graph_state_ids_cover_four_coordinates(self) -> None:
        pusher = np.asarray(((-1.0, -1.0), (1.0, 1.0)), dtype=np.float32)
        ball = np.asarray(((-1.0, 1.0), (1.0, -1.0)), dtype=np.float32)
        ids = _graph_state_ids(pusher, ball, grid_size=3)
        self.assertEqual(ids.tolist(), [2, 78])

    def test_best_assignment_is_permutation_invariant(self) -> None:
        rates = np.asarray(
            (
                (0.05, 0.90, 0.05),
                (0.80, 0.10, 0.10),
                (0.02, 0.08, 0.90),
            )
        )
        assignment, matched = _best_class_assignment(rates)
        self.assertEqual(assignment, [1, 0, 2])
        np.testing.assert_allclose(matched, (0.90, 0.80, 0.90))

    def test_tiny_training_and_evaluation(self) -> None:
        config = PusherTrainConfig(
            representation="semantic",
            envs_per_skill=8,
            iterations=2,
            episode_length=6,
            policy_grid_size=5,
            raw_feature_grid_size=5,
            eval_episodes_per_skill=8,
            seed=9,
        )
        policy, history = train(config)
        evaluation = evaluate(policy, config, seed=99)
        self.assertEqual(policy.shape, (3, 5**4, 9))
        self.assertEqual(len(history), 2)
        self.assertEqual(np.asarray(evaluation["class_rates"]).shape, (3, 3))

    def test_tiny_semantic_spread_training_records_coverage_reward(self) -> None:
        config = PusherTrainConfig(
            representation="semantic_spread",
            envs_per_skill=8,
            iterations=2,
            episode_length=6,
            policy_grid_size=5,
            raw_feature_grid_size=5,
            eval_episodes_per_skill=8,
            seed=12,
        )
        _, history = train(config)
        self.assertIn("coverage_reward_mean", history[-1])
        self.assertEqual(len(history[-1]["coverage_reward_by_skill"]), 3)


if __name__ == "__main__":
    unittest.main()
