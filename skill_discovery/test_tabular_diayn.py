from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.train_tabular_diayn import TrainConfig, _mutual_information, evaluate, train


class TabularDiaynTest(unittest.TestCase):
    def test_mutual_information_extremes(self) -> None:
        skills = np.asarray([0, 0, 1, 1], dtype=np.int64)
        self.assertAlmostEqual(_mutual_information(skills, skills, 2, 2), 1.0)
        self.assertAlmostEqual(
            _mutual_information(skills, np.asarray([0, 1, 0, 1]), 2, 2),
            0.0,
        )

    def test_tiny_training_and_evaluation(self) -> None:
        config = TrainConfig(
            representation="semantic",
            seed=3,
            envs_per_skill=8,
            iterations=2,
            episode_length=6,
            policy_grid_size=9,
            raw_feature_grid_size=5,
            eval_episodes_per_skill=8,
        )
        policy, history = train(config)
        metrics = evaluate(policy, config, seed=103)

        self.assertEqual(policy.shape, (2, 9, 9, 9))
        self.assertEqual(len(history), 2)
        self.assertEqual(len(metrics["inside_rates"]), 2)
        self.assertTrue(0.0 <= metrics["semantic_mi_bits"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
