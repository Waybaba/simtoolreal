"""Tests for mission-free GoToObject tabular skill training."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from skill_discovery.minigrid_gotoobject import make_gotoobject
from skill_discovery.train_minigrid_gotoobject_skills import (
    GoToObjectReward,
    GoToObjectTrainConfig,
    compact_relation_key,
    train_run,
)


class GoToObjectSkillTrainerTest(unittest.TestCase):
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
