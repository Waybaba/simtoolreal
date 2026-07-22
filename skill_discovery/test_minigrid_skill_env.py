"""Tests for MiniGrid skill observations and intrinsic rewards."""

from __future__ import annotations

import unittest

import numpy as np
from stable_baselines3.common.env_checker import check_env

from skill_discovery.minigrid_skill_env import (
    DoorKeySkillWrapper,
    MiniGridSkillConfig,
    OnlineIntrinsicReward,
)


class MiniGridSkillEnvTest(unittest.TestCase):
    def test_wrapper_passes_sb3_checker_and_encodes_skill(self) -> None:
        config = MiniGridSkillConfig()
        reward_model = OnlineIntrinsicReward("semantic_spread", config)
        first = DoorKeySkillWrapper(config, 0, reward_model)
        second = DoorKeySkillWrapper(config, 3, reward_model)
        try:
            check_env(first, warn=True)
            first_obs, _ = first.reset(seed=7)
            second_obs, _ = second.reset(seed=7)
            self.assertEqual(first_obs.shape, first.observation_space.shape)
            np.testing.assert_allclose(first_obs[:-4], second_obs[:-4])
            self.assertFalse(bool(np.array_equal(first_obs[-4:], second_obs[-4:])))
        finally:
            first.close()
            second.close()

    def test_native_reward_never_leaks_into_training_reward(self) -> None:
        config = MiniGridSkillConfig()
        reward_model = OnlineIntrinsicReward("random", config)
        env = DoorKeySkillWrapper(config, 0, reward_model)
        try:
            env.reset(seed=7)
            terminated = truncated = False
            while not (terminated or truncated):
                _, reward, terminated, truncated, info = env.step(env.action_space.sample())
                self.assertEqual(reward, 0.0)
            self.assertIn("native_reward", info)
            self.assertIn("furthest_stage", info)
        finally:
            env.close()

    def test_semantic_spread_rewards_rare_stage_without_target_mapping(self) -> None:
        config = MiniGridSkillConfig(semantic_decay=1.0)
        reward_model = OnlineIntrinsicReward("semantic_spread", config)
        common_rewards = []
        for _ in range(32):
            reward, _ = reward_model.reward(skill=0, stage=0, raw_feature=0)
            common_rewards.append(reward)
        rare_reward, parts = reward_model.reward(skill=1, stage=3, raw_feature=1)
        self.assertGreater(parts["coverage_reward"], 0.0)
        self.assertGreater(rare_reward, common_rewards[-1])
        self.assertEqual(reward_model.episode_counts.tolist(), [32, 1, 0, 0])

    def test_balanced_oracle_targets_are_seeded_and_not_observed(self) -> None:
        config = MiniGridSkillConfig(seed=19)
        first = OnlineIntrinsicReward("semantic_balanced", config)
        second = OnlineIntrinsicReward("semantic_balanced", config)
        np.testing.assert_array_equal(first.balanced_targets, second.balanced_targets)
        self.assertEqual(sorted(first.balanced_targets.tolist()), [0, 1, 2, 3])
        for skill, target in enumerate(first.balanced_targets):
            positive, _ = first.reward(skill, int(target), raw_feature=0)
            negative, _ = first.reward(skill, int((target + 1) % 4), raw_feature=0)
            self.assertEqual(positive, 1.0)
            self.assertEqual(negative, 0.0)


if __name__ == "__main__":
    unittest.main()
