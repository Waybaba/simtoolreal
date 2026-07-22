"""Tests for the official FrozenLake outcome bridge."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.frozenlake import (
    FROZENLAKE_OUTCOMES,
    SCRIPTED_ACTIONS,
    FrozenLakeConfig,
    make_frozenlake,
    rollout_actions,
)


class FrozenLakeBridgeTest(unittest.TestCase):
    def test_registered_environment_reproduces_and_renders(self) -> None:
        config = FrozenLakeConfig()
        env = make_frozenlake(config, render_mode="rgb_array")
        try:
            first_state, _ = env.reset(seed=7)
            first_frame = env.render()
            second_state, _ = env.reset(seed=7)
            second_frame = env.render()
            self.assertEqual(int(env.action_space.n), 4)
            self.assertEqual(int(env.observation_space.n), 16)
            self.assertEqual(int(first_state), int(second_state))
            np.testing.assert_array_equal(first_frame, second_frame)
            self.assertEqual(first_frame.shape, (256, 256, 3))
        finally:
            env.close()

    def test_scripted_actions_reach_each_outcome(self) -> None:
        config = FrozenLakeConfig()
        for expected, name in enumerate(FROZENLAKE_OUTCOMES):
            result = rollout_actions(
                config,
                SCRIPTED_ACTIONS[name],
                seed=7,
                render=True,
            )
            self.assertEqual(result.outcome, expected)
            self.assertGreater(len(result.frames), 1)
        safe = rollout_actions(config, SCRIPTED_ACTIONS["safe_timeout"], seed=7)
        hole = rollout_actions(config, SCRIPTED_ACTIONS["hole_terminal"], seed=7)
        goal = rollout_actions(config, SCRIPTED_ACTIONS["goal_terminal"], seed=7)
        self.assertTrue(safe.truncated)
        self.assertTrue(hole.terminated)
        self.assertEqual(hole.native_reward, 0.0)
        self.assertTrue(goal.terminated)
        self.assertEqual(goal.native_reward, 1.0)


if __name__ == "__main__":
    unittest.main()
