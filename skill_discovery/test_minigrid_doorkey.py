"""Tests for the official MiniGrid DoorKey audit adapter."""

from __future__ import annotations

import unittest

import gymnasium as gym

from skill_discovery.minigrid_doorkey import (
    DOORKEY_STAGES,
    reproducibility_signature,
    solve_doorkey_episode,
)


class MiniGridDoorKeyTest(unittest.TestCase):
    def test_registered_environments_reset_render_and_reproduce(self) -> None:
        for env_id, render_size in (
            ("MiniGrid-DoorKey-5x5-v0", 160),
            ("MiniGrid-DoorKey-8x8-v0", 256),
        ):
            env = gym.make(env_id, render_mode="rgb_array")
            try:
                first = reproducibility_signature(env, 7)
                second = reproducibility_signature(env, 7)
                self.assertEqual(first, second)
                observation, _ = env.reset(seed=7)
                self.assertEqual(observation["image"].shape, (7, 7, 3))
                self.assertEqual(env.action_space.n, 7)
                self.assertEqual(env.render().shape, (render_size, render_size, 3))
            finally:
                env.close()

    def test_scripted_solver_visits_all_stages(self) -> None:
        for env_id in ("MiniGrid-DoorKey-5x5-v0", "MiniGrid-DoorKey-8x8-v0"):
            env = gym.make(env_id, render_mode="rgb_array")
            try:
                result = solve_doorkey_episode(env, seed=17)
                self.assertEqual(sorted(set(result.stage_sequence)), list(range(len(DOORKEY_STAGES))))
                self.assertTrue(result.terminated)
                self.assertFalse(result.truncated)
                self.assertGreater(result.reward, 0.0)
                self.assertEqual(len(result.frames), 4)
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
