"""Tests for the deterministic Pusher-Cup contact rules."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.pusher_cup import PusherCupConfig, PusherCupEnv


class PusherCupTest(unittest.TestCase):
    def test_ball_only_moves_after_contact(self) -> None:
        env = PusherCupEnv(PusherCupConfig(num_envs=2, episode_length=20, seed=3))
        starts = env.ball_positions.copy()
        actions = np.asarray(((0.0, 1.0), (1.0, 0.0)), dtype=np.float32)
        for _ in range(4):
            env.step(actions)

        np.testing.assert_allclose(env.ball_positions[0], starts[0])
        self.assertGreater(env.ball_positions[1, 0], starts[1, 0])
        self.assertEqual(env.trajectory_classes().tolist(), [0, 1])

    def test_scripted_push_reaches_cup(self) -> None:
        env = PusherCupEnv(PusherCupConfig(num_envs=1, episode_length=20, seed=5))
        for _ in range(12):
            _, classes, _ = env.step(np.asarray(((1.0, 0.0),), dtype=np.float32))

        self.assertEqual(int(classes[0]), 2)
        self.assertTrue(bool(env.observe()["relation_features"][0, 2]))
        self.assertGreater(float(env.ball_path_length[0]), 0.4)

    def test_moving_away_does_not_pull_ball(self) -> None:
        config = PusherCupConfig(num_envs=1, episode_length=4)
        env = PusherCupEnv(config)
        pusher = np.asarray(((-0.34, 0.0),), dtype=np.float32)
        ball = np.asarray(((-0.25, 0.0),), dtype=np.float32)
        env.reset(pusher_positions=pusher, ball_positions=ball)
        env.step(np.asarray(((-1.0, 0.0),), dtype=np.float32))

        np.testing.assert_allclose(env.ball_positions, ball)
        self.assertFalse(bool(env.ever_contact[0]))

    def test_per_environment_layout_and_render(self) -> None:
        env = PusherCupEnv(PusherCupConfig(num_envs=2, episode_length=2))
        centers = np.asarray(((0.35, 0.0), (-0.25, 0.0)), dtype=np.float32)
        axes = np.asarray(((0.14, 0.16), (0.10, 0.22)), dtype=np.float32)
        env.set_layout(centers=centers, axes=axes)

        classes = env.trajectory_classes()
        self.assertEqual(classes.tolist(), [0, 2])
        frames = env.render(size=48)
        self.assertEqual(frames.shape, (2, 48, 48, 3))
        self.assertGreater(int(np.unique(frames.reshape(-1, 3), axis=0).shape[0]), 3)

    def test_world_boundary_clipping(self) -> None:
        env = PusherCupEnv(PusherCupConfig(num_envs=1, episode_length=2))
        env.reset(
            pusher_positions=np.asarray(((0.99, 0.99),), dtype=np.float32),
            ball_positions=np.asarray(((-0.25, 0.0),), dtype=np.float32),
        )
        env.step(np.asarray(((1.0, 1.0),), dtype=np.float32))
        np.testing.assert_allclose(env.pusher_positions, ((1.0, 1.0),))


if __name__ == "__main__":
    unittest.main()
