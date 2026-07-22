from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.point_cup import LABEL_TO_ID, PointCupEnv, ShapeWorldConfig


class PointCupEnvTest(unittest.TestCase):
    def make_env(self, num_envs: int = 2) -> PointCupEnv:
        return PointCupEnv(
            ShapeWorldConfig(
                num_envs=num_envs,
                episode_length=4,
                action_scale=0.1,
                cup_axes=(0.15, 0.15),
                seed=7,
            )
        )

    def test_entering_and_leaving_labels(self) -> None:
        env = self.make_env()
        env.reset(positions=np.asarray([[0.16, 0.0], [0.14, 0.0]], dtype=np.float32))

        _, labels, _ = env.step(np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32))

        self.assertEqual(labels[0], LABEL_TO_ID["entering"])
        self.assertEqual(labels[1], LABEL_TO_ID["leaving"])

    def test_layout_change_updates_relations(self) -> None:
        env = self.make_env(num_envs=1)
        env.reset(positions=np.asarray([[0.2, 0.0]], dtype=np.float32))
        self.assertEqual(env.observe()["semantic_label"][0], LABEL_TO_ID["outside"])

        env.set_layout(axes=(0.3, 0.1))

        self.assertEqual(env.observe()["semantic_label"][0], LABEL_TO_ID["inside"])

    def test_seeded_reset_is_reproducible(self) -> None:
        first = self.make_env().reset(seed=101)["state"]
        second = self.make_env().reset(seed=101)["state"]
        np.testing.assert_allclose(first, second)

    def test_render_returns_rgb_frames(self) -> None:
        env = self.make_env()
        env.reset(positions=np.asarray([[0.0, 0.0], [0.8, 0.8]], dtype=np.float32))

        frames = env.render(size=48)

        self.assertEqual(frames.shape, (2, 48, 48, 3))
        self.assertEqual(frames.dtype, np.uint8)
        self.assertGreater(np.unique(frames[0].reshape(-1, 3), axis=0).shape[0], 2)


if __name__ == "__main__":
    unittest.main()
