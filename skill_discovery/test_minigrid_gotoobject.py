"""Tests for mission-independent MiniGrid GoToObject stages."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.minigrid_gotoobject import (
    make_gotoobject,
    reproducibility_signature,
    scripted_relation_audit,
)


class MiniGridGoToObjectTest(unittest.TestCase):
    def test_mission_free_full_observation_and_seed_reproducibility(self) -> None:
        env = make_gotoobject(mission_free=True)
        raw = make_gotoobject()
        try:
            observation, _ = env.reset(seed=7)
            self.assertIsInstance(observation, np.ndarray)
            self.assertEqual(observation.shape, (6, 6, 3))
            self.assertNotIsInstance(env.observation_space, dict)
            self.assertEqual(
                reproducibility_signature(raw, 7),
                reproducibility_signature(raw, 7),
            )
        finally:
            env.close()
            raw.close()

    def test_scripted_relations_pick_up_real_object_without_mission(self) -> None:
        env = make_gotoobject(render_mode="rgb_array")
        try:
            result = scripted_relation_audit(env, seed=7)
            self.assertEqual(result.stages, (0, 1, 2))
            self.assertEqual(result.floor_counts, (2, 2, 1))
            self.assertEqual(
                result.carrying,
                (result.selected_object.object_type, result.selected_object.color),
            )
            self.assertFalse(result.selected_is_mission_target)
            self.assertEqual(len(result.frames), 3)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
