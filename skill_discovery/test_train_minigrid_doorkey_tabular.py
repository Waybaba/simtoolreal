"""Tests for tabular MiniGrid DoorKey controls."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.core.actions import Actions

from skill_discovery.minigrid_doorkey import object_positions, plan_to_face
from skill_discovery.train_minigrid_doorkey_tabular import (
    DOORKEY_CONTROL_MATRIX,
    DoorKeyTabularConfig,
    common_layout_seed,
    compact_doorkey_state,
    state_changing_action_indices,
    train_run,
)


class DoorKeyTabularTrainerTest(unittest.TestCase):
    def test_common_layout_seed_groups_four_skills(self) -> None:
        seeds = [common_layout_seed(7, episode) for episode in range(8)]
        self.assertEqual(seeds[:4], [7_000_000] * 4)
        self.assertEqual(seeds[4:], [7_000_001] * 4)

    def test_compact_state_tracks_key_carrying(self) -> None:
        env = gym.make("MiniGrid-DoorKey-5x5-v0")
        try:
            env.reset(seed=7)
            before = compact_doorkey_state(env)
            actions = plan_to_face(env, object_positions(env)["key"])
            for action in actions:
                env.step(action)
            env.step(int(Actions.pickup))
            after = compact_doorkey_state(env)
        finally:
            env.close()
        self.assertEqual(len(before), 12)
        self.assertEqual(before[5], 0)
        self.assertNotEqual(before[3:5], (-1, -1))
        self.assertEqual(after[3:5], (-1, -1))
        self.assertEqual(after[5], 1)

    def test_reward_matrix_protects_all_target_ancestors(self) -> None:
        self.assertEqual(DOORKEY_CONTROL_MATRIX[2], (0.0, 0.0, 1.0, -1.0))
        self.assertEqual(DOORKEY_CONTROL_MATRIX[3], (0.0, 0.0, 0.0, 1.0))

    def test_action_mask_removes_pickup_after_key_is_carried(self) -> None:
        env = gym.make("MiniGrid-DoorKey-5x5-v0")
        try:
            env.reset(seed=7)
            actions = plan_to_face(env, object_positions(env)["key"])
            for action in actions:
                env.step(action)
            before = state_changing_action_indices(env)
            env.step(int(Actions.pickup))
            after = state_changing_action_indices(env)
        finally:
            env.close()
        self.assertIn(3, before)
        self.assertNotIn(3, after)

    def test_tiny_run_writes_control_artifacts(self) -> None:
        config = DoorKeyTabularConfig(
            seed=127,
            episodes=40,
            horizon=8,
            evaluation_checkpoints=(20, 30, 40),
            eval_episodes_per_skill=2,
            valid_action_mask=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "run"
            output = train_run(config, output_dir)
            self.assertEqual(len(output["evaluations"]), 3)
            self.assertEqual(sum(map(sum, output["terminal_counts"])), 40)
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "q_table.npz").exists())
            self.assertTrue((output_dir / "policy_rollout_audit.png").exists())


if __name__ == "__main__":
    unittest.main()
