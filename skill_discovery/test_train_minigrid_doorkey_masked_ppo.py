"""Tests for masked neural DoorKey controls."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from minigrid.core.actions import Actions

from skill_discovery.minigrid_doorkey import object_positions, plan_to_face
from skill_discovery.train_minigrid_doorkey_masked_ppo import (
    MaskedDoorKeyControlEnv,
    MaskedPPOConfig,
    train_run,
)


class MaskedDoorKeyPPOTest(unittest.TestCase):
    def test_action_mask_keeps_only_state_changing_control_actions(self) -> None:
        config = MaskedPPOConfig(total_timesteps=150_000)
        env = MaskedDoorKeyControlEnv(config, skill_id=1)
        try:
            env.reset(seed=7)
            mask = env.action_masks()
            self.assertEqual(mask.shape, (7,))
            self.assertTrue(mask[int(Actions.left)])
            self.assertTrue(mask[int(Actions.right)])
            self.assertFalse(mask[int(Actions.drop)])
            self.assertFalse(mask[int(Actions.done)])
            for action in plan_to_face(env, object_positions(env)["key"]):
                self.assertTrue(env.action_masks()[action])
                env.step(action)
            self.assertTrue(env.action_masks()[int(Actions.pickup)])
        finally:
            env.close()

    def test_key_target_returns_option_state(self) -> None:
        config = MaskedPPOConfig(total_timesteps=150_000)
        env = MaskedDoorKeyControlEnv(config, skill_id=1)
        try:
            env.reset(seed=7)
            for action in plan_to_face(env, object_positions(env)["key"]):
                env.step(action)
            _, reward, terminated, truncated, info = env.step(int(Actions.pickup))
            self.assertEqual(reward, 1.0)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["option_terminated"])
            self.assertFalse(info["native_success"])
            self.assertEqual(info["carrying"], ["key", "yellow"])
            self.assertFalse(info["door_open"])
        finally:
            env.close()

    def test_random_valid_actions_never_raise_mask_error(self) -> None:
        config = MaskedPPOConfig(total_timesteps=150_000)
        env = MaskedDoorKeyControlEnv(config, skill_id=3)
        rng = np.random.default_rng(17)
        try:
            env.reset(seed=17)
            for _ in range(config.horizon):
                valid = np.flatnonzero(env.action_masks())
                _, _, terminated, truncated, _ = env.step(int(rng.choice(valid)))
                if terminated or truncated:
                    break
        finally:
            env.close()

    def test_tiny_training_writes_policy_and_audit(self) -> None:
        config = MaskedPPOConfig(
            seed=149,
            n_envs=4,
            total_timesteps=64,
            horizon=8,
            n_steps=8,
            batch_size=16,
            n_epochs=1,
            net_arch=(32, 32),
            eval_interval=32,
            checkpoint_eval_episodes_per_skill=1,
            final_eval_episodes_per_skill=1,
            stability_checkpoints=2,
            torch_threads=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "run"
            output = train_run(config, output_dir)
            self.assertEqual(len(output["evaluations"]), 2)
            self.assertTrue((output_dir / "masked_ppo_policy.zip").exists())
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "policy_rollout_audit.png").exists())
            self.assertTrue((output_dir / "stage_curves.svg").exists())


if __name__ == "__main__":
    unittest.main()
