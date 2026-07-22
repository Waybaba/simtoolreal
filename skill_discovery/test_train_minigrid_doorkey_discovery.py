"""Tests for two-phase DoorKey spread discovery."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from skill_discovery.train_minigrid_doorkey_discovery import (
    DoorKeyDiscoveryConfig,
    _policy_config,
    _replay_frozen_transitions,
    common_layout_seed,
    resolve_bootstrap,
    train_discovery_run,
)


class DoorKeyDiscoveryTrainerTest(unittest.TestCase):
    def test_common_layout_seed_groups_four_skills_by_phase(self) -> None:
        seeds = [common_layout_seed(7, 500_000, episode) for episode in range(8)]
        self.assertEqual(seeds[:4], [7_500_000] * 4)
        self.assertEqual(seeds[4:], [7_500_001] * 4)

    def test_balanced_assignment_and_transitive_ancestor_floor(self) -> None:
        config = DoorKeyDiscoveryConfig(
            bootstrap_episodes=4,
            policy_episodes=4,
            evaluation_checkpoints=(1, 2, 3, 4),
            minimum_transition_count=25,
        )
        raw_matrix = (
            (3.0, 0.0, 0.0, 4.0),
            (0.0, 3.0, 4.0, 0.0),
            (0.0, 4.0, 3.0, 0.0),
            (3.5, 0.0, 0.0, 4.0),
        )
        state = (1, 1, 0, 2, 2, 0, 3, 2, 0, 1, 3, 3)
        replay_buffer = []
        for episode in range(25):
            skill = episode % 4
            replay_buffer.append(
                [
                    (state, skill, 0, 1, state, (0, 1), False),
                    (state, skill, 1, 2, state, (0, 1), False),
                    (state, skill, 1, 3, None, (), True),
                ]
            )
        resolution = resolve_bootstrap(raw_matrix, replay_buffer, config)
        self.assertTrue(resolution["gate_passed"])
        self.assertTrue(resolution["independent_assignment_collision"])
        self.assertEqual(resolution["assigned_stages"], (3, 2, 1, 0))
        self.assertEqual(
            resolution["transitive_ancestors"],
            ((), (0,), (0, 1), (0, 1, 2)),
        )
        self.assertEqual(
            resolution["reward_matrix"],
            (
                (0.0, 0.0, 0.0, 1.0),
                (0.0, 0.0, 1.0, -1.0),
                (0.0, 1.0, -1.0, -1.0),
                (1.0, -1.0, -1.0, -1.0),
            ),
        )

    def test_reverse_replay_truncates_at_assigned_option_target(self) -> None:
        config = DoorKeyDiscoveryConfig(
            bootstrap_episodes=4,
            policy_episodes=4,
            evaluation_checkpoints=(1, 2, 3, 4),
        )
        assignment = (1, 0, 2, 3)
        matrix = (
            (0.0, 1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0, -1.0),
            (0.0, 0.0, 1.0, -1.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        policy_config = _policy_config(config, assignment, matrix)
        first = (1, 1, 0, 2, 2, 0, 3, 2, 0, 1, 3, 3)
        second = (1, 2, 0, 2, 2, 0, 3, 2, 0, 1, 3, 3)
        third = (1, 2, 1, -1, -1, 1, 3, 2, 0, 1, 3, 3)
        replay_buffer = [
            [
                (first, 0, 0, 0, second, (0, 1), False),
                (second, 0, 1, 1, third, (0, 1), False),
                (third, 0, 1, 2, None, (), True),
            ]
        ]
        q_table = {}
        visits = {}
        summary = _replay_frozen_transitions(
            replay_buffer,
            policy_config,
            q_table,
            visits,
        )
        self.assertEqual(summary["transitions"], 2)
        self.assertEqual(summary["option_truncated_episodes"], 1)
        self.assertAlmostEqual(q_table[second][0, 1], 1.0)
        self.assertAlmostEqual(q_table[first][0, 0], 0.99)
        self.assertNotIn(third, q_table)

    def test_tiny_run_preserves_bootstrap_artifacts(self) -> None:
        config = DoorKeyDiscoveryConfig(
            seed=131,
            bootstrap_episodes=40,
            policy_episodes=40,
            horizon=8,
            evaluation_checkpoints=(20, 30, 40),
            eval_episodes_per_skill=2,
            minimum_transition_count=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "run"
            output = train_discovery_run(config, output_dir)
            self.assertIn("bootstrap_gate_passed", output)
            self.assertEqual(len(output["bootstrap_assigned_stages"]), 4)
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "bootstrap_q_table.npz").exists())
            self.assertTrue((output_dir / "bootstrap_replay_buffer.npz").exists())
            if output["policy_phase_ran"]:
                self.assertTrue((output_dir / "q_table.npz").exists())
                self.assertTrue(
                    (output_dir / "relabelled_bootstrap_q_table.npz").exists()
                )


if __name__ == "__main__":
    unittest.main()
