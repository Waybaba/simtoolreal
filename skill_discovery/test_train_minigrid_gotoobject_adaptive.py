"""Tests for reward-deficit GoToObject allocation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from skill_discovery.train_minigrid_gotoobject_adaptive import (
    AdaptiveAllocationConfig,
    load_bootstrap_matrix,
    select_deficit_skill,
    train_adaptive_run,
)


class AdaptiveAllocationTrainerTest(unittest.TestCase):
    def _write_source(self, path: Path, *, gate: bool = True) -> None:
        path.write_text(
            json.dumps(
                {
                    "bootstrap_gate_passed": gate,
                    "bootstrap_calibrated_matrix": [
                        [1.0, -1.0, 0.0],
                        [-1.0, 0.0, 1.0],
                        [0.0, 1.0, -1.0],
                    ],
                }
            ),
            encoding="utf-8",
        )

    def test_source_requires_passed_unique_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.json"
            self._write_source(path)
            matrix = load_bootstrap_matrix(path)
            self.assertEqual(matrix[1], (-1.0, 0.0, 1.0))
            self._write_source(path, gate=False)
            with self.assertRaisesRegex(ValueError, "did not pass"):
                load_bootstrap_matrix(path)

    def test_selector_uses_lowest_reward_ema(self) -> None:
        rng = np.random.default_rng(7)
        selected = select_deficit_skill(np.asarray([0.4, -0.2, 0.1]), rng)
        self.assertEqual(selected, 1)

    def test_tiny_run_preserves_allocation_artifacts(self) -> None:
        config = AdaptiveAllocationConfig(
            seed=113,
            policy_episodes=40,
            horizon=8,
            eval_interval=20,
            eval_episodes_per_skill=2,
            stability_checkpoints=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            output_dir = root / "run"
            self._write_source(source)
            output = train_adaptive_run(config, source, output_dir)
            self.assertEqual(output["core_episode_counts"], [10, 10, 10])
            self.assertEqual(sum(output["extra_episode_counts"]), 10)
            self.assertEqual(sum(output["total_episode_counts"]), 40)
            self.assertEqual(len(output["ema_trace"]), 10)
            self.assertEqual(len(output["evaluations"]), 2)
            self.assertFalse(output["scheduler_reads_semantic_stage"])
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "q_table.npz").exists())


if __name__ == "__main__":
    unittest.main()
