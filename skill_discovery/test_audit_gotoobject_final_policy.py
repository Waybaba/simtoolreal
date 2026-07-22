"""Tests for frozen GoToObject final-policy audits."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from skill_discovery.audit_gotoobject_final_policy import (
    SeedBlockAuditConfig,
    audit_final_policy,
    load_saved_q_table,
)


class FinalPolicyAuditTest(unittest.TestCase):
    def _write_run(self, run_dir: Path) -> None:
        run_dir.mkdir()
        metrics = {
            "policy_phase_ran": True,
            "config": {
                "seed": 7,
                "policy_episodes": 8,
                "horizon": 4,
                "gamma": 0.99,
                "eval_interval": 4,
                "stability_checkpoints": 2,
            },
            "bootstrap_calibrated_matrix": [
                [1.0, -1.0, 0.0],
                [-1.0, 0.0, 1.0],
                [0.0, 1.0, -1.0],
            ],
        }
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics),
            encoding="utf-8",
        )
        np.savez_compressed(
            run_dir / "q_table.npz",
            relation_keys=np.asarray([(1, 1, 0, 2, 2, 3, 3, 0)]),
            q_values=np.zeros((1, 3, 5)),
            visits=np.zeros((1, 3, 5)),
        )

    def test_loader_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "q.npz"
            keys = np.asarray([(1, 1, 0, 2, 2, 3, 3, 0)] * 2)
            np.savez_compressed(
                path,
                relation_keys=keys,
                q_values=np.zeros((2, 3, 5)),
                visits=np.zeros((2, 3, 5)),
            )
            with self.assertRaisesRegex(ValueError, "duplicates"):
                load_saved_q_table(path)

    def test_tiny_audit_is_evaluation_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            self._write_run(run_dir)
            original = np.load(run_dir / "q_table.npz")["q_values"].copy()
            output = audit_final_policy(
                run_dir,
                SeedBlockAuditConfig(blocks=2, eval_episodes_per_skill=2),
            )
            current = np.load(run_dir / "q_table.npz")["q_values"]
            np.testing.assert_array_equal(current, original)
            self.assertEqual(len(output["blocks"]), 2)
            self.assertFalse(output["changes_training_gate_result"])
            self.assertTrue(
                (run_dir / "final_policy_seed_block_audit.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
