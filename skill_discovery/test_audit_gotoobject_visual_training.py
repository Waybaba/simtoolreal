"""Tests for paired visual/semantic GoToObject training audits."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from skill_discovery.audit_gotoobject_visual_training import compare_q_tables


class GoToObjectVisualTrainingAuditTest(unittest.TestCase):
    def test_q_table_comparison_requires_keys_values_and_visits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            keys = np.asarray([[1, 2, 0, 3, 3, -1, -1, 1]], dtype=np.int16)
            values = np.ones((1, 3, 5), dtype=np.float64)
            visits = np.ones((1, 3, 5), dtype=np.float64)
            np.savez_compressed(root / "left.npz", relation_keys=keys, q_values=values, visits=visits)
            np.savez_compressed(root / "right.npz", relation_keys=keys, q_values=values, visits=visits)
            equal = compare_q_tables(root / "left.npz", root / "right.npz")
            np.savez_compressed(
                root / "changed.npz",
                relation_keys=keys,
                q_values=values + 0.1,
                visits=visits,
            )
            changed = compare_q_tables(root / "left.npz", root / "changed.npz")
        self.assertTrue(equal["exact_equal"])
        self.assertEqual(equal["max_q_abs_difference"], 0.0)
        self.assertFalse(changed["exact_equal"])
        self.assertAlmostEqual(changed["max_q_abs_difference"], 0.1)


if __name__ == "__main__":
    unittest.main()
