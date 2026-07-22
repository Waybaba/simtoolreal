"""Tests for the explicit MountainCar none-class evaluator."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.evaluate_mountaincar_explicit_none import five_class_metrics


class MountainCarExplicitNoneTest(unittest.TestCase):
    def test_five_class_metrics_include_none(self) -> None:
        oracle = np.asarray([-1, -1, 0, 1, 2, 3], dtype=np.int8)
        predictions = np.asarray([-1, 0, 0, 1, 2, 3], dtype=np.int8)
        metrics = five_class_metrics(oracle, predictions)
        self.assertAlmostEqual(metrics["accuracy"], 5.0 / 6.0)
        self.assertAlmostEqual(metrics["macro_recall"], 0.9)
        self.assertEqual(metrics["recall_by_class"]["none"], 0.5)
        self.assertEqual(metrics["predicted_class_sizes"]["native_goal"], 1)

    def test_five_class_metrics_reject_invalid_labels(self) -> None:
        with self.assertRaises(ValueError):
            five_class_metrics(
                np.asarray([-2], dtype=np.int8),
                np.asarray([-1], dtype=np.int8),
            )


if __name__ == "__main__":
    unittest.main()
