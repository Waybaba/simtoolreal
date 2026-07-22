"""Tests for complete-domain Taxi visual metric evaluation."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.evaluate_taxi_visual_metric import (
    fit_reference_centers,
    reference_method,
    taxi_classification_metrics,
)


class TaxiVisualMetricTest(unittest.TestCase):
    def test_reference_centers_are_class_means(self) -> None:
        features = np.asarray(
            [[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0], [-1.0, 0.0]],
            dtype=np.float32,
        )
        stages = np.asarray([0, 0, 1, 1, 2])
        centers = fit_reference_centers(features, stages)
        np.testing.assert_allclose(centers, [[1, 0], [0, 1], [-1, 0]])

    def test_metrics_report_destination_and_orientation_groups(self) -> None:
        expected = np.asarray([0, 1, 2, 0, 1, 2])
        output = taxi_classification_metrics(
            expected.copy(),
            expected,
            np.asarray([1, 1, 1, 3, 3, 3]),
            np.asarray([1, 3, 1, 3, 1, 3]),
        )
        self.assertEqual(output["accuracy"], 1.0)
        self.assertEqual(output["by_destination"]["1"]["accuracy"], 1.0)
        self.assertEqual(output["by_orientation"]["3"]["accuracy"], 1.0)

    def test_reference_method_uses_only_split_zero_for_centers(self) -> None:
        features = np.tile(np.eye(3, dtype=np.float32), (4, 1))
        stages = np.tile(np.arange(3), 4)
        splits = np.repeat(np.arange(4), 3)
        metrics, _, predictions = reference_method(
            features,
            stages,
            np.repeat(np.arange(4), 3),
            np.repeat(np.arange(4), 3),
            splits,
        )
        np.testing.assert_array_equal(predictions, stages)
        self.assertEqual(metrics["joint_audit"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
