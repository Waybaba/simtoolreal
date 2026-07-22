"""Tests for MountainCar natural-rollout rejection helpers."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.audit_mountaincar_rollout_rejection import (
    leave_one_out_thresholds,
    oracle_relation,
    predict_nearest,
    predict_with_rejection,
)


class MountainCarRolloutRejectionTest(unittest.TestCase):
    def test_leave_one_out_thresholds_and_rejection(self) -> None:
        features = np.asarray(
            [[0.0], [0.1], [1.0], [1.1], [2.0], [2.1], [3.0], [3.1]],
            dtype=np.float32,
        )
        classes = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int8)
        splits = np.zeros(8, dtype=np.int8)
        thresholds, reference, reference_classes = leave_one_out_thresholds(
            features, classes, splits
        )
        predictions, _ = predict_with_rejection(
            np.asarray([[0.05], [1.05], [5.0]], dtype=np.float32),
            reference,
            reference_classes,
            thresholds,
        )
        self.assertEqual(predictions.tolist(), [0, 1, -1])

    def test_oracle_relation_includes_none(self) -> None:
        self.assertEqual(oracle_relation(np.asarray([-0.9, -0.02]), False), 0)
        self.assertEqual(oracle_relation(np.asarray([-0.5, 0.0]), False), -1)
        self.assertEqual(oracle_relation(np.asarray([0.46, 0.02]), True), 3)

    def test_nearest_prediction_supports_explicit_none_class(self) -> None:
        predictions, distances = predict_nearest(
            np.asarray([[0.05], [1.1], [2.95]], dtype=np.float32),
            np.asarray([[0.0], [1.0], [3.0]], dtype=np.float32),
            np.asarray([-1, 0, 3], dtype=np.int8),
        )
        self.assertEqual(predictions.tolist(), [-1, 0, 3])
        np.testing.assert_allclose(distances, [0.0025, 0.01, 0.0025], atol=1.0e-8)


if __name__ == "__main__":
    unittest.main()
