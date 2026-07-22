"""Tests for MountainCar frame-pair visual metric helpers."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.evaluate_mountaincar_frame_pair_metric import (
    car_x_pair_features,
    classification_metrics,
    nearest_neighbor_predictions,
)


class MountainCarFramePairMetricTest(unittest.TestCase):
    def test_car_x_feature_recovers_ordered_motion(self) -> None:
        background = np.full((400, 600, 3), 255, dtype=np.uint8)
        frames = np.repeat(background[None, None], 2, axis=0)
        frames = np.repeat(frames, 2, axis=1)
        frames[0, 0, 100:110, 100:110] = 0
        frames[0, 1, 100:110, 120:130] = 0
        frames[1, 0, 100:110, 300:310] = 0
        frames[1, 1, 100:110, 280:290] = 0
        features, car_x = car_x_pair_features(frames, background)
        self.assertEqual(features.shape, (2, 3))
        self.assertGreater(car_x[0, 1] - car_x[0, 0], 0)
        self.assertLess(car_x[1, 1] - car_x[1, 0], 0)

    def test_reference_one_nn_and_metrics(self) -> None:
        features = np.asarray([[0.0], [1.0], [0.1], [0.9]], dtype=np.float32)
        classes = np.asarray([0, 1, 0, 1], dtype=np.int8)
        splits = np.asarray([0, 0, 1, 1], dtype=np.int8)
        predictions = nearest_neighbor_predictions(features, classes, splits)
        metrics = classification_metrics(predictions[2:], classes[2:])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["predicted_class_sizes"][:2], [1, 1])


if __name__ == "__main__":
    unittest.main()
