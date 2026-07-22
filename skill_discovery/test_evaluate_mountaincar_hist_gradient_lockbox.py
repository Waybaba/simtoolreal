"""Tests for the frozen MountainCar lockbox classifier."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.evaluate_mountaincar_hist_gradient_lockbox import (
    build_classifier,
)


class MountainCarHistGradientLockboxTest(unittest.TestCase):
    def test_classifier_uses_frozen_parameters(self) -> None:
        params = build_classifier().get_params()
        self.assertEqual(params["learning_rate"], 0.08)
        self.assertEqual(params["max_iter"], 200)
        self.assertEqual(params["max_leaf_nodes"], 15)
        self.assertEqual(params["max_depth"], 6)
        self.assertEqual(params["min_samples_leaf"], 12)
        self.assertFalse(params["early_stopping"])
        self.assertEqual(params["random_state"], 18_100_007)

    def test_classifier_fits_five_classes(self) -> None:
        rng = np.random.default_rng(1)
        features = np.concatenate(
            [rng.normal(class_index, 0.01, size=(20, 3)) for class_index in range(5)]
        )
        labels = np.repeat(np.arange(-1, 4), 20)
        classifier = build_classifier()
        classifier.fit(features, labels)
        self.assertEqual(classifier.classes_.tolist(), [-1, 0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
