"""Tests for the frozen MountainCar visual decision tree."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.evaluate_mountaincar_visual_decision_tree import (
    VisualDecisionTreeConfig,
    build_classifier,
)


class MountainCarVisualDecisionTreeTest(unittest.TestCase):
    def test_classifier_uses_frozen_parameters(self) -> None:
        classifier = build_classifier(VisualDecisionTreeConfig())
        params = classifier.get_params()
        self.assertEqual(params["criterion"], "entropy")
        self.assertEqual(params["max_depth"], 8)
        self.assertEqual(params["min_samples_leaf"], 8)
        self.assertIsNone(params["class_weight"])
        self.assertEqual(params["random_state"], 10_100_007)

    def test_classifier_fits_visual_feature_rows(self) -> None:
        classifier = build_classifier(
            VisualDecisionTreeConfig(min_samples_leaf=1)
        )
        features = np.asarray(
            [[-1.0, -1.1, -0.1], [0.0, 0.1, 0.1], [0.5, 0.5, 0.0]],
            dtype=np.float32,
        )
        labels = np.asarray([0, 2, -1], dtype=np.int8)
        classifier.fit(features, labels)
        self.assertEqual(classifier.predict(features).tolist(), [0, 2, -1])

    def test_tree_budget_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            VisualDecisionTreeConfig(max_depth=0)


if __name__ == "__main__":
    unittest.main()
