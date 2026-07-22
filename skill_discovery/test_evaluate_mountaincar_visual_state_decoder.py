"""Tests for the MountainCar visual state decoder."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.evaluate_mountaincar_visual_state_decoder import (
    build_decoder,
    decoded_relations,
)


class MountainCarVisualStateDecoderTest(unittest.TestCase):
    def test_decoder_uses_frozen_polynomial_and_ridge(self) -> None:
        decoder = build_decoder()
        polynomial = decoder.named_steps["polynomial"]
        ridge = decoder.named_steps["ridge"]
        self.assertEqual(polynomial.degree, 3)
        self.assertFalse(polynomial.include_bias)
        self.assertEqual(ridge.alpha, 1.0e-8)
        self.assertEqual(ridge.solver, "svd")

    def test_decoded_relations_include_explicit_none(self) -> None:
        predictions = decoded_relations(
            np.asarray([-1.0, -0.4, 0.2, 0.46, -0.4]),
            np.asarray([-0.006, 0.006, 0.006, 0.01, 0.0]),
        )
        self.assertEqual(predictions.tolist(), [0, 1, 2, 3, -1])

    def test_relation_threshold_boundaries_are_frozen(self) -> None:
        predictions = decoded_relations(
            np.asarray([-0.75, -0.75, 0.0, 0.45]),
            np.asarray([-0.005, 0.005, 0.005, 0.0]),
        )
        self.assertEqual(predictions.tolist(), [0, -1, 2, 3])


if __name__ == "__main__":
    unittest.main()
