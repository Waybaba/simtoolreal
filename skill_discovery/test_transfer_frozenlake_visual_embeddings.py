"""Tests for FrozenLake visual layout transfer metrics."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.transfer_frozenlake_visual_embeddings import (
    classification_metrics,
)


class FrozenLakeVisualTransferTest(unittest.TestCase):
    def test_transfer_gate_requires_every_outcome(self) -> None:
        outcomes = np.tile(np.arange(3), 10)
        seeds = np.repeat((7, 17, 27), 10)
        passing = classification_metrics(outcomes.copy(), outcomes, seeds)
        self.assertTrue(passing["gate_passed"])
        predictions = outcomes.copy()
        predictions[outcomes == 2] = 1
        failing = classification_metrics(predictions, outcomes, seeds)
        self.assertEqual(failing["accuracy"], 2.0 / 3.0)
        self.assertEqual(failing["recall_by_outcome"]["goal_terminal"], 0.0)
        self.assertFalse(failing["gate_passed"])

    def test_per_seed_confusions_preserve_all_examples(self) -> None:
        outcomes = np.asarray([0, 1, 2, 0, 1, 2])
        predictions = np.asarray([0, 1, 2, 2, 1, 2])
        seeds = np.asarray([7, 7, 7, 17, 17, 17])
        metrics = classification_metrics(predictions, outcomes, seeds)
        self.assertEqual(metrics["per_seed"]["7"]["accuracy"], 1.0)
        self.assertEqual(metrics["per_seed"]["17"]["accuracy"], 2.0 / 3.0)
        self.assertEqual(np.asarray(metrics["confusion"]).sum(), 6)


if __name__ == "__main__":
    unittest.main()
