"""Tests for cached FrozenLake visual metric evaluation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from skill_discovery.evaluate_metrics import _farthest_point_sample
from skill_discovery.evaluate_frozenlake_visual_metrics import (
    apply_audit_nuisance,
    evaluate_dataset,
)


class FrozenLakeVisualMetricsTest(unittest.TestCase):
    def test_audit_nuisance_is_deterministic_and_leaves_train_unchanged(self) -> None:
        frames = np.full((4, 3, 8, 8, 3), 100, dtype=np.uint8)
        seeds = np.asarray([7, 17, 37, 47])
        split = np.asarray([0, 0, 1, 1])
        first = apply_audit_nuisance(frames, seeds, split)
        second = apply_audit_nuisance(frames, seeds, split)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[:2], frames[:2])
        self.assertFalse(np.array_equal(first[2:], frames[2:]))

    def test_farthest_point_sampling_never_repeats_tied_points(self) -> None:
        features = np.zeros((6, 3), dtype=np.float32)
        selected = _farthest_point_sample(features, 6)
        self.assertEqual(len(np.unique(selected)), 6)

    def test_semantic_oracle_passes_cross_seed_gate(self) -> None:
        outcomes = np.tile(np.arange(3, dtype=np.int8), 4)
        split = np.repeat(np.asarray([0, 0, 1, 1], dtype=np.int8), 3)
        frames = np.zeros((12, 3, 8, 8, 3), dtype=np.uint8)
        for index, outcome in enumerate(outcomes):
            frames[index, :, :, :, outcome] = 255
        states = np.zeros((12, 4), dtype=np.int16)
        lengths = np.full(12, 3, dtype=np.int16)
        states[np.arange(12), lengths] = outcomes + 5
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.npz"
            np.savez_compressed(
                dataset,
                frames=frames,
                outcomes=outcomes,
                split=split,
                states=states,
                lengths=lengths,
            )
            output = evaluate_dataset(dataset, root / "metrics")
            oracle = output["metrics"]["semantic_oracle"]
            self.assertTrue(output["oracle_gate_passed"])
            self.assertEqual(oracle["cross_seed"]["knn_accuracy"], 1.0)
            self.assertTrue((root / "metrics" / "metric_comparison.svg").exists())


if __name__ == "__main__":
    unittest.main()
