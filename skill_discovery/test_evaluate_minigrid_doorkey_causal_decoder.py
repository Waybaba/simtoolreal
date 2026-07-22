"""Tests for the unlabeled DoorKey causal ordered-cluster decoder."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.evaluate_minigrid_doorkey_causal_decoder import (
    causal_decode,
    decoded_stage_metrics,
    infer_cluster_order,
)


class DoorKeyCausalDecoderTest(unittest.TestCase):
    def test_unlabeled_order_uses_net_forward_transition_mass(self) -> None:
        sequences = [[2, 0, 3, 1] for _ in range(40)]
        sequences.extend([[2, 0, 3, 0, 3, 1] for _ in range(5)])
        output = infer_cluster_order(sequences)
        self.assertEqual(output["selected_cluster_order"], [2, 0, 3, 1])
        self.assertFalse(output["oracle_labels_used"])

    def test_multiple_reset_clusters_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "one root"):
            infer_cluster_order([[0, 1], [2, 1]])

    def test_causal_decoder_ignores_backward_and_skip_observations(self) -> None:
        decoded = causal_decode(
            [2, 2, 3, 0, 1, 3, 0, 1],
            (2, 0, 3, 1),
        )
        self.assertEqual(decoded, [0, 0, 0, 1, 1, 2, 2, 3])

    def test_metrics_report_fresh_sequence_safety(self) -> None:
        decoded = [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]] * 10
        oracle = [sequence.copy() for sequence in decoded]
        targets = np.asarray([0, 1, 2, 3] * 10)
        success = np.asarray([False, False, False, True] * 10)
        groups = np.asarray([117] * 20 + [127] * 20)
        output = decoded_stage_metrics(decoded, oracle, targets, success, groups)
        self.assertEqual(output["accuracy"], 1.0)
        self.assertEqual(output["goal_native_terminal_recall"], 1.0)
        self.assertEqual(output["non_goal_false_goal_sequence_rate"], 0.0)
        self.assertEqual(output["adjacent_transition_counts"], [30, 20, 10])


if __name__ == "__main__":
    unittest.main()
