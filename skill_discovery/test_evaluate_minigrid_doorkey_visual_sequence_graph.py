"""Tests for frozen DoorKey visual sequence graph auditing."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.evaluate_minigrid_doorkey_visual_sequence_graph import (
    audit_cluster_graph,
    classification_metrics,
)


class DoorKeyVisualSequenceGraphTest(unittest.TestCase):
    def test_classification_metrics_align_frozen_cluster_names(self) -> None:
        clusters = np.asarray([2, 0, 3, 1] * 2)
        stages = np.asarray([0, 1, 2, 3] * 2)
        mapping = {
            "0": "key_acquired",
            "1": "goal_reached",
            "2": "navigation_only",
            "3": "door_opened",
        }
        output = classification_metrics(clusters, stages, mapping)
        self.assertEqual(output["accuracy"], 1.0)
        self.assertTrue(output["all_clusters_nonempty"])

    def test_total_order_graph_passes(self) -> None:
        sequences = [[0, 1, 2, 3] for _ in range(30)]
        output = audit_cluster_graph(
            sequences,
            np.zeros(30, dtype=np.int64),
            np.full(30, 3, dtype=np.int64),
        )
        self.assertEqual(output["ancestor_cardinalities"], [0, 1, 2, 3])
        self.assertTrue(output["graph_gate_passed"])

    def test_supported_cycle_fails_closed(self) -> None:
        sequences = [[0, 1, 2, 3, 0] for _ in range(30)]
        output = audit_cluster_graph(
            sequences,
            np.zeros(30, dtype=np.int64),
            np.full(30, 3, dtype=np.int64),
        )
        self.assertFalse(output["cycle_free"])
        self.assertFalse(output["graph_gate_passed"])


if __name__ == "__main__":
    unittest.main()
