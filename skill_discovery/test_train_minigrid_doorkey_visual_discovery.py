"""Tests for causal visual-stage DoorKey discovery helpers."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.train_minigrid_doorkey_visual_discovery import (
    CausalVisualStageFactory,
    RenderedClusterLookup,
)


class IndexedFrameEncoder:
    def __init__(self):
        self.calls = 0

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.eye(4, dtype=np.float32)[int(frame[0, 0, 0])]


class SequenceLookup:
    def __init__(self, clusters: list[int]):
        self.clusters = iter(clusters)

    def query(self, env: object) -> int:
        del env
        return next(self.clusters)

    def summary(self) -> dict[str, object]:
        return {}


class DoorKeyVisualDiscoveryTest(unittest.TestCase):
    def test_rendered_lookup_encodes_each_compact_state_once(self) -> None:
        encoder = IndexedFrameEncoder()
        lookup = RenderedClusterLookup(np.eye(4, dtype=np.float32), encoder)
        key = tuple(range(12))
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[0, 0, 0] = 2
        self.assertEqual(lookup.cluster(key, frame), 2)
        self.assertEqual(lookup.cluster(key, frame.copy()), 2)
        self.assertEqual(encoder.calls, 1)
        self.assertEqual(lookup.query_count, 2)

    def test_rendered_lookup_rgb_alias_fails_closed(self) -> None:
        lookup = RenderedClusterLookup(
            np.eye(4, dtype=np.float32),
            IndexedFrameEncoder(),
        )
        key = tuple(range(12))
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        lookup.cluster(key, frame)
        frame[1, 1] = 255
        with self.assertRaisesRegex(ValueError, "multiple RGB"):
            lookup.cluster(key, frame)
        self.assertEqual(lookup.alias_count, 1)

    def test_tracker_is_causal_and_ignores_skip_and_backward_clusters(self) -> None:
        lookup = SequenceLookup([2, 3, 0, 1, 3, 0, 1])
        factory = CausalVisualStageFactory(lookup)  # type: ignore[arg-type]
        tracker = factory()
        decoded = [tracker.observe(object()) for _ in range(7)]
        self.assertEqual(decoded, [0, 0, 1, 1, 2, 2, 3])
        self.assertEqual(factory.decoded_query_counts.tolist(), [2, 2, 2, 1])

    def test_tracker_rejects_nonroot_reset(self) -> None:
        lookup = SequenceLookup([0])
        factory = CausalVisualStageFactory(lookup)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "root"):
            factory().observe(object())


if __name__ == "__main__":
    unittest.main()
