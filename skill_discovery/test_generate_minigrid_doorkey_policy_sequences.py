"""Tests for the unique-state DoorKey policy sequence generator."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.generate_minigrid_doorkey_policy_sequences import (
    PolicySequenceDatasetConfig,
    UniqueRenderedStateCache,
)


class DoorKeyPolicySequenceDatasetTest(unittest.TestCase):
    def test_generation_groups_must_be_unique(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            PolicySequenceDatasetConfig(generation_groups=(97, 97))

    def test_identical_compact_state_reuses_cached_frame(self) -> None:
        cache = UniqueRenderedStateCache()
        key = tuple(range(12))
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.assertEqual(cache.add(key, frame, 1), 0)
        self.assertEqual(cache.add(key, frame.copy(), 1), 0)
        self.assertEqual(len(cache.frames), 1)

    def test_rgb_alias_fails_closed(self) -> None:
        cache = UniqueRenderedStateCache()
        key = tuple(range(12))
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        cache.add(key, frame, 1)
        frame[0, 0] = 255
        with self.assertRaisesRegex(ValueError, "multiple RGB"):
            cache.add(key, frame, 1)

    def test_stage_alias_fails_closed(self) -> None:
        cache = UniqueRenderedStateCache()
        key = tuple(range(12))
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        cache.add(key, frame, 1)
        with self.assertRaisesRegex(ValueError, "multiple oracle"):
            cache.add(key, frame, 2)


if __name__ == "__main__":
    unittest.main()
