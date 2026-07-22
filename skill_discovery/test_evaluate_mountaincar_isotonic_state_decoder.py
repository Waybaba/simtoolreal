"""Tests for the MountainCar isotonic visual decoder."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.evaluate_mountaincar_isotonic_state_decoder import (
    build_decoder,
    decode_visual_pairs,
)


class MountainCarIsotonicStateDecoderTest(unittest.TestCase):
    def test_decoder_uses_frozen_bounds(self) -> None:
        decoder = build_decoder()
        self.assertTrue(decoder.increasing)
        self.assertEqual(decoder.y_min, -1.2)
        self.assertEqual(decoder.y_max, 0.6)
        self.assertEqual(decoder.out_of_bounds, "clip")

    def test_left_wall_resets_negative_velocity(self) -> None:
        decoder = build_decoder()
        decoder.fit(
            np.asarray([0.0, 0.5, 1.0]),
            np.asarray([-1.2, -0.3, 0.6]),
        )
        features = np.asarray([[0.01, 0.0, -0.01], [0.5, 0.4, -0.1]])
        before, after, velocity = decode_visual_pairs(decoder, features)
        self.assertLess(after[0], before[0])
        self.assertEqual(velocity[0], 0.0)
        self.assertLess(velocity[1], 0.0)


if __name__ == "__main__":
    unittest.main()
