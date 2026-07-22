"""Tests for balanced DoorKey policy-state reservoir sampling."""

from __future__ import annotations

import unittest

from skill_discovery.generate_minigrid_doorkey_policy_states import (
    BalancedStateReservoir,
    PolicyStateDatasetConfig,
)


class DoorKeyPolicyStateDatasetTest(unittest.TestCase):
    def test_generation_groups_must_be_unique(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            PolicyStateDatasetConfig(generation_groups=(57, 57))

    def test_reservoir_preserves_fixed_capacity_per_group_stage(self) -> None:
        reservoir = BalancedStateReservoir((57, 67), 3, seed=7)
        for group in (57, 67):
            for stage in range(4):
                for index in range(20):
                    reservoir.add(
                        group,
                        stage,
                        {"group": group, "stage": stage, "index": index},
                    )
        rows = reservoir.balanced_rows()
        self.assertEqual(len(rows), 24)
        for group in (57, 67):
            for stage in range(4):
                selected = [
                    row
                    for row in rows
                    if row["group"] == group and row["stage"] == stage
                ]
                self.assertEqual(len(selected), 3)
                self.assertEqual(reservoir.seen[(group, stage)], 20)

    def test_incomplete_reservoir_fails_closed(self) -> None:
        reservoir = BalancedStateReservoir((57,), 2, seed=7)
        reservoir.add(57, 0, {"stage": 0})
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            reservoir.balanced_rows()


if __name__ == "__main__":
    unittest.main()
