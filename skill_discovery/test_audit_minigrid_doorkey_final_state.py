"""Tests for DoorKey final-state persistence audits."""

from __future__ import annotations

import unittest

from skill_discovery.audit_minigrid_doorkey_final_state import (
    _final_state_success,
)


class DoorKeyFinalStateAuditTest(unittest.TestCase):
    def test_door_requires_open_final_state(self) -> None:
        rollout = {
            "stage": 2,
            "carrying": ["key", "yellow"],
            "door_open": False,
            "native_success": False,
        }
        self.assertFalse(_final_state_success(2, rollout))
        rollout["door_open"] = True
        self.assertTrue(_final_state_success(2, rollout))

    def test_key_requires_carried_key_before_open_door(self) -> None:
        rollout = {
            "stage": 1,
            "carrying": ["key", "yellow"],
            "door_open": False,
            "native_success": False,
        }
        self.assertTrue(_final_state_success(1, rollout))
        rollout["door_open"] = True
        self.assertFalse(_final_state_success(1, rollout))


if __name__ == "__main__":
    unittest.main()
