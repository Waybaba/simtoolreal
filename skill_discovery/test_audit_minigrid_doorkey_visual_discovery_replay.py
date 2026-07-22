"""Tests for post-hoc causal-visual DoorKey replay auditing."""

from __future__ import annotations

import unittest

import numpy as np

from skill_discovery.audit_minigrid_doorkey_visual_discovery_replay import (
    stage_confusion,
    stage_from_compact_state,
)


class DoorKeyVisualDiscoveryReplayAuditTest(unittest.TestCase):
    def test_compact_state_stage_is_irreversible_object_progress(self) -> None:
        base = np.asarray([1, 1, 0, 2, 2, 0, 3, 2, 0, 1, 3, 3])
        self.assertEqual(stage_from_compact_state(base), 0)
        carrying = base.copy()
        carrying[5] = 1
        self.assertEqual(stage_from_compact_state(carrying), 1)
        open_door = carrying.copy()
        open_door[5] = 0
        open_door[8] = 1
        self.assertEqual(stage_from_compact_state(open_door), 2)
        goal = open_door.copy()
        goal[0:2] = goal[10:12]
        self.assertEqual(stage_from_compact_state(goal), 3)

    def test_stage_confusion_uses_oracle_rows_and_decoded_columns(self) -> None:
        confusion = stage_confusion(
            np.asarray([0, 1, 2, 2, 3]),
            np.asarray([0, 1, 2, 3, 3]),
        )
        self.assertEqual(confusion[2, 3], 1)
        self.assertEqual(int(confusion.trace()), 4)


if __name__ == "__main__":
    unittest.main()
