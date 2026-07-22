"""Tests for synchronized Hammer state-rollout auditing."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from skill_discovery.analyze_hammer_state_rollouts import (
    FINGERTIP_NAMES,
    analyze_segment,
    classify_stage,
    point_to_hammer_geometry_distance,
    quaternion_rotate,
)


class HammerStateRolloutAuditTest(unittest.TestCase):
    def test_quaternion_rotation_uses_wxyz_order(self) -> None:
        half = math.sqrt(0.5)
        rotated = quaternion_rotate((half, 0.0, 0.0, half), (1.0, 0.0, 0.0))
        self.assertAlmostEqual(rotated[0], 0.0, places=6)
        self.assertAlmostEqual(rotated[1], 1.0, places=6)
        self.assertAlmostEqual(rotated[2], 0.0, places=6)

    def test_hammer_geometry_contains_handle_and_head(self) -> None:
        self.assertEqual(point_to_hammer_geometry_distance((0.0, 0.0, 0.0)), 0.0)
        self.assertEqual(point_to_hammer_geometry_distance((0.1325, 0.03, 0.0)), 0.0)
        self.assertAlmostEqual(
            point_to_hammer_geometry_distance((0.0, 0.02125, 0.0)),
            0.01,
            places=6,
        )

    def test_stage_requires_true_lift_for_near_goal(self) -> None:
        self.assertEqual(classify_stage(0.0, 0.0, 0.01, lift_delta=0.03), "rest")
        self.assertEqual(
            classify_stage(0.0, 0.03, 0.01, lift_delta=0.03),
            "moved_on_table",
        )
        self.assertEqual(
            classify_stage(0.04, 0.03, 0.08, lift_delta=0.03), "lifted_far"
        )
        self.assertEqual(
            classify_stage(0.04, 0.03, 0.06, lift_delta=0.03), "near_goal_7cm"
        )

    def test_segment_reconstructs_environment_lift_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            segment = Path(temporary)
            body_names = list(FINGERTIP_NAMES)
            manifest = {
                "source": {"video_path": None},
                "timing": {"max_frames": 3, "start_update": 10},
                "selection": {"logged_env_ids": [0], "seed": 7},
                "robot_model": {"body_names": body_names},
            }
            (segment / "manifest.json").write_text(json.dumps(manifest))
            (segment / "summary.json").write_text(
                json.dumps({"frames_written": 3, "closed_reason": "max_frames"})
            )
            positions = ((0.0, 0.0, 0.666), (0.03, 0.0, 0.666), (0.03, 0.0, 0.706))
            frames = []
            for index, position in enumerate(positions, start=1):
                far_positions = [[1.0, 1.0, 1.0] for _ in body_names]
                identities = [[1.0, 0.0, 0.0, 0.0] for _ in body_names]
                frames.append(
                    {
                        "control_step": 99 + index,
                        "envs": [
                            {
                                "env_id": 0,
                                "episode_step": index,
                                "done": False,
                                "successes": 0.0,
                                "table": {"pos_env": [0.0, 0.0, 0.5]},
                                "object": {
                                    "pos_env": list(position),
                                    "pose_w": {
                                        "pos": list(position),
                                        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                                    },
                                },
                                "goal": {"pos_env": [0.03, 0.0, 0.706]},
                                "robot": {
                                    "body_pos_w": far_positions,
                                    "body_quat_wxyz": identities,
                                },
                            }
                        ],
                    }
                )
            (segment / "frames.jsonl").write_text(
                "\n".join(json.dumps(frame) for frame in frames) + "\n"
            )
            config = {
                "reset_position_noise_z": 0.0,
                "table_object_z_offset": 0.166,
                "lifting_bonus_threshold": 0.08,
                "object_goal_pos_success_tolerance": 0.04,
            }
            result, rows = analyze_segment(segment, config, expected_frames=3)

        self.assertTrue(result["integrity"]["passed"])
        self.assertAlmostEqual(result["baseline"]["reconstructed_object_init_z_m"], 0.666)
        self.assertEqual([row["stage"] for row in rows], ["rest", "moved_on_table", "near_goal_7cm"])
        self.assertEqual(result["metrics"]["lifted_frames"], 1)
        self.assertEqual(result["metrics"]["strict_position_success_frames"], 1)
        self.assertFalse(result["interpretation"]["plausible_grasp_observed"])


if __name__ == "__main__":
    unittest.main()
