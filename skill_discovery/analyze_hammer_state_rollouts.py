"""Audit synchronized Hammer state rollouts without trusting legacy success counters."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


FINGERTIP_NAMES = (
    "left_index_DP",
    "left_middle_DP",
    "left_ring_DP",
    "left_thumb_DP",
    "left_pinky_DP",
)
FINGERTIP_OFFSET = (0.02, 0.002, 0.0)
HAMMER_HANDLE_HALF_LENGTH = 0.1125
HAMMER_HANDLE_RADIUS = 0.01125
HAMMER_HEAD_CENTER = (0.1325, 0.0, 0.0)
HAMMER_HEAD_HALF_SIZE = (0.02, 0.0425, 0.02)
PREREG_GOAL_DISTANCE = 0.07
FINGERTIP_PROXIMITY_DISTANCE = 0.03
STAGES = ("rest", "moved_on_table", "lifted_far", "near_goal_7cm")


def quaternion_rotate(
    quaternion_wxyz: Sequence[float], vector: Sequence[float]
) -> tuple[float, float, float]:
    """Rotate a 3D vector by a unit wxyz quaternion."""
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    vx, vy, vz = (float(value) for value in vector)
    uv = (y * vz - z * vy, z * vx - x * vz, x * vy - y * vx)
    uuv = (
        y * uv[2] - z * uv[1],
        z * uv[0] - x * uv[2],
        x * uv[1] - y * uv[0],
    )
    return tuple(
        component + 2.0 * (w * uv_component + uuv_component)
        for component, uv_component, uuv_component in zip(vector, uv, uuv)
    )


def quaternion_inverse_rotate(
    quaternion_wxyz: Sequence[float], vector: Sequence[float]
) -> tuple[float, float, float]:
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    return quaternion_rotate((w, -x, -y, -z), vector)


def point_to_hammer_geometry_distance(point_local: Sequence[float]) -> float:
    """Distance from a point to the union of the round handle and box head."""
    x, y, z = (float(value) for value in point_local)
    axial_outside = max(abs(x) - HAMMER_HANDLE_HALF_LENGTH, 0.0)
    radial_outside = max(math.hypot(y, z) - HAMMER_HANDLE_RADIUS, 0.0)
    handle_distance = math.hypot(axial_outside, radial_outside)

    head_relative = (x - HAMMER_HEAD_CENTER[0], y, z)
    head_outside = tuple(
        max(abs(value) - half_size, 0.0)
        for value, half_size in zip(head_relative, HAMMER_HEAD_HALF_SIZE)
    )
    head_distance = math.sqrt(sum(value * value for value in head_outside))
    return min(handle_distance, head_distance)


def fingertip_hammer_distance(
    env: dict[str, object], body_names: Sequence[str]
) -> float:
    """Return a geometry-based proximity proxy; this is not a contact sensor."""
    robot = env["robot"]
    object_pose = env["object"]["pose_w"]
    object_position = object_pose["pos"]
    object_quaternion = object_pose["quat_wxyz"]
    distances = []
    for name in FINGERTIP_NAMES:
        body_index = body_names.index(name)
        body_position = robot["body_pos_w"][body_index]
        body_quaternion = robot["body_quat_wxyz"][body_index]
        rotated_offset = quaternion_rotate(body_quaternion, FINGERTIP_OFFSET)
        fingertip_position = tuple(
            float(position) + offset
            for position, offset in zip(body_position, rotated_offset)
        )
        relative_world = tuple(
            fingertip - float(origin)
            for fingertip, origin in zip(fingertip_position, object_position)
        )
        relative_local = quaternion_inverse_rotate(object_quaternion, relative_world)
        distances.append(point_to_hammer_geometry_distance(relative_local))
    return min(distances)


def classify_stage(
    object_rise: float,
    object_xy_motion: float,
    object_goal_distance: float,
    *,
    lift_delta: float,
) -> str:
    if object_rise > lift_delta:
        return "near_goal_7cm" if object_goal_distance <= PREREG_GOAL_DISTANCE else "lifted_far"
    if object_xy_motion > 0.02:
        return "moved_on_table"
    return "rest"


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(
        sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))
    )


def _max_consecutive(values: Iterable[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def analyze_segment(
    segment_dir: Path,
    config: dict[str, object],
    *,
    expected_frames: int = 240,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest = json.loads((segment_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((segment_dir / "summary.json").read_text(encoding="utf-8"))
    frames = _read_jsonl(segment_dir / "frames.jsonl")
    if not frames:
        raise ValueError(f"empty trajectory segment: {segment_dir}")
    if float(config.get("reset_position_noise_z", 0.0) or 0.0) != 0.0:
        raise ValueError("exact reset-z reconstruction requires reset_position_noise_z=0")

    body_names = manifest["robot_model"]["body_names"]
    first_env = frames[0]["envs"][0]
    table_z = float(first_env["table"]["pos_env"][2])
    table_object_z_offset = float(config["table_object_z_offset"])
    object_init_z = table_z + table_object_z_offset
    first_object_position = first_env["object"]["pos_env"]
    lift_delta = float(config["lifting_bonus_threshold"]) - 0.05
    strict_goal_tolerance = float(config["object_goal_pos_success_tolerance"])
    if lift_delta <= 0.0:
        raise ValueError("the environment lift threshold does not require a true z increase")

    rows = []
    for frame_index, frame in enumerate(frames, start=1):
        if len(frame["envs"]) != 1:
            raise ValueError("state audit requires exactly one logged environment per frame")
        env = frame["envs"][0]
        object_position = env["object"]["pos_env"]
        goal_position = env["goal"]["pos_env"]
        object_rise = float(object_position[2]) - object_init_z
        object_xy_motion = math.hypot(
            float(object_position[0]) - float(first_object_position[0]),
            float(object_position[1]) - float(first_object_position[1]),
        )
        object_goal_distance = _distance(object_position, goal_position)
        fingertip_distance = fingertip_hammer_distance(env, body_names)
        lifted = object_rise > lift_delta
        fingertip_proximal = fingertip_distance <= FINGERTIP_PROXIMITY_DISTANCE
        rows.append(
            {
                "frame": frame_index,
                "control_step": int(frame["control_step"]),
                "episode_step": int(env["episode_step"]),
                "object_rise_m": object_rise,
                "object_xy_motion_from_first_frame_m": object_xy_motion,
                "object_goal_distance_m": object_goal_distance,
                "fingertip_geometry_distance_m": fingertip_distance,
                "stage": classify_stage(
                    object_rise,
                    object_xy_motion,
                    object_goal_distance,
                    lift_delta=lift_delta,
                ),
                "lifted": lifted,
                "strict_position_success": lifted
                and object_goal_distance <= strict_goal_tolerance,
                "fingertip_proximal": fingertip_proximal,
                "lift_with_fingertip_proximity": lifted and fingertip_proximal,
                "legacy_successes": float(env["successes"]),
                "done": bool(env["done"]),
            }
        )

    control_steps = [row["control_step"] for row in rows]
    episode_steps = [row["episode_step"] for row in rows]
    table_z_values = [
        float(frame["envs"][0]["table"]["pos_env"][2]) for frame in frames
    ]
    integrity = {
        "frames_written": len(rows),
        "expected_frames": expected_frames,
        "summary_frames_match": int(summary["frames_written"]) == len(rows),
        "closed_at_max_frames": summary["closed_reason"] == "max_frames",
        "manifest_max_frames_match": int(manifest["timing"]["max_frames"]) == len(rows),
        "logged_env_ids": manifest["selection"]["logged_env_ids"],
        "only_env0": manifest["selection"]["logged_env_ids"] == [0]
        and all(frame["envs"][0]["env_id"] == 0 for frame in frames),
        "control_steps_consecutive": all(
            right == left + 1 for left, right in zip(control_steps, control_steps[1:])
        ),
        "episode_steps_consecutive": all(
            right == left + 1 for left, right in zip(episode_steps, episode_steps[1:])
        ),
        "single_episode": episode_steps[0] == 1
        and episode_steps[-1] == len(episode_steps)
        and not any(row["done"] for row in rows),
        "video_path_is_null": manifest["source"]["video_path"] is None,
        "table_pose_stable": max(table_z_values) - min(table_z_values) < 1.0e-6,
    }
    integrity["passed"] = bool(
        len(rows) == expected_frames
        and all(
            value
            for key, value in integrity.items()
            if key
            not in {
                "frames_written",
                "expected_frames",
                "logged_env_ids",
                "video_path_is_null",
            }
        )
        and integrity["video_path_is_null"]
    )

    stage_counts = Counter(row["stage"] for row in rows)
    lifted_flags = [bool(row["lifted"]) for row in rows]
    grasp_proxy_flags = [bool(row["lift_with_fingertip_proximity"]) for row in rows]
    strict_flags = [bool(row["strict_position_success"]) for row in rows]
    first_lift_frame = next(
        (row["frame"] for row in rows if row["lifted"]), None
    )
    first_proximity_frame = next(
        (row["frame"] for row in rows if row["fingertip_proximal"]), None
    )
    if any(strict_flags):
        behavior = "strict_position_success"
    elif any(grasp_proxy_flags):
        behavior = "lift_with_fingertip_proximity"
    elif any(lifted_flags):
        behavior = "non_grasp_lift_from_initial_transient"
    elif max(row["object_xy_motion_from_first_frame_m"] for row in rows) > 0.02:
        behavior = "moved_on_table_only"
    else:
        behavior = "no_effect"

    result = {
        "segment_dir": str(segment_dir),
        "seed": int(manifest["selection"]["seed"]),
        "start_update": int(manifest["timing"]["start_update"]),
        "start_control_step": control_steps[0],
        "end_control_step": control_steps[-1],
        "integrity": integrity,
        "thresholds": {
            "environment_lift_delta_m": lift_delta,
            "environment_lift_formula": "0.05 + object_z - object_init_z > lifting_bonus_threshold",
            "preregistered_near_goal_m": PREREG_GOAL_DISTANCE,
            "strict_object_goal_position_m": strict_goal_tolerance,
            "fingertip_geometry_proximity_m": FINGERTIP_PROXIMITY_DISTANCE,
        },
        "baseline": {
            "table_z_m": table_z,
            "table_object_z_offset_m": table_object_z_offset,
            "reconstructed_object_init_z_m": object_init_z,
            "first_logged_object_position": first_object_position,
            "xy_motion_baseline": "first logged frame after action 1",
        },
        "stage_frame_counts": {stage: stage_counts.get(stage, 0) for stage in STAGES},
        "metrics": {
            "max_object_rise_m": max(row["object_rise_m"] for row in rows),
            "max_object_xy_motion_from_first_frame_m": max(
                row["object_xy_motion_from_first_frame_m"] for row in rows
            ),
            "min_object_goal_distance_m": min(
                row["object_goal_distance_m"] for row in rows
            ),
            "final_object_goal_distance_m": rows[-1]["object_goal_distance_m"],
            "min_fingertip_geometry_distance_m": min(
                row["fingertip_geometry_distance_m"] for row in rows
            ),
            "lifted_frames": sum(lifted_flags),
            "max_consecutive_lifted_frames": _max_consecutive(lifted_flags),
            "fingertip_proximity_frames": sum(
                bool(row["fingertip_proximal"]) for row in rows
            ),
            "lift_with_fingertip_proximity_frames": sum(grasp_proxy_flags),
            "max_consecutive_lift_with_fingertip_proximity_frames": _max_consecutive(
                grasp_proxy_flags
            ),
            "strict_position_success_frames": sum(strict_flags),
            "legacy_success_max": max(row["legacy_successes"] for row in rows),
            "first_lift_frame": first_lift_frame,
            "first_fingertip_proximity_frame": first_proximity_frame,
        },
        "behavior": behavior,
        "interpretation": {
            "plausible_grasp_observed": any(grasp_proxy_flags),
            "strict_position_success_observed": any(strict_flags),
            "initial_lift_precedes_fingertip_proximity": first_lift_frame is not None
            and (first_proximity_frame is None or first_lift_frame < first_proximity_frame),
            "proximity_caveat": "Geometry proximity is diagnostic only; no contact-force sensor was logged.",
        },
    }
    return result, rows


def _segment_frame_count(segment_dir: Path) -> int:
    with (segment_dir / "frames.jsonl").open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def primary_segment(run_dir: Path) -> Path:
    segments = list(run_dir.glob("training_logs/*/segment_*"))
    if not segments:
        raise FileNotFoundError(f"no trajectory segments under {run_dir}")
    return max(segments, key=_segment_frame_count)


def _run_sort_key(run_dir: Path) -> tuple[int, str]:
    order = {"early": 0, "middle": 1, "late": 2, "final": 3}
    match = re.search(r"hammer_sync_(early|middle|late|final)", run_dir.name)
    return (order.get(match.group(1), 99) if match else 99, run_dir.name)


def _run_label(run_name: str) -> str:
    phase = re.search(r"hammer_sync_(early|middle|late|final)", run_name)
    update = re.search(r"_u(\d+)", run_name)
    if phase and update:
        return f"{phase.group(1)} u{update.group(1)}"
    return run_name


def _write_rows_csv(path: Path, run_rows: dict[str, list[dict[str, object]]]) -> None:
    fieldnames = ["run", *next(iter(run_rows.values()))[0].keys()]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run_name, rows in run_rows.items():
            for row in rows:
                writer.writerow({"run": run_name, **row})


def _svg_polyline(
    values: Sequence[float],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    lower: float,
    upper: float,
) -> str:
    scale = max(upper - lower, 1.0e-9)
    points = []
    for index, value in enumerate(values):
        x = left + width * index / max(len(values) - 1, 1)
        y = top + height * (upper - value) / scale
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _write_chart(path: Path, run_rows: dict[str, list[dict[str, object]]]) -> None:
    colors = ("#2875a4", "#d17031", "#4c8b52", "#8b5a9a")
    panels = (
        ("object rise (m)", "object_rise_m", -0.01, 0.15, ((0.03, "lift 0.03 m"),)),
        (
            "object-goal distance (m)",
            "object_goal_distance_m",
            0.0,
            0.15,
            ((0.04, "strict 0.04 m"), (0.07, "stage 0.07 m")),
        ),
        (
            "fingertip-to-hammer geometry (m)",
            "fingertip_geometry_distance_m",
            0.0,
            0.36,
            ((0.03, "proxy 0.03 m"),),
        ),
    )
    width, height = 1160, 850
    left, panel_width, panel_height = 100, 970, 190
    elements = [
        f'<rect width="{width}" height="{height}" fill="#f8fafb"/>',
        '<text x="100" y="38" font-family="sans-serif" font-size="24" fill="#172027">Hammer env0 synchronized state audit</text>',
        '<text x="100" y="64" font-family="sans-serif" font-size="13" fill="#52616b">State-only rollout; fingertip geometry is a proximity proxy, not a contact sensor.</text>',
    ]
    for panel_index, (title, key, lower, upper, thresholds) in enumerate(panels):
        top = 105 + panel_index * 240
        elements.append(
            f'<rect x="{left}" y="{top}" width="{panel_width}" height="{panel_height}" fill="#ffffff" stroke="#c8d0d5"/>'
        )
        elements.append(
            f'<text x="{left}" y="{top - 12}" font-family="sans-serif" font-size="15" fill="#26343d">{title}</text>'
        )
        for threshold, label in thresholds:
            y = top + panel_height * (upper - threshold) / (upper - lower)
            elements.append(
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + panel_width}" y2="{y:.2f}" stroke="#89969e" stroke-dasharray="5 5"/>'
            )
            elements.append(
                f'<text x="{left + panel_width - 4}" y="{y - 5:.2f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#66757e">{label}</text>'
            )
        for run_index, (run_name, rows) in enumerate(run_rows.items()):
            points = _svg_polyline(
                [float(row[key]) for row in rows],
                left=left,
                top=top,
                width=panel_width,
                height=panel_height,
                lower=lower,
                upper=upper,
            )
            elements.append(
                f'<polyline points="{points}" fill="none" stroke="{colors[run_index]}" stroke-width="2"/>'
            )
        elements.extend(
            [
                f'<text x="{left - 10}" y="{top + 5}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#66757e">{upper:.2f}</text>',
                f'<text x="{left - 10}" y="{top + panel_height}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#66757e">{lower:.2f}</text>',
                f'<text x="{left}" y="{top + panel_height + 18}" font-family="sans-serif" font-size="11" fill="#66757e">frame 1</text>',
                f'<text x="{left + panel_width}" y="{top + panel_height + 18}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#66757e">frame 240</text>',
            ]
        )
    legend_y = 815
    for index, run_name in enumerate(run_rows):
        x = 105 + index * 245
        elements.append(
            f'<line x1="{x}" y1="{legend_y}" x2="{x + 25}" y2="{legend_y}" stroke="{colors[index]}" stroke-width="3"/>'
        )
        elements.append(
            f'<text x="{x + 32}" y="{legend_y + 5}" font-family="sans-serif" font-size="13" fill="#26343d">{_run_label(run_name)}</text>'
        )
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        + "".join(elements)
        + "</svg>\n",
        encoding="utf-8",
    )


def audit_runs(run_root: Path, output_dir: Path) -> dict[str, object]:
    run_dirs = sorted((path for path in run_root.iterdir() if path.is_dir()), key=_run_sort_key)
    if len(run_dirs) != 4:
        raise ValueError(f"expected four synchronized Hammer runs, found {len(run_dirs)}")
    output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    run_rows = {}
    for run_dir in run_dirs:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        result, rows = analyze_segment(primary_segment(run_dir), config)
        result["run_name"] = run_dir.name
        result["checkpoint"] = config["checkpoint"]
        results.append(result)
        run_rows[run_dir.name] = rows

    total_stages = Counter()
    for result in results:
        total_stages.update(result["stage_frame_counts"])
    output = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "run_root": str(run_root),
        "criteria": {
            "source_of_lift_definition": "src/isaaclab_env/isaaclab_env/tasks/direct/simtoolreal/env.py",
            "visual_metric_status": "not_run_renderer_hardware_blocked",
            "proximity_is_not_contact_ground_truth": True,
        },
        "runs": results,
        "aggregate": {
            "all_integrity_gates_passed": all(
                result["integrity"]["passed"] for result in results
            ),
            "state_stage_frame_counts": {stage: total_stages[stage] for stage in STAGES},
            "state_stage_coverage": [stage for stage in STAGES if total_stages[stage] > 0],
            "runs_with_any_true_lift": sum(
                result["metrics"]["lifted_frames"] > 0 for result in results
            ),
            "runs_with_lift_and_fingertip_proximity": sum(
                result["metrics"]["lift_with_fingertip_proximity_frames"] > 0
                for result in results
            ),
            "runs_with_strict_position_success": sum(
                result["metrics"]["strict_position_success_frames"] > 0
                for result in results
            ),
            "legacy_success_max": max(
                result["metrics"]["legacy_success_max"] for result in results
            ),
            "conclusion": "No env0 rollout shows a lift while the fingertips are geometrically proximal, and none reaches the strict 4 cm position-success region.",
        },
    }
    (output_dir / "audit.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    _write_rows_csv(output_dir / "frames.csv", run_rows)
    _write_chart(output_dir / "hammer_state_audit.svg", run_rows)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/skill_discovery/hammer_sync"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("hammer_state_audit_%Y%m%d_%H%M%S")
    output_dir = args.output_root / timestamp
    output = audit_runs(args.run_root, output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "aggregate": output["aggregate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
