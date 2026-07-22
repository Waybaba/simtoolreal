"""Render scripted Pusher-Cup trajectories and verify their semantic labels."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from skill_discovery.generate_point_cup_dataset import _write_png
from skill_discovery.pusher_cup import PusherCupConfig, PusherCupEnv, TRAJECTORY_CLASSES


ROW_COLORS = np.asarray(
    (
        (108, 117, 124),
        (42, 116, 162),
        (45, 137, 94),
        (195, 93, 45),
    ),
    dtype=np.uint8,
)


def _scripted_actions(episode_length: int) -> np.ndarray:
    actions = np.zeros((episode_length, 4, 2), dtype=np.float32)
    actions[:, 0, 1] = -1.0
    actions[:5, 1, 0] = 1.0
    actions[:12, 2, 0] = 1.0
    actions[:4, 3, 1] = 1.0
    actions[4:21, 3, 0] = 1.0
    actions[21:25, 3, 1] = -1.0
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--frame-size", type=int, default=112)
    args = parser.parse_args()

    run_id = f"environment_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path("outputs/skill_discovery/pusher_cup") / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    config = PusherCupConfig(num_envs=4, episode_length=32, seed=13)
    env = PusherCupEnv(config)
    actions = _scripted_actions(config.episode_length)
    pusher_history = np.empty((config.episode_length + 1, config.num_envs, 2), dtype=np.float32)
    ball_history = np.empty_like(pusher_history)
    contact_history = np.empty((config.episode_length + 1, config.num_envs), dtype=bool)
    pusher_history[0] = env.pusher_positions
    ball_history[0] = env.ball_positions
    contact_history[0] = env.current_contact
    for step in range(config.episode_length):
        env.step(actions[step])
        pusher_history[step + 1] = env.pusher_positions
        ball_history[step + 1] = env.ball_positions
        contact_history[step + 1] = env.current_contact

    frame_indices = (0, 4, 8, 12, 20, 25, 32)
    frame_batches = []
    for frame_index in frame_indices:
        env.reset(
            pusher_positions=pusher_history[frame_index],
            ball_positions=ball_history[frame_index],
        )
        env.current_contact = contact_history[frame_index].copy()
        frame_batches.append(env.render(size=args.frame_size))

    final_pusher = pusher_history[-1]
    final_ball = ball_history[-1]
    env.reset(pusher_positions=final_pusher, ball_positions=final_ball)
    ball_path_lengths = np.linalg.norm(np.diff(ball_history, axis=0), axis=-1).sum(axis=0)
    ever_contact = contact_history.any(axis=0)
    env.ball_path_length = ball_path_lengths.astype(np.float32)
    env.ever_contact = ever_contact
    classes = env.trajectory_classes()
    pusher_rho = np.linalg.norm((final_pusher - env.cup_centers) / env.cup_axes, axis=-1)
    ball_inside = env.observe()["relation_features"][:, 2].astype(bool)

    expected_classes = np.asarray((0, 1, 2, 0), dtype=np.int64)
    checks = {
        "semantic_classes_match": bool(np.array_equal(classes, expected_classes)),
        "no_contact_ball_static": bool(ball_path_lengths[0] == 0.0),
        "contact_moves_ball": bool(ball_path_lengths[1] > 0.0 and not ball_inside[1]),
        "push_reaches_cup": bool(ball_path_lengths[2] > 0.4 and ball_inside[2]),
        "pusher_in_cup_is_not_ball_success": bool(
            pusher_rho[3] <= 1.0 and ball_path_lengths[3] == 0.0 and not ball_inside[3]
        ),
    }

    marker_width = 9
    gap = 4
    row_gap = 7
    sheet_width = marker_width + len(frame_indices) * args.frame_size + (len(frame_indices) - 1) * gap
    sheet_height = config.num_envs * args.frame_size + (config.num_envs - 1) * row_gap
    sheet = np.full((sheet_height, sheet_width, 3), 255, dtype=np.uint8)
    for row in range(config.num_envs):
        y = row * (args.frame_size + row_gap)
        sheet[y : y + args.frame_size, :marker_width] = ROW_COLORS[row]
        for column, frames in enumerate(frame_batches):
            x = marker_width + column * (args.frame_size + gap)
            sheet[y : y + args.frame_size, x : x + args.frame_size] = frames[row]

    image_path = output_dir / "scripted_trajectory_audit.png"
    _write_png(image_path, sheet)
    output = {
        "run_id": run_id,
        "image": str(image_path.resolve()),
        "frame_indices": frame_indices,
        "row_names": (
            "no_contact",
            "contact_without_inside",
            "ball_inside",
            "pusher_inside_false_positive",
        ),
        "row_colors": ROW_COLORS.tolist(),
        "observed_classes": [TRAJECTORY_CLASSES[index] for index in classes],
        "ball_path_lengths": ball_path_lengths.tolist(),
        "final_pusher_positions": final_pusher.tolist(),
        "final_ball_positions": final_ball.tolist(),
        "checks": checks,
        "passed": all(checks.values()),
    }
    manifest_path = output_dir / "audit.json"
    manifest_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
