"""Render deterministic Pusher-Cup policy rollouts for visual auditing."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from skill_discovery.generate_point_cup_dataset import _write_png
from skill_discovery.pusher_cup import PusherCupConfig, PusherCupEnv, TRAJECTORY_CLASSES
from skill_discovery.train_pusher_diayn import (
    PusherTrainConfig,
    _best_class_assignment,
    _episode_batch,
)


METHOD_COLORS = {
    "random": (111, 120, 127),
    "raw": (42, 112, 163),
    "semantic": (185, 95, 44),
    "semantic_spread": (26, 151, 143),
    "semantic_balanced": (43, 137, 94),
}
SKILL_COLORS = ((221, 117, 42), (112, 77, 157), (36, 137, 166))


def _load_run(run_dir: Path) -> tuple[str, PusherTrainConfig, np.ndarray]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = PusherTrainConfig(**metrics["config"])
    with np.load(run_dir / "policy.npz") as policy:
        logits = policy["logits"].copy()
    return config.representation, config, logits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-episodes-per-skill", type=int, default=128)
    parser.add_argument("--shown-episodes-per-skill", type=int, default=3)
    parser.add_argument("--frame-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=37001)
    args = parser.parse_args()
    if args.audit_episodes_per_skill < args.shown_episodes_per_skill:
        raise ValueError("audit episodes must be at least shown episodes")

    method_order = (
        "random",
        "raw",
        "semantic",
        "semantic_spread",
        "semantic_balanced",
    )
    runs = [_load_run(path) for path in args.run_dirs]
    runs.sort(key=lambda item: method_order.index(item[0]))
    if len({method for method, _, _ in runs}) != len(runs):
        raise ValueError("provide at most one run for each representation")

    all_method_frames: list[list[np.ndarray]] = []
    rendered_rows: list[dict[str, object]] = []
    method_metrics: dict[str, object] = {}
    frame_indices: list[int] | None = None
    for method_index, (method, config, logits) in enumerate(runs):
        audit_config = PusherTrainConfig(
            **{
                **asdict(config),
                "envs_per_skill": args.audit_episodes_per_skill,
                "iterations": 1,
                "seed": args.seed + method_index,
            }
        )
        batch = _episode_batch(
            logits,
            audit_config,
            np.random.default_rng(args.seed + method_index),
            epsilon=0.0,
        )
        skills = batch["skills"]
        classes = batch["trajectory_classes"]
        class_rates = np.zeros((config.num_skills, len(TRAJECTORY_CLASSES)), dtype=np.float64)
        for skill in range(config.num_skills):
            class_rates[skill] = np.bincount(
                classes[skills == skill], minlength=len(TRAJECTORY_CLASSES)
            ) / args.audit_episodes_per_skill
        assignment, matched_rates = _best_class_assignment(class_rates)
        inside_skill = assignment.index(2)
        inside_ball_path = float(
            batch["ball_path_length"][skills == inside_skill].mean()
        )
        passed = bool(min(matched_rates) >= 0.70 and inside_ball_path >= 0.40)
        method_metrics[method] = {
            "class_rates": class_rates.tolist(),
            "assignment": assignment,
            "assignment_names": [TRAJECTORY_CLASSES[index] for index in assignment],
            "matched_rates": matched_rates,
            "inside_skill_ball_path_mean": inside_ball_path,
            "passed": passed,
        }

        if frame_indices is None:
            frame_indices = np.linspace(
                0, config.episode_length, num=7, dtype=np.int64
            ).tolist()
        row_env_ids = []
        for skill in range(config.num_skills):
            skill_ids = np.flatnonzero(skills == skill)[: args.shown_episodes_per_skill]
            for episode_index, env_id in enumerate(skill_ids):
                row_env_ids.append(int(env_id))
                rendered_rows.append(
                    {
                        "method": method,
                        "skill": skill,
                        "episode": episode_index,
                        "observed_class": TRAJECTORY_CLASSES[int(classes[env_id])],
                        "ball_path_length": float(batch["ball_path_length"][env_id]),
                        "terminal_pusher_position": batch["terminal_pusher_positions"][env_id].tolist(),
                        "terminal_ball_position": batch["terminal_ball_positions"][env_id].tolist(),
                    }
                )

        row_env_ids_array = np.asarray(row_env_ids, dtype=np.int64)
        render_env = PusherCupEnv(
            PusherCupConfig(
                num_envs=len(row_env_ids),
                episode_length=1,
                action_scale=config.action_scale,
                seed=args.seed,
            )
        )
        frames_for_method = []
        for frame_index in frame_indices:
            render_env.reset(
                pusher_positions=batch["pusher_history"][frame_index, row_env_ids_array],
                ball_positions=batch["ball_history"][frame_index, row_env_ids_array],
            )
            render_env.current_contact = batch["contact_history"][
                frame_index, row_env_ids_array
            ].copy()
            frames_for_method.append(render_env.render(size=args.frame_size))
        all_method_frames.append(frames_for_method)

    assert frame_indices is not None
    rows_per_method = 3 * args.shown_episodes_per_skill
    gap = 3
    block_gap = 10
    marker_width = 14
    sheet_width = marker_width + len(frame_indices) * args.frame_size + (len(frame_indices) - 1) * gap
    sheet_height = (
        len(runs) * rows_per_method * args.frame_size
        + len(runs) * (rows_per_method - 1) * gap
        + (len(runs) - 1) * block_gap
    )
    sheet = np.full((sheet_height, sheet_width, 3), 255, dtype=np.uint8)
    y = 0
    for method_index, (method, _, _) in enumerate(runs):
        for local_row in range(rows_per_method):
            skill = local_row // args.shown_episodes_per_skill
            sheet[y : y + args.frame_size, :7] = METHOD_COLORS[method]
            sheet[y : y + args.frame_size, 7:marker_width] = SKILL_COLORS[skill]
            for column, frames in enumerate(all_method_frames[method_index]):
                x = marker_width + column * (args.frame_size + gap)
                sheet[y : y + args.frame_size, x : x + args.frame_size] = frames[local_row]
            y += args.frame_size
            if local_row < rows_per_method - 1:
                y += gap
        if method_index < len(runs) - 1:
            y += block_gap

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / "policy_rollout_audit.png"
    _write_png(image_path, sheet)
    balanced_present = "semantic_balanced" in method_metrics
    primary_method = (
        "semantic_spread"
        if "semantic_spread" in method_metrics
        else "semantic_balanced"
        if balanced_present
        else None
    )
    output = {
        "image": str(image_path.resolve()),
        "source_runs": [str(path.resolve()) for path in args.run_dirs],
        "frame_indices": frame_indices,
        "method_order": [method for method, _, _ in runs],
        "method_colors": METHOD_COLORS,
        "skill_colors": SKILL_COLORS,
        "method_metrics": method_metrics,
        "balanced_oracle_sample_gate_passed": bool(
            balanced_present and method_metrics["semantic_balanced"]["passed"]
        ),
        "primary_method": primary_method,
        "passed": bool(
            primary_method is not None and method_metrics[primary_method]["passed"]
        ),
        "rows": rendered_rows,
    }
    manifest_path = args.output_dir / "policy_rollout_audit.json"
    manifest_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("image", "method_metrics", "passed")}, indent=2))


if __name__ == "__main__":
    main()
