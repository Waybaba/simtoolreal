"""Evaluate trained Pusher-Cup policies on held-out cup layouts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from skill_discovery.generate_point_cup_dataset import _write_png
from skill_discovery.pusher_cup import PusherCupConfig, PusherCupEnv
from skill_discovery.train_pusher_diayn import (
    PusherTrainConfig,
    _batch_metrics,
    _episode_batch,
)


def _layouts() -> list[dict[str, object]]:
    layouts = []
    for x in (0.25, 0.35, 0.45):
        for y in (-0.20, 0.00, 0.20):
            layouts.append(
                {
                    "name": f"position_x{x:.2f}_y{y:.2f}",
                    "center": (x, y),
                    "axes": (0.14, 0.16),
                    "kind": "position",
                }
            )
    for index, axes in enumerate(((0.10, 0.20), (0.20, 0.10), (0.18, 0.18))):
        layouts.append(
            {
                "name": f"shape_{index}",
                "center": (0.35, 0.0),
                "axes": axes,
                "kind": "shape",
            }
        )
    return layouts


def _load_run(run_dir: Path) -> tuple[PusherTrainConfig, np.ndarray]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = PusherTrainConfig(**metrics["config"])
    if config.representation != "semantic_spread":
        raise ValueError(f"{run_dir} is not a semantic_spread run")
    with np.load(run_dir / "policy.npz") as policy:
        logits = policy["logits"].copy()
    return config, logits


def _evaluate_layout(
    logits: np.ndarray,
    config: PusherTrainConfig,
    center: tuple[float, float],
    axes: tuple[float, float],
    seed: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    audit_config = PusherTrainConfig(
        **{
            **asdict(config),
            "envs_per_skill": config.eval_episodes_per_skill,
            "iterations": 1,
            "seed": seed,
        }
    )
    num_envs = audit_config.num_skills * audit_config.envs_per_skill
    centers = np.broadcast_to(np.asarray(center, dtype=np.float32), (num_envs, 2)).copy()
    cup_axes = np.broadcast_to(np.asarray(axes, dtype=np.float32), (num_envs, 2)).copy()
    batch = _episode_batch(
        logits,
        audit_config,
        np.random.default_rng(seed),
        epsilon=0.0,
        cup_centers=centers,
        cup_axes=cup_axes,
    )
    rewards = np.zeros(num_envs, dtype=np.float32)
    metrics = _batch_metrics(batch, rewards, audit_config)
    assignment = metrics["class_assignment"]
    inside_skill = assignment.index(2)
    metrics["inside_skill"] = inside_skill
    metrics["inside_skill_ball_path_mean"] = float(
        batch["ball_path_length"][batch["skills"] == inside_skill].mean()
    )
    metrics["passed"] = bool(
        min(metrics["matched_class_rates"]) >= config.class_rate_gate
        and metrics["inside_skill_ball_path_mean"] >= config.inside_ball_path_gate
    )
    return metrics, batch


def _render_seed_audit(
    output_dir: Path,
    layouts: list[dict[str, object]],
    rows: list[dict[str, object]],
    frame_size: int,
) -> Path:
    frame_indices = rows[0]["frame_indices"]
    row_count = len(rows)
    centers = np.asarray([layout["center"] for layout in layouts], dtype=np.float32)
    axes = np.asarray([layout["axes"] for layout in layouts], dtype=np.float32)
    env = PusherCupEnv(PusherCupConfig(num_envs=row_count, episode_length=1))
    env.set_layout(centers=centers, axes=axes)
    frame_batches = []
    for frame_offset, _ in enumerate(frame_indices):
        env.reset(
            pusher_positions=np.asarray(
                [row["pusher_frames"][frame_offset] for row in rows], dtype=np.float32
            ),
            ball_positions=np.asarray(
                [row["ball_frames"][frame_offset] for row in rows], dtype=np.float32
            ),
        )
        env.current_contact = np.asarray(
            [row["contact_frames"][frame_offset] for row in rows], dtype=bool
        )
        frame_batches.append(env.render(size=frame_size))

    marker_width = 9
    gap = 4
    row_gap = 6
    sheet_width = marker_width + len(frame_indices) * frame_size + (len(frame_indices) - 1) * gap
    sheet_height = row_count * frame_size + (row_count - 1) * row_gap
    sheet = np.full((sheet_height, sheet_width, 3), 255, dtype=np.uint8)
    for row_index, row in enumerate(rows):
        y = row_index * (frame_size + row_gap)
        marker = (43, 137, 94) if row["passed"] else (202, 73, 52)
        sheet[y : y + frame_size, :marker_width] = marker
        for column, frames in enumerate(frame_batches):
            x = marker_width + column * (frame_size + gap)
            sheet[y : y + frame_size, x : x + frame_size] = frames[row_index]
    image_path = output_dir / "layout_generalization_audit.png"
    _write_png(image_path, sheet)
    return image_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-episodes-per-skill", type=int, default=512)
    parser.add_argument("--frame-size", type=int, default=80)
    args = parser.parse_args()

    layouts = _layouts()
    run_outputs = []
    visual_rows: list[dict[str, object]] = []
    for run_index, run_dir in enumerate(args.run_dirs):
        config, logits = _load_run(run_dir)
        config = PusherTrainConfig(
            **{**asdict(config), "eval_episodes_per_skill": args.eval_episodes_per_skill}
        )
        base_metrics, _ = _evaluate_layout(
            logits,
            config,
            (0.35, 0.0),
            (0.14, 0.16),
            seed=config.seed + 40000,
        )
        layout_outputs = []
        for layout_index, layout in enumerate(layouts):
            metrics, batch = _evaluate_layout(
                logits,
                config,
                layout["center"],
                layout["axes"],
                seed=config.seed + 41000 + layout_index,
            )
            layout_outputs.append({"layout": layout, "metrics": metrics})
            if config.seed == 7:
                inside_skill = int(base_metrics["inside_skill"])
                env_id = inside_skill * config.eval_episodes_per_skill
                frame_indices = np.linspace(
                    0, config.episode_length, num=6, dtype=np.int64
                ).tolist()
                visual_rows.append(
                    {
                        "layout": layout["name"],
                        "passed": metrics["passed"],
                        "frame_indices": frame_indices,
                        "pusher_frames": batch["pusher_history"][frame_indices, env_id].tolist(),
                        "ball_frames": batch["ball_history"][frame_indices, env_id].tolist(),
                        "contact_frames": batch["contact_history"][frame_indices, env_id].tolist(),
                    }
                )
        passed_layouts = sum(bool(row["metrics"]["passed"]) for row in layout_outputs)
        run_outputs.append(
            {
                "run_dir": str(run_dir.resolve()),
                "seed": config.seed,
                "base_control": base_metrics,
                "layouts_passed": passed_layouts,
                "seed_gate_passed": passed_layouts >= 9 and bool(base_metrics["passed"]),
                "layouts": layout_outputs,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = _render_seed_audit(
        args.output_dir,
        layouts,
        visual_rows,
        args.frame_size,
    )
    seeds_passed = sum(bool(run["seed_gate_passed"]) for run in run_outputs)
    output = {
        "source_runs": [str(path.resolve()) for path in args.run_dirs],
        "layout_count": len(layouts),
        "required_layouts_per_seed": 9,
        "required_seed_count": 4,
        "seeds_passed": seeds_passed,
        "passed": seeds_passed >= 4,
        "visual_audit_image": str(image_path.resolve()),
        "runs": run_outputs,
        "visual_rows": visual_rows,
    }
    output_path = args.output_dir / "layout_generalization_audit.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path.resolve()),
                "image": output["visual_audit_image"],
                "layouts_passed_by_seed": {
                    str(run["seed"]): run["layouts_passed"] for run in run_outputs
                },
                "seeds_passed": seeds_passed,
                "passed": output["passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
