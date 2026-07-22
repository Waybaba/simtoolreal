"""Render deterministic Point-Cup policy rollouts for visual auditing."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from skill_discovery.generate_point_cup_dataset import _write_png
from skill_discovery.point_cup import PointCupEnv, ShapeWorldConfig
from skill_discovery.train_tabular_diayn import TrainConfig, _episode_batch


METHOD_COLORS = {
    "random": (113, 123, 130),
    "raw": (40, 111, 161),
    "semantic": (43, 132, 93),
}
SKILL_COLORS = ((221, 117, 42), (112, 77, 157))


def _load_run(run_dir: Path) -> tuple[str, TrainConfig, np.ndarray]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = TrainConfig(**metrics["config"])
    with np.load(run_dir / "policy.npz") as policy:
        logits = policy["logits"].copy()
    return config.representation, config, logits


def _rollout(
    config: TrainConfig,
    logits: np.ndarray,
    episodes_per_skill: int,
    seed: int,
) -> dict[str, np.ndarray | float]:
    audit_config = TrainConfig(
        **{
            **asdict(config),
            "envs_per_skill": episodes_per_skill,
            "iterations": 1,
            "seed": seed,
        }
    )
    return _episode_batch(logits, audit_config, np.random.default_rng(seed), epsilon=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-episodes-per-skill", type=int, default=64)
    parser.add_argument("--shown-episodes-per-skill", type=int, default=4)
    parser.add_argument("--frame-size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=27001)
    args = parser.parse_args()

    if args.audit_episodes_per_skill < args.shown_episodes_per_skill:
        raise ValueError("audit episodes must be at least the number of shown episodes")

    runs = [_load_run(run_dir) for run_dir in args.run_dirs]
    runs.sort(key=lambda item: ("random", "raw", "semantic").index(item[0]))
    if {method for method, _, _ in runs} != {"random", "raw", "semantic"}:
        raise ValueError("provide exactly one random, raw, and semantic run")

    rendered_rows: list[dict[str, object]] = []
    full_rates: dict[str, list[float]] = {}
    frame_batches: list[np.ndarray] = []
    frame_indices: list[int] | None = None

    for method_index, (method, config, logits) in enumerate(runs):
        batch = _rollout(
            config,
            logits,
            args.audit_episodes_per_skill,
            args.seed + method_index,
        )
        skills = np.asarray(batch["skills"])
        terminal_inside = np.asarray(batch["terminal_inside"])
        position_history = np.asarray(batch["position_history"])
        full_rates[method] = [
            float(terminal_inside[skills == skill].mean()) for skill in range(config.num_skills)
        ]

        if frame_indices is None:
            frame_indices = np.linspace(
                0,
                config.episode_length,
                num=5,
                dtype=np.int64,
            ).tolist()
        row_env_ids: list[int] = []
        for skill in range(config.num_skills):
            skill_ids = np.flatnonzero(skills == skill)[: args.shown_episodes_per_skill]
            for episode_index, env_id in enumerate(skill_ids):
                row_env_ids.append(int(env_id))
                rendered_rows.append(
                    {
                        "method": method,
                        "skill": skill,
                        "episode": episode_index,
                        "terminal_inside": bool(terminal_inside[env_id]),
                        "terminal_position": position_history[-1, env_id].tolist(),
                    }
                )

        row_env_ids_array = np.asarray(row_env_ids, dtype=np.int64)
        render_env = PointCupEnv(
            ShapeWorldConfig(
                num_envs=len(row_env_ids),
                episode_length=1,
                action_scale=config.action_scale,
                cup_center=(0.0, 0.0),
                cup_axes=(0.14, 0.14),
                seed=args.seed,
            )
        )
        method_frames = []
        for frame_index in frame_indices:
            render_env.reset(positions=position_history[frame_index, row_env_ids_array])
            method_frames.append(render_env.render(size=args.frame_size))
        frame_batches.extend(method_frames)

    assert frame_indices is not None
    rows_per_method = 2 * args.shown_episodes_per_skill
    row_count = len(rendered_rows)
    gap = 3
    block_gap = 9
    method_marker_width = 7
    skill_marker_width = 7
    marker_width = method_marker_width + skill_marker_width
    sheet_width = marker_width + len(frame_indices) * args.frame_size + (len(frame_indices) - 1) * gap
    sheet_height = (
        row_count * args.frame_size
        + (row_count - len(runs)) * gap
        + (len(runs) - 1) * block_gap
    )
    sheet = np.full((sheet_height, sheet_width, 3), 255, dtype=np.uint8)

    output_row = 0
    for method_index, (method, _, _) in enumerate(runs):
        method_frames = frame_batches[
            method_index * len(frame_indices) : (method_index + 1) * len(frame_indices)
        ]
        for local_row in range(rows_per_method):
            if local_row > 0:
                output_row += gap
            y = output_row
            skill = local_row // args.shown_episodes_per_skill
            sheet[y : y + args.frame_size, :method_marker_width] = METHOD_COLORS[method]
            sheet[
                y : y + args.frame_size,
                method_marker_width:marker_width,
            ] = SKILL_COLORS[skill]
            for frame_offset, frames in enumerate(method_frames):
                x = marker_width + frame_offset * (args.frame_size + gap)
                sheet[y : y + args.frame_size, x : x + args.frame_size] = frames[local_row]
            output_row += args.frame_size
        if method_index < len(runs) - 1:
            output_row += block_gap

    semantic_rates = full_rates["semantic"]
    passed = bool(max(semantic_rates) >= 0.75 and min(semantic_rates) <= 0.25)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / "policy_rollout_audit.png"
    _write_png(image_path, sheet)
    output = {
        "image": str(image_path.resolve()),
        "source_runs": [str(path.resolve()) for path in args.run_dirs],
        "audit_episodes_per_skill": args.audit_episodes_per_skill,
        "shown_episodes_per_skill": args.shown_episodes_per_skill,
        "frame_indices": frame_indices,
        "method_order": [method for method, _, _ in runs],
        "method_colors": METHOD_COLORS,
        "skill_colors": SKILL_COLORS,
        "inside_rates": full_rates,
        "semantic_sample_gate_passed": passed,
        "passed": passed,
        "rows": rendered_rows,
    }
    manifest_path = args.output_dir / "policy_rollout_audit.json"
    manifest_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "image": output["image"],
                "inside_rates": full_rates,
                "passed": passed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
