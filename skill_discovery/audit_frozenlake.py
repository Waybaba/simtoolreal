"""Audit official FrozenLake outcomes and random reachability."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from skill_discovery.frozenlake import (
    FROZENLAKE_OUTCOMES,
    SCRIPTED_ACTIONS,
    FrozenLakeConfig,
    classify_outcome,
    make_frozenlake,
    rollout_actions,
)
from skill_discovery.generate_point_cup_dataset import _write_png


def _resize_nearest(image: np.ndarray, size: int) -> np.ndarray:
    y = np.linspace(0, image.shape[0] - 1, size).astype(np.int64)
    x = np.linspace(0, image.shape[1] - 1, size).astype(np.int64)
    return image[y][:, x]


def _write_contact_sheet(path: Path, rows: list[np.ndarray]) -> None:
    frame_size = rows[0].shape[1]
    gap = 6
    marker_width = 10
    sheet = np.full(
        (
            len(rows) * frame_size + (len(rows) - 1) * gap,
            marker_width + 3 * frame_size + 2 * gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    colors = ((48, 116, 173), (198, 72, 58), (42, 137, 94))
    for row_index, frames in enumerate(rows):
        y = row_index * (frame_size + gap)
        sheet[y : y + frame_size, :marker_width] = colors[row_index]
        for column, frame in enumerate(frames):
            x = marker_width + column * (frame_size + gap)
            sheet[y : y + frame_size, x : x + frame_size] = frame
    _write_png(path, sheet)


def _scripted_audit(
    config: FrozenLakeConfig,
    seeds: list[int],
) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    rows = []
    renders = []
    for expected, name in enumerate(FROZENLAKE_OUTCOMES):
        seed_runs = []
        for seed in seeds:
            result = rollout_actions(
                config,
                SCRIPTED_ACTIONS[name],
                seed=seed,
                render=seed == seeds[0],
            )
            seed_runs.append(
                {
                    "seed": seed,
                    "outcome": FROZENLAKE_OUTCOMES[result.outcome],
                    "action_count": len(result.actions),
                    "final_state": result.final_state,
                    "native_reward": result.native_reward,
                    "terminated": result.terminated,
                    "truncated": result.truncated,
                }
            )
            if result.frames:
                indices = (0, len(result.frames) // 2, len(result.frames) - 1)
                renders.append(
                    np.stack(
                        [_resize_nearest(result.frames[index], 192) for index in indices]
                    )
                )
        rows.append(
            {
                "intended_outcome": name,
                "runs": seed_runs,
                "passed": all(run["outcome"] == name for run in seed_runs),
            }
        )
    return rows, renders


def _random_reachability(
    config: FrozenLakeConfig,
    seeds: list[int],
    episodes_per_seed: int,
) -> list[dict[str, object]]:
    outputs = []
    for seed in seeds:
        env = make_frozenlake(config)
        rng = np.random.default_rng(seed + 50_000)
        counts = np.zeros(len(FROZENLAKE_OUTCOMES), dtype=np.int64)
        try:
            for episode in range(episodes_per_seed):
                state, _ = env.reset(seed=seed * 100_000 + episode)
                terminated = truncated = False
                native_reward = 0.0
                while not (terminated or truncated):
                    state, native_reward, terminated, truncated, _ = env.step(
                        int(rng.integers(env.action_space.n))
                    )
                outcome = classify_outcome(
                    env,
                    int(state),
                    terminated=terminated,
                    truncated=truncated,
                )
                counts[outcome] += 1
        finally:
            env.close()
        outputs.append(
            {
                "seed": seed,
                "episodes": episodes_per_seed,
                "outcome_counts": {
                    name: int(counts[index])
                    for index, name in enumerate(FROZENLAKE_OUTCOMES)
                },
                "outcome_rates": {
                    name: float(counts[index] / episodes_per_seed)
                    for index, name in enumerate(FROZENLAKE_OUTCOMES)
                },
            }
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 17, 27, 37, 47))
    parser.add_argument("--random-episodes-per-seed", type=int, default=2_000)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = FrozenLakeConfig()
    seeds = list(args.seeds)
    run_id = f"frozenlake_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/frozenlake"
    ) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    scripted, renders = _scripted_audit(config, seeds)
    random_runs = _random_reachability(
        config,
        seeds,
        args.random_episodes_per_seed,
    )
    image_path = output_dir / "scripted_outcome_audit.png"
    _write_contact_sheet(image_path, renders)
    totals = {
        name: sum(run["outcome_counts"][name] for run in random_runs)
        for name in FROZENLAKE_OUTCOMES
    }
    output = {
        "run_id": run_id,
        "version": {"gymnasium": importlib.metadata.version("gymnasium")},
        "config": asdict(config),
        "seeds": seeds,
        "outcomes": FROZENLAKE_OUTCOMES,
        "scripted": scripted,
        "random": {
            "episodes_per_seed": args.random_episodes_per_seed,
            "runs": random_runs,
            "total_outcome_counts": totals,
        },
        "image": str(image_path.resolve()),
        "passed": all(row["passed"] for row in scripted),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(summary_path.resolve()),
                "image": output["image"],
                "scripted_gate_passed": output["passed"],
                "random_total_outcome_counts": totals,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
