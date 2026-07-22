"""Audit official MiniGrid DoorKey semantics and random reachability."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np

from skill_discovery.generate_point_cup_dataset import _write_png
from skill_discovery.minigrid_doorkey import (
    DOORKEY_STAGES,
    reproducibility_signature,
    semantic_stage,
    solve_doorkey_episode,
)


ENV_IDS = ("MiniGrid-DoorKey-5x5-v0", "MiniGrid-DoorKey-8x8-v0")


def _resize_nearest(image: np.ndarray, size: int) -> np.ndarray:
    y = np.linspace(0, image.shape[0] - 1, size).astype(np.int64)
    x = np.linspace(0, image.shape[1] - 1, size).astype(np.int64)
    return image[y][:, x]


def _scripted_audit(seeds: list[int]) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    outputs = []
    render_rows = []
    for env_id in ENV_IDS:
        env = gym.make(env_id, render_mode="rgb_array")
        try:
            seed_outputs = []
            for seed in seeds:
                result = solve_doorkey_episode(env, seed)
                seed_outputs.append(
                    {
                        "seed": seed,
                        "action_count": len(result.actions),
                        "stages_seen": sorted(set(result.stage_sequence)),
                        "reward": result.reward,
                        "terminated": result.terminated,
                        "truncated": result.truncated,
                    }
                )
                if seed == seeds[0]:
                    render_rows.append(
                        np.stack([_resize_nearest(frame, 192) for frame in result.frames])
                    )
            signature_a = reproducibility_signature(env, seeds[0])
            signature_b = reproducibility_signature(env, seeds[0])
            outputs.append(
                {
                    "env_id": env_id,
                    "action_space_n": int(env.action_space.n),
                    "observation_image_shape": list(env.observation_space["image"].shape),
                    "seed_reproducible": signature_a == signature_b,
                    "runs": seed_outputs,
                    "passed": bool(
                        signature_a == signature_b
                        and all(
                            row["stages_seen"] == list(range(len(DOORKEY_STAGES)))
                            and row["terminated"]
                            and not row["truncated"]
                            and row["reward"] > 0
                            for row in seed_outputs
                        )
                    ),
                }
            )
        finally:
            env.close()
    return outputs, render_rows


def _random_seed_reachability(seed: int, episodes_per_seed: int) -> dict[str, object]:
    env = gym.make("MiniGrid-DoorKey-8x8-v0")
    try:
        rng = np.random.default_rng(seed + 50000)
        counts = np.zeros(len(DOORKEY_STAGES), dtype=np.int64)
        rewards = []
        for episode in range(episodes_per_seed):
            env.reset(seed=seed * 100000 + episode)
            furthest_stage = 0
            final_reward = 0.0
            for _ in range(env.unwrapped.max_steps):
                action = int(rng.integers(env.action_space.n))
                _, reward, terminated, truncated, _ = env.step(action)
                final_reward = float(reward)
                furthest_stage = max(
                    furthest_stage,
                    semantic_stage(env, terminated=terminated, reward=final_reward),
                )
                if terminated or truncated:
                    break
            counts[furthest_stage] += 1
            rewards.append(final_reward)
        return {
            "seed": seed,
            "episodes": episodes_per_seed,
            "furthest_stage_counts": {
                name: int(counts[index]) for index, name in enumerate(DOORKEY_STAGES)
            },
            "furthest_stage_rates": {
                name: float(counts[index] / episodes_per_seed)
                for index, name in enumerate(DOORKEY_STAGES)
            },
            "native_success_count": int(np.count_nonzero(np.asarray(rewards) > 0)),
        }
    finally:
        env.close()


def _random_reachability(seeds: list[int], episodes_per_seed: int) -> list[dict[str, object]]:
    with ProcessPoolExecutor(max_workers=len(seeds)) as executor:
        return list(
            executor.map(
                _random_seed_reachability,
                seeds,
                [episodes_per_seed] * len(seeds),
            )
        )


def _json_default(value: object) -> int | float | bool:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _write_contact_sheet(path: Path, rows: list[np.ndarray]) -> None:
    frame_size = rows[0].shape[1]
    gap = 6
    marker_width = 9
    sheet = np.full(
        (
            len(rows) * frame_size + (len(rows) - 1) * gap,
            marker_width + 4 * frame_size + 3 * gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    row_colors = ((42, 112, 163), (43, 137, 94))
    for row_index, frames in enumerate(rows):
        y = row_index * (frame_size + gap)
        sheet[y : y + frame_size, :marker_width] = row_colors[row_index]
        for column, frame in enumerate(frames):
            x = marker_width + column * (frame_size + gap)
            sheet[y : y + frame_size, x : x + frame_size] = frame
    _write_png(path, sheet)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 17, 27, 37, 47))
    parser.add_argument("--random-episodes-per-seed", type=int, default=512)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    run_id = f"doorkey_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path("outputs/skill_discovery/minigrid_doorkey") / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    seeds = list(args.seeds)
    scripted, render_rows = _scripted_audit(seeds)
    random_runs = _random_reachability(seeds, args.random_episodes_per_seed)
    image_path = output_dir / "scripted_stage_audit.png"
    _write_contact_sheet(image_path, render_rows)

    random_totals = {
        name: int(
            sum(run["furthest_stage_counts"][name] for run in random_runs)
        )
        for name in DOORKEY_STAGES
    }
    output = {
        "run_id": run_id,
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("minigrid", "gymnasium", "pygame-ce")
        },
        "seeds": seeds,
        "stage_names": DOORKEY_STAGES,
        "scripted": scripted,
        "random": {
            "episodes_per_seed": args.random_episodes_per_seed,
            "runs": random_runs,
            "total_furthest_stage_counts": random_totals,
        },
        "image": str(image_path.resolve()),
        "scripted_gate_passed": all(row["passed"] for row in scripted),
        "passed": all(row["passed"] for row in scripted),
    }
    output_path = output_dir / "summary.json"
    output_path.write_text(
        json.dumps(output, indent=2, default=_json_default), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output_path.resolve()),
                "image": output["image"],
                "versions": output["versions"],
                "scripted_gate_passed": output["scripted_gate_passed"],
                "random_total_furthest_stage_counts": random_totals,
                "passed": output["passed"],
            },
            indent=2,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
