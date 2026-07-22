"""Measure natural and scripted reachability in Pusher-Cup."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from skill_discovery.pusher_cup import PusherCupConfig, PusherCupEnv, TRAJECTORY_CLASSES


ACTION_VECTORS = np.asarray(
    (
        (0.0, 0.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (0.7071, 0.7071),
        (0.7071, -0.7071),
        (-0.7071, 0.7071),
        (-0.7071, -0.7071),
    ),
    dtype=np.float32,
)


def _class_summary(classes: np.ndarray) -> dict[str, int | float]:
    counts = np.bincount(classes, minlength=len(TRAJECTORY_CLASSES))
    total = int(counts.sum())
    return {
        f"{name}_count": int(counts[index])
        for index, name in enumerate(TRAJECTORY_CLASSES)
    } | {
        f"{name}_rate": float(counts[index] / total)
        for index, name in enumerate(TRAJECTORY_CLASSES)
    }


def _random_rollout(seed: int, num_envs: int, episode_length: int) -> dict[str, object]:
    config = PusherCupConfig(num_envs=num_envs, episode_length=episode_length, seed=seed)
    env = PusherCupEnv(config)
    rng = np.random.default_rng(seed + 1000)
    for _ in range(episode_length):
        action_ids = rng.integers(0, len(ACTION_VECTORS), size=num_envs)
        env.step(ACTION_VECTORS[action_ids])
    return {
        "seed": seed,
        **_class_summary(env.trajectory_classes()),
        "mean_ball_path_length": float(env.ball_path_length.mean()),
        "max_ball_path_length": float(env.ball_path_length.max()),
    }


def _scripted_rollout(seed: int, episodes_per_class: int, episode_length: int) -> dict[str, object]:
    num_envs = episodes_per_class * len(TRAJECTORY_CLASSES)
    config = PusherCupConfig(num_envs=num_envs, episode_length=episode_length, seed=seed)
    env = PusherCupEnv(config)
    rng = np.random.default_rng(seed + 2000)
    intended = np.repeat(np.arange(len(TRAJECTORY_CLASSES), dtype=np.int64), episodes_per_class)

    for step in range(episode_length):
        actions = np.zeros((num_envs, 2), dtype=np.float32)
        no_contact = intended == 0
        contact_only = intended == 1
        ball_inside = intended == 2
        if step < 10:
            vertical_sign = np.where(np.arange(num_envs) % 2 == 0, 1.0, -1.0)
            actions[no_contact, 1] = vertical_sign[no_contact]
        if step < 5:
            actions[contact_only, 0] = 1.0
            actions[contact_only, 1] = rng.uniform(-0.05, 0.05, size=contact_only.sum())
        if step < 12:
            actions[ball_inside, 0] = 1.0
            actions[ball_inside, 1] = rng.uniform(-0.025, 0.025, size=ball_inside.sum())
        env.step(actions)

    observed = env.trajectory_classes()
    per_class_recall = {
        name: float((observed[intended == class_id] == class_id).mean())
        for class_id, name in enumerate(TRAJECTORY_CLASSES)
    }
    confusion = np.zeros((len(TRAJECTORY_CLASSES), len(TRAJECTORY_CLASSES)), dtype=np.int64)
    np.add.at(confusion, (intended, observed), 1)
    return {
        "seed": seed,
        "episodes_per_class": episodes_per_class,
        "per_class_recall": per_class_recall,
        "confusion_matrix": confusion.tolist(),
        "mean_ball_path_length_by_intent": [
            float(env.ball_path_length[intended == class_id].mean())
            for class_id in range(len(TRAJECTORY_CLASSES))
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 17, 27, 37, 47))
    parser.add_argument("--random-envs", type=int, default=8192)
    parser.add_argument("--scripted-episodes-per-class", type=int, default=512)
    parser.add_argument("--episode-length", type=int, default=48)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    run_id = f"reachability_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path("outputs/skill_discovery/pusher_cup") / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    random_runs = [
        _random_rollout(seed, args.random_envs, args.episode_length) for seed in args.seeds
    ]
    scripted = _scripted_rollout(
        args.seeds[0],
        args.scripted_episodes_per_class,
        args.episode_length,
    )
    random_rate_summary = {
        name: {
            "mean": float(
                np.mean([run[f"{name}_rate"] for run in random_runs])
            ),
            "std": float(
                np.std([run[f"{name}_rate"] for run in random_runs])
            ),
            "total_count": int(
                sum(run[f"{name}_count"] for run in random_runs)
            ),
        }
        for name in TRAJECTORY_CLASSES
    }
    scripted_passed = all(
        recall >= 0.95 for recall in scripted["per_class_recall"].values()
    )
    output = {
        "run_id": run_id,
        "config": {
            "environment": asdict(
                PusherCupConfig(
                    num_envs=args.random_envs,
                    episode_length=args.episode_length,
                )
            ),
            "seeds": args.seeds,
            "random_envs_per_seed": args.random_envs,
            "scripted_episodes_per_class": args.scripted_episodes_per_class,
        },
        "random_runs": random_runs,
        "random_rate_summary": random_rate_summary,
        "scripted": scripted,
        "scripted_reachability_gate_passed": scripted_passed,
        "random_inside_observed": random_rate_summary["ball_inside"]["total_count"] > 0,
        "passed": scripted_passed,
    }
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path.resolve()), **output}, indent=2))


if __name__ == "__main__":
    main()
