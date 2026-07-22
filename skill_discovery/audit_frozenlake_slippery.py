"""Audit stochastic FrozenLake outcome controllability before training."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from skill_discovery.frozenlake import (
    FROZENLAKE_OUTCOMES,
    FrozenLakeConfig,
    classify_outcome,
    make_frozenlake,
)
from skill_discovery.frozenlake_slippery import (
    OutcomePolicy,
    finite_horizon_outcome_policy,
    rollout_outcome_policy,
)
from skill_discovery.generate_point_cup_dataset import _write_png


def _environment_smoke(
    config: FrozenLakeConfig,
    seed: int,
) -> dict[str, object]:
    env = make_frozenlake(config, render_mode="rgb_array")
    try:
        first_state, _ = env.reset(seed=seed)
        first_frame = env.render()
        second_state, _ = env.reset(seed=seed)
        second_frame = env.render()
        return {
            "state_count": int(env.observation_space.n),
            "action_count": int(env.action_space.n),
            "map_shape": list(env.unwrapped.desc.shape),
            "map_rows": [row.tobytes().decode("ascii") for row in env.unwrapped.desc],
            "frame_shape": list(first_frame.shape),
            "seed_reproducible": bool(
                int(first_state) == int(second_state)
                and np.array_equal(first_frame, second_frame)
            ),
        }
    finally:
        env.close()


def _random_reachability(
    config: FrozenLakeConfig,
    seeds: list[int],
    episodes_per_seed: int,
) -> list[dict[str, object]]:
    outputs = []
    for seed in seeds:
        env = make_frozenlake(config)
        rng = np.random.default_rng(seed + 150_000)
        counts = np.zeros(len(FROZENLAKE_OUTCOMES), dtype=np.int64)
        try:
            for episode in range(episodes_per_seed):
                state, _ = env.reset(seed=seed * 1_000_000 + episode)
                terminated = truncated = False
                while not (terminated or truncated):
                    state, _, terminated, truncated, _ = env.step(
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
                "counts": counts.tolist(),
                "rates": (counts / episodes_per_seed).tolist(),
            }
        )
    return outputs


def _monte_carlo_policy(
    config: FrozenLakeConfig,
    policy: OutcomePolicy,
    seeds: list[int],
    episodes_per_seed: int,
) -> dict[str, object]:
    seed_runs = []
    total_counts = np.zeros(len(FROZENLAKE_OUTCOMES), dtype=np.int64)
    for seed in seeds:
        counts = np.zeros(len(FROZENLAKE_OUTCOMES), dtype=np.int64)
        step_sums = np.zeros(len(FROZENLAKE_OUTCOMES), dtype=np.int64)
        for episode in range(episodes_per_seed):
            rollout = rollout_outcome_policy(
                config,
                policy,
                seed=seed * 1_000_000 + episode,
            )
            outcome = int(rollout["outcome"])
            counts[outcome] += 1
            step_sums[outcome] += int(rollout["steps"])
        total_counts += counts
        seed_runs.append(
            {
                "seed": seed,
                "counts": counts.tolist(),
                "rates": (counts / episodes_per_seed).tolist(),
                "target_rate": float(
                    counts[policy.target_outcome] / episodes_per_seed
                ),
            }
        )
    total = episodes_per_seed * len(seeds)
    empirical = float(total_counts[policy.target_outcome] / total)
    return {
        "target_outcome": FROZENLAKE_OUTCOMES[policy.target_outcome],
        "probability_upper_bound": policy.probability_upper_bound,
        "runs": seed_runs,
        "total_counts": total_counts.tolist(),
        "total_rates": (total_counts / total).tolist(),
        "empirical_target_rate": empirical,
        "absolute_bound_error": abs(empirical - policy.probability_upper_bound),
    }


def _representative_rollout(
    config: FrozenLakeConfig,
    policy: OutcomePolicy,
    seed_start: int,
) -> dict[str, object]:
    for offset in range(10_000):
        rollout = rollout_outcome_policy(
            config,
            policy,
            seed=seed_start + offset,
            render=True,
        )
        if rollout["outcome"] == policy.target_outcome:
            return rollout
    raise RuntimeError("could not render a successful target-outcome rollout")


def _write_contact_sheet(
    path: Path,
    config: FrozenLakeConfig,
    policies: list[OutcomePolicy],
) -> list[dict[str, object]]:
    frame_size = 192
    columns = 4
    gap = 6
    marker_width = 10
    colors = ((48, 116, 173), (198, 72, 58), (42, 137, 94))
    sheet = np.full(
        (
            len(policies) * frame_size + (len(policies) - 1) * gap,
            marker_width + columns * frame_size + (columns - 1) * gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    manifest = []
    for row, policy in enumerate(policies):
        rollout = _representative_rollout(
            config,
            policy,
            seed_start=2_000_000 + row * 100_000,
        )
        frames = rollout.pop("frames")
        indices = np.linspace(0, len(frames) - 1, columns).astype(np.int64)
        y = row * (frame_size + gap)
        sheet[y : y + frame_size, :marker_width] = colors[row]
        for column, index in enumerate(indices):
            frame = frames[int(index)]
            rows = np.linspace(0, frame.shape[0] - 1, frame_size).astype(np.int64)
            cols = np.linspace(0, frame.shape[1] - 1, frame_size).astype(np.int64)
            x = marker_width + column * (frame_size + gap)
            sheet[y : y + frame_size, x : x + frame_size] = frame[rows][:, cols]
        manifest.append(
            {
                "target_outcome": FROZENLAKE_OUTCOMES[row],
                "actual_outcome": FROZENLAKE_OUTCOMES[int(rollout["outcome"])],
                **rollout,
            }
        )
    _write_png(path, sheet)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 17, 27, 37, 47))
    parser.add_argument("--random-episodes-per-seed", type=int, default=2_000)
    parser.add_argument("--policy-episodes-per-seed", type=int, default=512)
    parser.add_argument("--bound-tolerance", type=float, default=0.05)
    parser.add_argument("--map-name", default="4x4")
    parser.add_argument("--max-episode-steps", type=int, default=32)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = FrozenLakeConfig(
        map_name=args.map_name,
        is_slippery=True,
        max_episode_steps=args.max_episode_steps,
    )
    seeds = list(args.seeds)
    environment = _environment_smoke(config, seeds[0])
    policies = [
        finite_horizon_outcome_policy(config, target)
        for target in range(len(FROZENLAKE_OUTCOMES))
    ]
    policy_audits = [
        _monte_carlo_policy(
            config,
            policy,
            seeds,
            args.policy_episodes_per_seed,
        )
        for policy in policies
    ]
    random_runs = _random_reachability(
        config,
        seeds,
        args.random_episodes_per_seed,
    )
    first = rollout_outcome_policy(config, policies[2], seed=987_654)
    second = rollout_outcome_policy(config, policies[2], seed=987_654)
    seed_reproducible = first["states"] == second["states"]

    run_id = (
        f"frozenlake_{config.map_name}_slippery_audit_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/frozenlake_slippery"
    ) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    image_path = output_dir / "optimal_outcome_policy_audit.png"
    manifest = _write_contact_sheet(image_path, config, policies)
    random_counts = np.sum(
        np.asarray([run["counts"] for run in random_runs], dtype=np.int64),
        axis=0,
    )
    passed = bool(
        environment["seed_reproducible"]
        and seed_reproducible
        and all(
            audit["absolute_bound_error"] <= args.bound_tolerance
            for audit in policy_audits
        )
        and all(
            row["target_outcome"] == row["actual_outcome"] for row in manifest
        )
    )
    output = {
        "run_id": run_id,
        "config": asdict(config),
        "environment": environment,
        "seed_reproducible": seed_reproducible,
        "seeds": seeds,
        "random": {
            "episodes_per_seed": args.random_episodes_per_seed,
            "runs": random_runs,
            "total_counts": random_counts.tolist(),
            "total_rates": (
                random_counts / (len(seeds) * args.random_episodes_per_seed)
            ).tolist(),
        },
        "finite_horizon_policies": policy_audits,
        "bound_tolerance": args.bound_tolerance,
        "image": str(image_path.resolve()),
        "image_manifest": manifest,
        "passed": passed,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(summary_path.resolve()),
                "image": output["image"],
                "environment": environment,
                "seed_reproducible": seed_reproducible,
                "random_total_rates": output["random"]["total_rates"],
                "policy_rates": [
                    {
                        "outcome": audit["target_outcome"],
                        "bound": audit["probability_upper_bound"],
                        "empirical": audit["empirical_target_rate"],
                        "error": audit["absolute_bound_error"],
                    }
                    for audit in policy_audits
                ],
                "passed": passed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
