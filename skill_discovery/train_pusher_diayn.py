"""Train tabular DIAYN-style policies in the Pusher-Cup environment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import permutations
from pathlib import Path

import numpy as np

from skill_discovery.pusher_cup import PusherCupConfig, PusherCupEnv, TRAJECTORY_CLASSES
from skill_discovery.train_tabular_diayn import (
    ACTION_VECTORS,
    _discriminator_rewards,
    _mutual_information,
    _sample_categorical,
    _softmax,
)


@dataclass(frozen=True)
class PusherTrainConfig:
    representation: str = "semantic"
    seed: int = 7
    num_skills: int = 3
    envs_per_skill: int = 512
    iterations: int = 800
    episode_length: int = 48
    policy_grid_size: int = 9
    raw_feature_grid_size: int = 9
    action_scale: float = 0.055
    learning_rate: float = 0.30
    epsilon_start: float = 0.35
    epsilon_end: float = 0.00
    exploration_burst_length: int = 12
    discriminator_decay: float = 0.97
    discriminator_pseudocount: float = 2.0
    baseline_alpha: float = 0.08
    pusher_start: tuple[float, float] = (-0.55, 0.0)
    ball_start: tuple[float, float] = (-0.25, 0.0)
    start_noise: float = 0.006
    eval_episodes_per_skill: int = 2048
    class_rate_gate: float = 0.70
    inside_ball_path_gate: float = 0.40
    semantic_coverage_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.representation not in {
            "raw",
            "semantic",
            "semantic_spread",
            "semantic_balanced",
            "random",
        }:
            raise ValueError(
                "representation must be raw, semantic, semantic_spread, "
                "semantic_balanced, or random"
            )
        if self.num_skills != len(TRAJECTORY_CLASSES):
            raise ValueError("Pusher-Cup is intentionally fixed to three skills")
        if min(
            self.envs_per_skill,
            self.iterations,
            self.episode_length,
            self.policy_grid_size,
            self.raw_feature_grid_size,
            self.eval_episodes_per_skill,
            self.exploration_burst_length,
        ) <= 0:
            raise ValueError("training sizes must be positive")
        if self.semantic_coverage_weight < 0:
            raise ValueError("semantic coverage weight must be non-negative")


def _coordinate_ids(positions: np.ndarray, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.floor(
        (np.clip(positions, -1.0, 1.0) + 1.0) * 0.5 * grid_size
    ).astype(np.int64)
    coordinates = np.clip(coordinates, 0, grid_size - 1)
    ids = coordinates[:, 0] * grid_size + coordinates[:, 1]
    return coordinates, ids


def _graph_state_ids(
    pusher_positions: np.ndarray,
    ball_positions: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    pusher_coordinates, _ = _coordinate_ids(pusher_positions, grid_size)
    ball_coordinates, _ = _coordinate_ids(ball_positions, grid_size)
    return (
        ((pusher_coordinates[:, 0] * grid_size + pusher_coordinates[:, 1]) * grid_size
        + ball_coordinates[:, 0])
        * grid_size
        + ball_coordinates[:, 1]
    )


def _episode_batch(
    policy_logits: np.ndarray,
    config: PusherTrainConfig,
    rng: np.random.Generator,
    epsilon: float,
    *,
    cup_centers: np.ndarray | None = None,
    cup_axes: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    num_envs = config.num_skills * config.envs_per_skill
    skills = np.repeat(np.arange(config.num_skills, dtype=np.int64), config.envs_per_skill)
    env = PusherCupEnv(
        PusherCupConfig(
            num_envs=num_envs,
            episode_length=config.episode_length,
            action_scale=config.action_scale,
            pusher_start=tuple(config.pusher_start),
            ball_start=tuple(config.ball_start),
            seed=config.seed,
        )
    )
    pusher_starts = np.broadcast_to(
        np.asarray(config.pusher_start, dtype=np.float32), (num_envs, 2)
    ).copy()
    ball_starts = np.broadcast_to(
        np.asarray(config.ball_start, dtype=np.float32), (num_envs, 2)
    ).copy()
    pusher_starts += rng.normal(0.0, config.start_noise, size=pusher_starts.shape).astype(np.float32)
    ball_starts += rng.normal(0.0, config.start_noise, size=ball_starts.shape).astype(np.float32)
    env.reset(pusher_positions=pusher_starts, ball_positions=ball_starts)
    if cup_centers is not None or cup_axes is not None:
        env.set_layout(centers=cup_centers, axes=cup_axes)

    state_ids = np.empty((config.episode_length, num_envs), dtype=np.int32)
    actions = np.empty_like(state_ids)
    probabilities = np.empty(
        (config.episode_length, num_envs, len(ACTION_VECTORS)), dtype=np.float32
    )
    pusher_history = np.empty((config.episode_length + 1, num_envs, 2), dtype=np.float32)
    ball_history = np.empty_like(pusher_history)
    contact_history = np.empty((config.episode_length + 1, num_envs), dtype=bool)
    pusher_history[0] = env.pusher_positions
    ball_history[0] = env.ball_positions
    contact_history[0] = env.current_contact
    burst_actions = np.zeros(num_envs, dtype=np.int64)
    burst_steps_remaining = np.zeros(num_envs, dtype=np.int16)

    for step in range(config.episode_length):
        ids = _graph_state_ids(env.pusher_positions, env.ball_positions, config.policy_grid_size)
        policy_probabilities = _softmax(policy_logits[skills, ids])
        policy_actions = _sample_categorical(policy_probabilities, rng)
        start_burst = (burst_steps_remaining == 0) & (rng.random(num_envs) < epsilon)
        burst_actions[start_burst] = rng.integers(
            0,
            len(ACTION_VECTORS),
            size=int(start_burst.sum()),
        )
        burst_steps_remaining[start_burst] = config.exploration_burst_length
        burst_active = burst_steps_remaining > 0
        sampled_actions = np.where(burst_active, burst_actions, policy_actions)
        burst_steps_remaining[burst_active] -= 1
        env.step(ACTION_VECTORS[sampled_actions])
        state_ids[step] = ids
        actions[step] = sampled_actions
        probabilities[step] = policy_probabilities
        pusher_history[step + 1] = env.pusher_positions
        ball_history[step + 1] = env.ball_positions
        contact_history[step + 1] = env.current_contact

    terminal_raw_features = _graph_state_ids(
        env.pusher_positions,
        env.ball_positions,
        config.raw_feature_grid_size,
    )
    _, terminal_pusher_cells = _coordinate_ids(env.pusher_positions, config.policy_grid_size)
    _, terminal_ball_cells = _coordinate_ids(env.ball_positions, config.policy_grid_size)
    trajectory_classes = env.trajectory_classes()
    return {
        "skills": skills,
        "state_ids": state_ids,
        "actions": actions,
        "policy_probabilities": probabilities,
        "terminal_raw_features": terminal_raw_features,
        "terminal_pusher_cells": terminal_pusher_cells,
        "terminal_ball_cells": terminal_ball_cells,
        "trajectory_classes": trajectory_classes,
        "ever_contact": env.ever_contact.copy(),
        "ball_inside": (trajectory_classes == 2),
        "ball_path_length": env.ball_path_length.copy(),
        "terminal_pusher_positions": env.pusher_positions.copy(),
        "terminal_ball_positions": env.ball_positions.copy(),
        "pusher_history": pusher_history,
        "ball_history": ball_history,
        "contact_history": contact_history,
    }


def _best_class_assignment(class_rates: np.ndarray) -> tuple[list[int], list[float]]:
    best_assignment: tuple[int, ...] | None = None
    best_rates: list[float] = []
    best_score = -1.0
    for assignment in permutations(range(class_rates.shape[1])):
        rates = [float(class_rates[skill, class_id]) for skill, class_id in enumerate(assignment)]
        score = float(np.mean(rates))
        if score > best_score:
            best_assignment = assignment
            best_rates = rates
            best_score = score
    assert best_assignment is not None
    return list(best_assignment), best_rates


def _balanced_class_targets(seed: int, num_skills: int) -> np.ndarray:
    return np.random.default_rng(seed + 30000).permutation(num_skills).astype(np.int64)


def _batch_metrics(
    batch: dict[str, np.ndarray],
    rewards: np.ndarray,
    config: PusherTrainConfig,
) -> dict[str, object]:
    skills = batch["skills"]
    classes = batch["trajectory_classes"]
    class_rates = np.zeros((config.num_skills, len(TRAJECTORY_CLASSES)), dtype=np.float64)
    for skill in range(config.num_skills):
        class_rates[skill] = np.bincount(
            classes[skills == skill], minlength=len(TRAJECTORY_CLASSES)
        ) / config.envs_per_skill
    assignment, matched_rates = _best_class_assignment(class_rates)
    inside_skill = assignment.index(2)
    ball_path_means = [
        float(batch["ball_path_length"][skills == skill].mean())
        for skill in range(config.num_skills)
    ]
    specialization_passed = bool(
        min(matched_rates) >= config.class_rate_gate
        and ball_path_means[inside_skill] >= config.inside_ball_path_gate
    )
    raw_feature_count = config.raw_feature_grid_size**4
    return {
        "class_rates": class_rates.tolist(),
        "class_assignment": assignment,
        "class_assignment_names": [TRAJECTORY_CLASSES[index] for index in assignment],
        "matched_class_rates": matched_rates,
        "matched_class_rate_mean": float(np.mean(matched_rates)),
        "semantic_mi_bits": _mutual_information(
            skills, classes, config.num_skills, len(TRAJECTORY_CLASSES)
        ),
        "terminal_graph_mi_bits": _mutual_information(
            skills, batch["terminal_raw_features"], config.num_skills, raw_feature_count
        ),
        "pusher_terminal_coverage": float(
            len(np.unique(batch["terminal_pusher_cells"])) / config.policy_grid_size**2
        ),
        "ball_terminal_coverage": float(
            len(np.unique(batch["terminal_ball_cells"])) / config.policy_grid_size**2
        ),
        "contact_rates": [
            float(batch["ever_contact"][skills == skill].mean())
            for skill in range(config.num_skills)
        ],
        "inside_rates": [
            float(batch["ball_inside"][skills == skill].mean())
            for skill in range(config.num_skills)
        ],
        "ball_path_length_means": ball_path_means,
        "reward_means": [
            float(rewards[skills == skill].mean()) for skill in range(config.num_skills)
        ],
        "reward_mean": float(rewards.mean()),
        "specialization_gate_passed": specialization_passed,
    }


def _policy_update(
    policy_logits: np.ndarray,
    batch: dict[str, np.ndarray],
    advantages: np.ndarray,
    config: PusherTrainConfig,
) -> None:
    skills = batch["skills"]
    state_ids = batch["state_ids"]
    actions = batch["actions"]
    probabilities = batch["policy_probabilities"]
    scale = config.learning_rate / config.envs_per_skill
    for step in range(config.episode_length):
        for action_id in range(len(ACTION_VECTORS)):
            gradient = advantages * (
                (actions[step] == action_id).astype(np.float32)
                - probabilities[step, :, action_id]
            )
            np.add.at(
                policy_logits[..., action_id],
                (skills, state_ids[step]),
                scale * gradient,
            )
    np.clip(policy_logits, -12.0, 12.0, out=policy_logits)


def train(
    config: PusherTrainConfig,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    rng = np.random.default_rng(config.seed)
    state_count = config.policy_grid_size**4
    policy_logits = np.zeros(
        (config.num_skills, state_count, len(ACTION_VECTORS)), dtype=np.float32
    )
    feature_count = (
        config.raw_feature_grid_size**4
        if config.representation == "raw"
        else len(TRAJECTORY_CLASSES)
    )
    discriminator_counts = np.full(
        (config.num_skills, feature_count),
        config.discriminator_pseudocount,
        dtype=np.float64,
    )
    baselines = np.zeros(config.num_skills, dtype=np.float32)
    balanced_targets = _balanced_class_targets(config.seed, config.num_skills)
    history: list[dict[str, object]] = []

    for iteration in range(config.iterations):
        progress = iteration / max(config.iterations - 1, 1)
        epsilon = config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)
        batch = _episode_batch(policy_logits, config, rng, epsilon)
        skills = batch["skills"]
        features = (
            batch["terminal_raw_features"]
            if config.representation == "raw"
            else batch["trajectory_classes"]
        )
        if config.representation == "random":
            rewards = np.zeros(len(skills), dtype=np.float32)
            coverage_rewards = np.zeros(len(skills), dtype=np.float32)
        elif config.representation == "semantic_balanced":
            rewards = (
                features == balanced_targets[skills]
            ).astype(np.float32)
            coverage_rewards = np.zeros(len(skills), dtype=np.float32)
        else:
            rewards, discriminator_counts = _discriminator_rewards(
                skills,
                features,
                discriminator_counts,
                config,
            )
            if config.representation == "semantic_spread":
                global_counts = discriminator_counts.sum(axis=0)
                class_probabilities = global_counts / global_counts.sum()
                coverage_rewards = -np.log(
                    np.maximum(
                        len(TRAJECTORY_CLASSES) * class_probabilities[features],
                        1.0e-8,
                    )
                ).astype(np.float32)
                rewards += config.semantic_coverage_weight * coverage_rewards
            else:
                coverage_rewards = np.zeros(len(skills), dtype=np.float32)

        if config.representation != "random":
            advantages = rewards - baselines[skills]
            for skill in range(config.num_skills):
                mask = skills == skill
                skill_advantages = advantages[mask]
                advantages[mask] = skill_advantages / max(float(skill_advantages.std()), 0.1)
                reward_mean = float(rewards[mask].mean())
                baselines[skill] = (
                    (1.0 - config.baseline_alpha) * baselines[skill]
                    + config.baseline_alpha * reward_mean
                )
            _policy_update(policy_logits, batch, np.clip(advantages, -5.0, 5.0), config)

        history.append(
            {
                "iteration": iteration + 1,
                "epsilon": float(epsilon),
                "balanced_target_classes": balanced_targets.tolist()
                if config.representation == "semantic_balanced"
                else None,
                "coverage_reward_mean": float(coverage_rewards.mean()),
                "coverage_reward_by_skill": [
                    float(coverage_rewards[skills == skill].mean())
                    for skill in range(config.num_skills)
                ],
                **_batch_metrics(batch, rewards, config),
            }
        )
    return policy_logits, history


def evaluate(policy_logits: np.ndarray, config: PusherTrainConfig, seed: int) -> dict[str, object]:
    eval_config = PusherTrainConfig(
        **{
            **asdict(config),
            "envs_per_skill": config.eval_episodes_per_skill,
            "iterations": 1,
            "seed": seed,
        }
    )
    batch = _episode_batch(
        policy_logits,
        eval_config,
        np.random.default_rng(seed),
        epsilon=0.0,
    )
    rewards = np.zeros(config.num_skills * config.eval_episodes_per_skill, dtype=np.float32)
    metrics = _batch_metrics(batch, rewards, eval_config)
    skills = batch["skills"]
    metrics["terminal_pusher_position_mean"] = [
        batch["terminal_pusher_positions"][skills == skill].mean(axis=0).tolist()
        for skill in range(config.num_skills)
    ]
    metrics["terminal_ball_position_mean"] = [
        batch["terminal_ball_positions"][skills == skill].mean(axis=0).tolist()
        for skill in range(config.num_skills)
    ]
    return metrics


def _write_curves(path: Path, history: list[dict[str, object]]) -> None:
    width, height = 900, 440
    left, top, chart_width, chart_height = 65, 60, 780, 300
    colors = ("#2779a7", "#d26c2f", "#2c8b62")
    polylines = []
    for skill in range(3):
        values = np.asarray([row["inside_rates"][skill] for row in history], dtype=np.float64)
        points = []
        for index, value in enumerate(values):
            x = left + chart_width * index / max(len(values) - 1, 1)
            y = top + chart_height * (1.0 - value)
            points.append(f"{x:.1f},{y:.1f}")
        polylines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[skill]}" stroke-width="2"/>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fbfcfd"/>
<text x="30" y="32" font-family="sans-serif" font-size="20" fill="#172b3a">Pusher-Cup training: final ball-inside rate</text>
<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#83919a"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#83919a"/>
<line x1="{left}" y1="{top + 0.3 * chart_height}" x2="{left + chart_width}" y2="{top + 0.3 * chart_height}" stroke="#b8c2c8" stroke-dasharray="4 4"/>
<text x="22" y="{top + 5}" font-family="sans-serif" font-size="12">1.0</text>
<text x="22" y="{top + chart_height + 5}" font-family="sans-serif" font-size="12">0.0</text>
<text x="{left}" y="{height - 30}" font-family="sans-serif" font-size="12">iteration 1</text>
<text x="{left + chart_width - 95}" y="{height - 30}" font-family="sans-serif" font-size="12">iteration {len(history)}</text>
{''.join(polylines)}
<rect x="600" y="20" width="14" height="4" fill="{colors[0]}"/><text x="620" y="27" font-family="sans-serif" font-size="12">skill 0</text>
<rect x="690" y="20" width="14" height="4" fill="{colors[1]}"/><text x="710" y="27" font-family="sans-serif" font-size="12">skill 1</text>
<rect x="780" y="20" width="14" height="4" fill="{colors[2]}"/><text x="800" y="27" font-family="sans-serif" font-size="12">skill 2</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--representation",
        choices=(
            "raw",
            "semantic",
            "semantic_spread",
            "semantic_balanced",
            "random",
        ),
        default="semantic",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--envs-per-skill", type=int, default=512)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = PusherTrainConfig(
        representation=args.representation,
        seed=args.seed,
        iterations=args.iterations,
        envs_per_skill=args.envs_per_skill,
    )
    run_id = f"pusher_diayn_{args.representation}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path("outputs/skill_discovery/pusher_cup_training") / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    policy_logits, history = train(config)
    evaluation = evaluate(policy_logits, config, seed=config.seed + 10000)
    stability_window = history[-min(20, len(history)) :]
    gate_values = [bool(row["specialization_gate_passed"]) for row in stability_window]
    output = {
        "run_id": run_id,
        "config": asdict(config),
        "balanced_target_classes": _balanced_class_targets(
            config.seed, config.num_skills
        ).tolist()
        if config.representation == "semantic_balanced"
        else None,
        "evaluation": evaluation,
        "training_stability": {
            "window": len(stability_window),
            "specialization_gate_fraction": float(np.mean(gate_values)),
            "matched_class_rate_mean": float(
                np.mean([row["matched_class_rate_mean"] for row in stability_window])
            ),
            "semantic_mi_bits_mean": float(
                np.mean([row["semantic_mi_bits"] for row in stability_window])
            ),
            "passed": bool(np.mean(gate_values) >= 0.8),
        },
        "last_train_iteration": history[-1],
    }
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    with (output_dir / "history.jsonl").open("w", encoding="utf-8") as stream:
        for row in history:
            stream.write(json.dumps(row) + "\n")
    (output_dir / "metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    np.savez_compressed(output_dir / "policy.npz", logits=policy_logits)
    _write_curves(output_dir / "training_curves.svg", history)
    print(json.dumps({"output_dir": str(output_dir.resolve()), **output}, indent=2))


if __name__ == "__main__":
    main()
