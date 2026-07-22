"""Train a tiny tabular DIAYN-style policy in the Point-Cup environment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from skill_discovery.point_cup import PointCupEnv, ShapeWorldConfig


ACTION_VECTORS = np.asarray(
    [
        (0.0, 0.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (0.7071, 0.7071),
        (0.7071, -0.7071),
        (-0.7071, 0.7071),
        (-0.7071, -0.7071),
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class TrainConfig:
    representation: str = "semantic"
    seed: int = 7
    num_skills: int = 2
    envs_per_skill: int = 256
    iterations: int = 400
    episode_length: int = 40
    policy_grid_size: int = 25
    raw_feature_grid_size: int = 9
    action_scale: float = 0.055
    learning_rate: float = 0.35
    epsilon_start: float = 0.25
    epsilon_end: float = 0.04
    discriminator_decay: float = 0.96
    discriminator_pseudocount: float = 2.0
    baseline_alpha: float = 0.08
    start_position: tuple[float, float] = (-0.35, 0.0)
    start_noise: float = 0.018
    eval_episodes_per_skill: int = 1024

    def __post_init__(self) -> None:
        if self.representation not in {"raw", "semantic", "random"}:
            raise ValueError("representation must be raw, semantic, or random")
        if self.num_skills != 2:
            raise ValueError("the first Point-Cup probe is intentionally fixed to two skills")
        if min(
            self.envs_per_skill,
            self.iterations,
            self.episode_length,
            self.policy_grid_size,
            self.raw_feature_grid_size,
            self.eval_episodes_per_skill,
        ) <= 0:
            raise ValueError("training sizes must be positive")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)


def _sample_categorical(probabilities: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    draws = rng.random(len(probabilities))
    cumulative = np.cumsum(probabilities, axis=-1)
    return np.sum(draws[:, None] > cumulative, axis=-1).astype(np.int64)


def _grid_ids(positions: np.ndarray, grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.floor((np.clip(positions, -1.0, 1.0) + 1.0) * 0.5 * grid_size).astype(np.int64)
    coordinates = np.clip(coordinates, 0, grid_size - 1)
    ids = coordinates[:, 0] * grid_size + coordinates[:, 1]
    return coordinates[:, 0], coordinates[:, 1], ids


def _mutual_information(skills: np.ndarray, features: np.ndarray, skill_count: int, feature_count: int) -> float:
    counts = np.zeros((skill_count, feature_count), dtype=np.float64)
    np.add.at(counts, (skills, features), 1.0)
    total = counts.sum()
    if total == 0:
        return 0.0
    joint = counts / total
    skill_probability = joint.sum(axis=1, keepdims=True)
    feature_probability = joint.sum(axis=0, keepdims=True)
    expected = skill_probability * feature_probability
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log2(joint[mask] / expected[mask])))


def _xy_coverage(visited_grid_ids: np.ndarray, grid_size: int) -> float:
    return float(len(np.unique(visited_grid_ids)) / (grid_size * grid_size))


def _episode_batch(
    policy_logits: np.ndarray,
    config: TrainConfig,
    rng: np.random.Generator,
    epsilon: float,
) -> dict[str, np.ndarray | float]:
    num_envs = config.num_skills * config.envs_per_skill
    skills = np.repeat(np.arange(config.num_skills, dtype=np.int64), config.envs_per_skill)
    env = PointCupEnv(
        ShapeWorldConfig(
            num_envs=num_envs,
            episode_length=config.episode_length,
            action_scale=config.action_scale,
            seed=config.seed,
        )
    )
    starts = np.broadcast_to(np.asarray(config.start_position, dtype=np.float32), (num_envs, 2)).copy()
    starts += rng.normal(0.0, config.start_noise, size=starts.shape).astype(np.float32)
    env.reset(positions=starts)

    state_x = np.empty((config.episode_length, num_envs), dtype=np.int16)
    state_y = np.empty_like(state_x)
    actions = np.empty_like(state_x)
    policy_probabilities = np.empty(
        (config.episode_length, num_envs, len(ACTION_VECTORS)),
        dtype=np.float32,
    )
    visited_ids = np.empty((config.episode_length + 1, num_envs), dtype=np.int32)
    position_history = np.empty((config.episode_length + 1, num_envs, 2), dtype=np.float32)
    _, _, visited_ids[0] = _grid_ids(env.positions, config.policy_grid_size)
    position_history[0] = env.positions

    for step in range(config.episode_length):
        grid_x, grid_y, _ = _grid_ids(env.positions, config.policy_grid_size)
        probabilities = _softmax(policy_logits[skills, grid_x, grid_y])
        behavior = (1.0 - epsilon) * probabilities + epsilon / len(ACTION_VECTORS)
        sampled_actions = _sample_categorical(behavior, rng)
        env.step(ACTION_VECTORS[sampled_actions])
        state_x[step] = grid_x
        state_y[step] = grid_y
        actions[step] = sampled_actions
        policy_probabilities[step] = probabilities
        _, _, visited_ids[step + 1] = _grid_ids(env.positions, config.policy_grid_size)
        position_history[step + 1] = env.positions

    _, _, terminal_raw_features = _grid_ids(env.positions, config.raw_feature_grid_size)
    terminal_inside = env.observe()["relation_features"][:, 0].astype(np.int64)
    return {
        "skills": skills,
        "state_x": state_x,
        "state_y": state_y,
        "actions": actions,
        "policy_probabilities": policy_probabilities,
        "terminal_raw_features": terminal_raw_features,
        "terminal_inside": terminal_inside,
        "visited_ids": visited_ids,
        "terminal_positions": env.positions.copy(),
        "position_history": position_history,
    }


def _discriminator_rewards(
    skills: np.ndarray,
    features: np.ndarray,
    counts: np.ndarray,
    config: TrainConfig,
) -> tuple[np.ndarray, np.ndarray]:
    batch_counts = np.zeros_like(counts)
    np.add.at(batch_counts, (skills, features), 1.0)
    next_counts = config.discriminator_decay * counts + batch_counts
    posterior = next_counts / next_counts.sum(axis=0, keepdims=True)
    rewards = np.log(np.maximum(posterior[skills, features], 1.0e-8) * config.num_skills)
    return rewards.astype(np.float32), next_counts


def _policy_update(
    policy_logits: np.ndarray,
    batch: dict[str, np.ndarray | float],
    advantages: np.ndarray,
    config: TrainConfig,
) -> None:
    skills = np.asarray(batch["skills"])
    state_x = np.asarray(batch["state_x"])
    state_y = np.asarray(batch["state_y"])
    actions = np.asarray(batch["actions"])
    probabilities = np.asarray(batch["policy_probabilities"])
    scale = config.learning_rate / config.envs_per_skill

    for step in range(config.episode_length):
        for action_id in range(len(ACTION_VECTORS)):
            gradient = advantages * ((actions[step] == action_id).astype(np.float32) - probabilities[step, :, action_id])
            np.add.at(
                policy_logits[..., action_id],
                (skills, state_x[step], state_y[step]),
                scale * gradient,
            )
    np.clip(policy_logits, -12.0, 12.0, out=policy_logits)


def _iteration_metrics(
    batch: dict[str, np.ndarray | float],
    rewards: np.ndarray,
    config: TrainConfig,
) -> dict[str, float | list[float]]:
    skills = np.asarray(batch["skills"])
    inside = np.asarray(batch["terminal_inside"])
    raw_features = np.asarray(batch["terminal_raw_features"])
    inside_rates = [float(inside[skills == skill].mean()) for skill in range(config.num_skills)]
    reward_means = [float(rewards[skills == skill].mean()) for skill in range(config.num_skills)]
    return {
        "inside_rates": inside_rates,
        "inside_rate_gap": float(abs(inside_rates[0] - inside_rates[1])),
        "semantic_mi_bits": _mutual_information(skills, inside, config.num_skills, 2),
        "endpoint_grid_mi_bits": _mutual_information(
            skills,
            raw_features,
            config.num_skills,
            config.raw_feature_grid_size**2,
        ),
        "xy_coverage": _xy_coverage(np.asarray(batch["visited_ids"]), config.policy_grid_size),
        "reward_means": reward_means,
        "reward_mean": float(rewards.mean()),
    }


def train(config: TrainConfig) -> tuple[np.ndarray, list[dict[str, float | int | list[float]]]]:
    rng = np.random.default_rng(config.seed)
    policy_logits = np.zeros(
        (config.num_skills, config.policy_grid_size, config.policy_grid_size, len(ACTION_VECTORS)),
        dtype=np.float32,
    )
    if config.representation == "raw":
        feature_count = config.raw_feature_grid_size**2
    else:
        feature_count = 2
    discriminator_counts = np.full(
        (config.num_skills, feature_count),
        config.discriminator_pseudocount,
        dtype=np.float64,
    )
    baselines = np.zeros(config.num_skills, dtype=np.float32)
    history: list[dict[str, float | int | list[float]]] = []

    for iteration in range(config.iterations):
        progress = iteration / max(config.iterations - 1, 1)
        epsilon = config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)
        batch = _episode_batch(policy_logits, config, rng, epsilon)
        skills = np.asarray(batch["skills"])
        if config.representation == "raw":
            features = np.asarray(batch["terminal_raw_features"])
        else:
            features = np.asarray(batch["terminal_inside"])

        if config.representation == "random":
            rewards = np.zeros(len(skills), dtype=np.float32)
        else:
            rewards, discriminator_counts = _discriminator_rewards(
                skills,
                features,
                discriminator_counts,
                config,
            )
            advantages = rewards - baselines[skills]
            for skill in range(config.num_skills):
                skill_advantages = advantages[skills == skill]
                scale = max(float(skill_advantages.std()), 0.1)
                advantages[skills == skill] = skill_advantages / scale
                reward_mean = float(rewards[skills == skill].mean())
                baselines[skill] = (
                    (1.0 - config.baseline_alpha) * baselines[skill]
                    + config.baseline_alpha * reward_mean
                )
            _policy_update(policy_logits, batch, np.clip(advantages, -5.0, 5.0), config)

        metrics = _iteration_metrics(batch, rewards, config)
        history.append({"iteration": iteration + 1, "epsilon": float(epsilon), **metrics})

    return policy_logits, history


def evaluate(policy_logits: np.ndarray, config: TrainConfig, seed: int) -> dict[str, float | list[float]]:
    eval_config = TrainConfig(
        **{
            **asdict(config),
            "envs_per_skill": config.eval_episodes_per_skill,
            "iterations": 1,
            "seed": seed,
        }
    )
    rng = np.random.default_rng(seed)
    batch = _episode_batch(policy_logits, eval_config, rng, epsilon=0.0)
    zero_rewards = np.zeros(config.num_skills * config.eval_episodes_per_skill, dtype=np.float32)
    metrics = _iteration_metrics(batch, zero_rewards, eval_config)
    skills = np.asarray(batch["skills"])
    positions = np.asarray(batch["terminal_positions"])
    metrics["terminal_position_mean"] = [
        positions[skills == skill].mean(axis=0).tolist() for skill in range(config.num_skills)
    ]
    metrics["terminal_position_std"] = [
        positions[skills == skill].std(axis=0).tolist() for skill in range(config.num_skills)
    ]
    inside_rates = list(metrics["inside_rates"])
    metrics["specialization_gate_passed"] = bool(
        max(inside_rates) >= 0.8 and min(inside_rates) <= 0.2
    )
    return metrics


def _write_curves(path: Path, history: list[dict[str, float | int | list[float]]]) -> None:
    width, height = 820, 380
    left, top, chart_width, chart_height = 60, 52, 720, 260
    colors = ("#2477a5", "#d96a2b")
    points = []
    for skill in range(2):
        values = np.asarray([row["inside_rates"][skill] for row in history], dtype=np.float64)
        coordinates = []
        for index, value in enumerate(values):
            x = left + chart_width * index / max(len(values) - 1, 1)
            y = top + chart_height * (1.0 - value)
            coordinates.append(f"{x:.1f},{y:.1f}")
        points.append(
            f'<polyline points="{" ".join(coordinates)}" fill="none" stroke="{colors[skill]}" stroke-width="2"/>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fbfcfd"/>
<text x="30" y="30" font-family="sans-serif" font-size="20" fill="#172b3a">Point-Cup training: final inside rate</text>
<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#83919a"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#83919a"/>
<line x1="{left}" y1="{top + 0.2 * chart_height}" x2="{left + chart_width}" y2="{top + 0.2 * chart_height}" stroke="#b8c2c8" stroke-dasharray="4 4"/>
<line x1="{left}" y1="{top + 0.8 * chart_height}" x2="{left + chart_width}" y2="{top + 0.8 * chart_height}" stroke="#b8c2c8" stroke-dasharray="4 4"/>
<text x="20" y="{top + 5}" font-family="sans-serif" font-size="12">1.0</text>
<text x="20" y="{top + chart_height + 5}" font-family="sans-serif" font-size="12">0.0</text>
<text x="{left}" y="{height - 28}" font-family="sans-serif" font-size="12">iteration 1</text>
<text x="{left + chart_width - 90}" y="{height - 28}" font-family="sans-serif" font-size="12">iteration {len(history)}</text>
{''.join(points)}
<rect x="590" y="18" width="14" height="4" fill="{colors[0]}"/><text x="610" y="25" font-family="sans-serif" font-size="12">skill 0</text>
<rect x="680" y="18" width="14" height="4" fill="{colors[1]}"/><text x="700" y="25" font-family="sans-serif" font-size="12">skill 1</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", choices=("raw", "semantic", "random"), default="semantic")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--envs-per-skill", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = TrainConfig(
        representation=args.representation,
        seed=args.seed,
        iterations=args.iterations,
        envs_per_skill=args.envs_per_skill,
    )
    run_id = f"tabular_diayn_{args.representation}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path("outputs/skill_discovery/point_cup_training") / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    policy_logits, history = train(config)
    evaluation = evaluate(policy_logits, config, seed=args.seed + 10000)
    stability_window = history[-min(20, len(history)) :]
    stability_inside_rates = np.asarray([row["inside_rates"] for row in stability_window])
    stability_gate = (
        stability_inside_rates.max(axis=1) >= 0.8
    ) & (
        stability_inside_rates.min(axis=1) <= 0.2
    )
    training_stability = {
        "window": len(stability_window),
        "inside_rates_mean": stability_inside_rates.mean(axis=0).tolist(),
        "inside_rates_std": stability_inside_rates.std(axis=0).tolist(),
        "semantic_mi_bits_mean": float(
            np.mean([row["semantic_mi_bits"] for row in stability_window])
        ),
        "specialization_gate_fraction": float(stability_gate.mean()),
        "passed": bool(stability_gate.mean() >= 0.8),
    }
    output = {
        "run_id": run_id,
        "config": asdict(config),
        "evaluation": evaluation,
        "training_stability": training_stability,
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
