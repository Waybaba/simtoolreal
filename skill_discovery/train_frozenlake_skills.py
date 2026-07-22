"""Train tabular skill-conditioned policies on official FrozenLake."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from skill_discovery.frozenlake import (
    FROZENLAKE_OUTCOMES,
    FrozenLakeConfig,
    classify_outcome,
    make_frozenlake,
)
from skill_discovery.generate_point_cup_dataset import _write_png
from skill_discovery.train_pusher_diayn import _best_class_assignment


@dataclass(frozen=True)
class FrozenLakeTrainConfig:
    objective: str = "semantic_spread"
    seed: int = 7
    num_skills: int = 3
    episodes: int = 30_000
    learning_rate: float = 0.15
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.0
    epsilon_decay_fraction: float = 0.90
    pseudocount: float = 2.0
    semantic_decay: float = 0.9995
    semantic_coverage_weight: float = 1.0
    eval_interval: int = 1_000
    eval_episodes_per_skill: int = 32
    class_rate_gate: float = 0.95
    lake: FrozenLakeConfig = FrozenLakeConfig()

    def __post_init__(self) -> None:
        if self.objective not in {
            "random",
            "raw",
            "semantic",
            "semantic_spread",
            "semantic_balanced",
        }:
            raise ValueError("unknown objective")
        if self.num_skills != len(FROZENLAKE_OUTCOMES):
            raise ValueError("FrozenLake probe is fixed to three skills")
        if self.episodes <= 0 or self.eval_interval <= 0:
            raise ValueError("episode and evaluation intervals must be positive")
        if not 0 < self.epsilon_decay_fraction <= 1:
            raise ValueError("epsilon decay fraction must be in (0, 1]")


class FrozenLakeIntrinsicReward:
    def __init__(self, config: FrozenLakeTrainConfig):
        self.config = config
        self.semantic_counts = np.full(
            (config.num_skills, len(FROZENLAKE_OUTCOMES)),
            config.pseudocount,
            dtype=np.float64,
        )
        self.raw_counts: dict[int, np.ndarray] = {}
        self.episode_counts = np.zeros(config.num_skills, dtype=np.int64)
        self.outcome_counts = np.zeros(len(FROZENLAKE_OUTCOMES), dtype=np.int64)
        self.balanced_targets = np.random.default_rng(
            config.seed + 70_000
        ).permutation(config.num_skills)

    def reward(
        self,
        skill: int,
        outcome: int,
        terminal_state: int,
    ) -> tuple[float, dict[str, float]]:
        self.episode_counts[skill] += 1
        self.outcome_counts[outcome] += 1
        if self.config.objective == "random":
            return 0.0, {"diayn_reward": 0.0, "coverage_reward": 0.0}
        if self.config.objective == "semantic_balanced":
            value = float(outcome == self.balanced_targets[skill])
            return value, {"diayn_reward": value, "coverage_reward": 0.0}
        if self.config.objective == "raw":
            counts = self.raw_counts.setdefault(
                terminal_state,
                np.full(
                    self.config.num_skills,
                    self.config.pseudocount,
                    dtype=np.float64,
                ),
            )
            counts[skill] += 1.0
            posterior = counts[skill] / counts.sum()
            diayn_reward = math.log(
                max(posterior * self.config.num_skills, 1.0e-8)
            )
            return diayn_reward, {
                "diayn_reward": diayn_reward,
                "coverage_reward": 0.0,
            }

        self.semantic_counts *= self.config.semantic_decay
        self.semantic_counts[skill, outcome] += 1.0
        posterior = (
            self.semantic_counts[skill, outcome]
            / self.semantic_counts[:, outcome].sum()
        )
        diayn_reward = math.log(
            max(posterior * self.config.num_skills, 1.0e-8)
        )
        coverage_reward = 0.0
        if self.config.objective == "semantic_spread":
            totals = self.semantic_counts.sum(axis=0)
            probability = totals[outcome] / totals.sum()
            coverage_reward = -math.log(
                max(len(FROZENLAKE_OUTCOMES) * probability, 1.0e-8)
            )
        total = diayn_reward + self.config.semantic_coverage_weight * coverage_reward
        return total, {
            "diayn_reward": diayn_reward,
            "coverage_reward": coverage_reward,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "objective": self.config.objective,
            "semantic_counts": self.semantic_counts.tolist(),
            "raw_feature_count": len(self.raw_counts),
            "episode_counts": self.episode_counts.tolist(),
            "outcome_counts": self.outcome_counts.tolist(),
            "balanced_targets": self.balanced_targets.tolist()
            if self.config.objective == "semantic_balanced"
            else None,
        }


def _epsilon(config: FrozenLakeTrainConfig, episode: int) -> float:
    decay_episodes = max(int(config.episodes * config.epsilon_decay_fraction), 1)
    progress = min(episode / decay_episodes, 1.0)
    return config.epsilon_start + progress * (
        config.epsilon_end - config.epsilon_start
    )


def _policy_action(q_values: np.ndarray, rng: np.random.Generator) -> int:
    maximum = q_values.max()
    candidates = np.flatnonzero(np.isclose(q_values, maximum))
    return int(rng.choice(candidates))


def _rollout_policy(
    q_table: np.ndarray,
    config: FrozenLakeTrainConfig,
    skill: int,
    *,
    seed: int,
    render: bool = False,
) -> dict[str, object]:
    env = make_frozenlake(
        config.lake,
        render_mode="rgb_array" if render else None,
    )
    rng = np.random.default_rng(seed + 90_000)
    try:
        state, _ = env.reset(seed=seed)
        states = [int(state)]
        actions = []
        frames = [env.render()] if render else []
        terminated = truncated = False
        native_reward = 0.0
        while not (terminated or truncated):
            action = _policy_action(q_table[skill, int(state)], rng)
            state, native_reward, terminated, truncated, _ = env.step(action)
            states.append(int(state))
            actions.append(action)
            if render:
                frames.append(env.render())
        outcome = classify_outcome(
            env,
            int(state),
            terminated=terminated,
            truncated=truncated,
        )
        return {
            "outcome": outcome,
            "terminal_state": int(state),
            "native_reward": float(native_reward),
            "steps": len(actions),
            "states": states,
            "actions": actions,
            "frames": frames,
        }
    finally:
        env.close()


def evaluate_q_table(
    q_table: np.ndarray,
    config: FrozenLakeTrainConfig,
    *,
    seed: int,
) -> dict[str, object]:
    rates = np.zeros(
        (config.num_skills, len(FROZENLAKE_OUTCOMES)),
        dtype=np.float64,
    )
    mean_steps = []
    native_success_rates = []
    for skill in range(config.num_skills):
        outcomes = []
        steps = []
        successes = []
        for episode in range(config.eval_episodes_per_skill):
            rollout = _rollout_policy(
                q_table,
                config,
                skill,
                seed=seed + skill * 1_000_000 + episode,
            )
            outcomes.append(int(rollout["outcome"]))
            steps.append(int(rollout["steps"]))
            successes.append(float(rollout["native_reward"]) > 0)
        rates[skill] = np.bincount(
            outcomes,
            minlength=len(FROZENLAKE_OUTCOMES),
        ) / config.eval_episodes_per_skill
        mean_steps.append(float(np.mean(steps)))
        native_success_rates.append(float(np.mean(successes)))
    assignment, matched_rates = _best_class_assignment(rates)
    goal_skill = assignment.index(2)
    return {
        "episodes_per_skill": config.eval_episodes_per_skill,
        "outcome_rates": rates.tolist(),
        "outcome_assignment": assignment,
        "outcome_assignment_names": [
            FROZENLAKE_OUTCOMES[index] for index in assignment
        ],
        "matched_outcome_rates": matched_rates,
        "matched_outcome_rate_mean": float(np.mean(matched_rates)),
        "matched_outcome_rate_min": float(np.min(matched_rates)),
        "goal_skill": goal_skill,
        "goal_skill_native_success_rate": native_success_rates[goal_skill],
        "native_success_rates": native_success_rates,
        "mean_episode_steps": mean_steps,
        "specialization_gate_passed": bool(
            min(matched_rates) >= config.class_rate_gate
            and native_success_rates[goal_skill] >= config.class_rate_gate
        ),
    }


def _write_curves(path: Path, evaluations: list[dict[str, object]]) -> None:
    width, height = 900, 420
    left, top, chart_width, chart_height = 65, 55, 770, 290
    colors = ("#2875a4", "#c6483a", "#2b895f")
    lines = []
    for outcome in range(len(FROZENLAKE_OUTCOMES)):
        values = []
        for row in evaluations:
            assignment = row["outcome_assignment"]
            skill = assignment.index(outcome)
            values.append(row["matched_outcome_rates"][skill])
        points = []
        for index, value in enumerate(values):
            x = left + chart_width * index / max(len(values) - 1, 1)
            y = top + chart_height * (1.0 - value)
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{colors[outcome]}" stroke-width="2"/>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fbfcfd"/>
<text x="30" y="30" font-family="sans-serif" font-size="19" fill="#172b3a">FrozenLake deterministic outcome rates</text>
<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#83919a"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#83919a"/>
<text x="22" y="{top + 5}" font-family="sans-serif" font-size="12">1.0</text>
<text x="22" y="{top + chart_height + 5}" font-family="sans-serif" font-size="12">0.0</text>
{''.join(lines)}
</svg>'''
    path.write_text(svg, encoding="utf-8")


def _write_rollout_audit(
    output_dir: Path,
    q_table: np.ndarray,
    config: FrozenLakeTrainConfig,
) -> list[dict[str, object]]:
    frame_size = 192
    columns = 4
    gap = 6
    marker_width = 10
    sheet = np.full(
        (
            config.num_skills * frame_size + (config.num_skills - 1) * gap,
            marker_width + columns * frame_size + (columns - 1) * gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    colors = ((48, 116, 173), (198, 72, 58), (42, 137, 94))
    manifest = []
    for skill in range(config.num_skills):
        rollout = _rollout_policy(
            q_table,
            config,
            skill,
            seed=config.seed + 990_000 + skill,
            render=True,
        )
        frames = rollout.pop("frames")
        indices = np.linspace(0, len(frames) - 1, columns).astype(np.int64)
        y = skill * (frame_size + gap)
        outcome = int(rollout["outcome"])
        sheet[y : y + frame_size, :marker_width] = colors[outcome]
        for column, index in enumerate(indices):
            frame = frames[int(index)]
            rows = np.linspace(0, frame.shape[0] - 1, frame_size).astype(np.int64)
            cols = np.linspace(0, frame.shape[1] - 1, frame_size).astype(np.int64)
            resized = frame[rows][:, cols]
            x = marker_width + column * (frame_size + gap)
            sheet[y : y + frame_size, x : x + frame_size] = resized
        manifest.append(
            {
                "skill": skill,
                "outcome": FROZENLAKE_OUTCOMES[outcome],
                **rollout,
            }
        )
    _write_png(output_dir / "policy_rollout_audit.png", sheet)
    (output_dir / "policy_rollout_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def train_run(
    config: FrozenLakeTrainConfig,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(config.seed)
    q_table = rng.normal(0.0, 1.0e-4, size=(config.num_skills, 16, 4))
    reward_model = FrozenLakeIntrinsicReward(config)
    training_counts = np.zeros(
        (config.num_skills, len(FROZENLAKE_OUTCOMES)),
        dtype=np.int64,
    )
    reward_sums = {"diayn_reward": 0.0, "coverage_reward": 0.0}
    evaluations = []
    env = make_frozenlake(config.lake)
    started = time.monotonic()
    try:
        for episode in range(config.episodes):
            skill = episode % config.num_skills
            state, _ = env.reset(seed=config.seed * 1_000_000 + episode)
            terminated = truncated = False
            epsilon = _epsilon(config, episode)
            while not (terminated or truncated):
                if rng.random() < epsilon:
                    action = int(rng.integers(env.action_space.n))
                else:
                    action = _policy_action(q_table[skill, int(state)], rng)
                next_state, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    outcome = classify_outcome(
                        env,
                        int(next_state),
                        terminated=terminated,
                        truncated=truncated,
                    )
                    reward, parts = reward_model.reward(
                        skill,
                        outcome,
                        int(next_state),
                    )
                    target = reward
                    training_counts[skill, outcome] += 1
                    for name in reward_sums:
                        reward_sums[name] += parts[name]
                else:
                    target = config.gamma * q_table[skill, int(next_state)].max()
                q_table[skill, int(state), action] += config.learning_rate * (
                    target - q_table[skill, int(state), action]
                )
                state = next_state
            if (episode + 1) % config.eval_interval == 0:
                evaluation = evaluate_q_table(
                    q_table,
                    config,
                    seed=config.seed + 100_000 + episode,
                )
                evaluations.append({"episodes": episode + 1, **evaluation})
    finally:
        env.close()
    elapsed = time.monotonic() - started
    final_evaluation = evaluate_q_table(
        q_table,
        config,
        seed=config.seed + 900_000,
    )
    recent = evaluations[-min(5, len(evaluations)) :]
    stability = bool(
        len(recent) >= 3
        and all(row["specialization_gate_passed"] for row in recent)
    )
    manifest = _write_rollout_audit(output_dir, q_table, config)
    output = {
        "config": asdict(config),
        "version": {"gymnasium": importlib.metadata.version("gymnasium")},
        "elapsed_seconds": elapsed,
        "reward_model": reward_model.state_dict(),
        "training_outcome_counts": training_counts.tolist(),
        "reward_sums": reward_sums,
        "evaluations": evaluations,
        "final_evaluation": final_evaluation,
        "checkpoint_stability_passed": stability,
        "signal_gate_passed": bool(
            final_evaluation["specialization_gate_passed"] and stability
        ),
        "policy_rollout_manifest": manifest,
    }
    np.savez_compressed(output_dir / "q_table.npz", q_table=q_table)
    (output_dir / "metrics.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    _write_curves(output_dir / "outcome_curves.svg", evaluations)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objective",
        choices=(
            "random",
            "raw",
            "semantic",
            "semantic_spread",
            "semantic_balanced",
        ),
        default="semantic_spread",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=30_000)
    parser.add_argument("--eval-interval", type=int, default=1_000)
    parser.add_argument("--epsilon-decay-fraction", type=float, default=0.90)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = FrozenLakeTrainConfig(
        objective=args.objective,
        seed=args.seed,
        episodes=args.episodes,
        eval_interval=args.eval_interval,
        epsilon_decay_fraction=args.epsilon_decay_fraction,
    )
    run_id = (
        f"frozenlake_{config.objective}_seed{config.seed}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/frozenlake_training"
    ) / run_id
    output = train_run(config, output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "elapsed_seconds": output["elapsed_seconds"],
                "final_evaluation": output["final_evaluation"],
                "checkpoint_stability_passed": output[
                    "checkpoint_stability_passed"
                ],
                "signal_gate_passed": output["signal_gate_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
