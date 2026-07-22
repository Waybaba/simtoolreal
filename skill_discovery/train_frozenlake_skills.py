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
    learning_rate_schedule: str = "constant"
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
    outcome_rate_gates: tuple[float, float, float] | None = None
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
        if self.learning_rate_schedule not in {"constant", "visit_decay"}:
            raise ValueError("unknown learning-rate schedule")
        if not 0 < self.epsilon_decay_fraction <= 1:
            raise ValueError("epsilon decay fraction must be in (0, 1]")
        if self.outcome_rate_gates is not None:
            if len(self.outcome_rate_gates) != len(FROZENLAKE_OUTCOMES):
                raise ValueError("one rate gate is required per outcome")
            if any(not 0 <= gate <= 1 for gate in self.outcome_rate_gates):
                raise ValueError("outcome rate gates must be in [0, 1]")

    def resolved_outcome_rate_gates(self) -> tuple[float, float, float]:
        if self.outcome_rate_gates is not None:
            return tuple(self.outcome_rate_gates)
        return (self.class_rate_gate,) * len(FROZENLAKE_OUTCOMES)


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


def _step_size(config: FrozenLakeTrainConfig, visit_count: int) -> float:
    if config.learning_rate_schedule == "constant":
        return config.learning_rate
    return float(visit_count**-0.6)


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
    matched_by_outcome = np.zeros(len(FROZENLAKE_OUTCOMES), dtype=np.float64)
    for skill, outcome in enumerate(assignment):
        matched_by_outcome[outcome] = matched_rates[skill]
    rate_gates = config.resolved_outcome_rate_gates()
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
        "matched_rate_by_outcome": {
            name: float(matched_by_outcome[index])
            for index, name in enumerate(FROZENLAKE_OUTCOMES)
        },
        "outcome_rate_gates": {
            name: float(rate_gates[index])
            for index, name in enumerate(FROZENLAKE_OUTCOMES)
        },
        "goal_skill": goal_skill,
        "goal_skill_native_success_rate": native_success_rates[goal_skill],
        "native_success_rates": native_success_rates,
        "mean_episode_steps": mean_steps,
        "specialization_gate_passed": bool(
            all(
                matched_by_outcome[index] >= rate_gates[index]
                for index in range(len(FROZENLAKE_OUTCOMES))
            )
            and native_success_rates[goal_skill] >= rate_gates[2]
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
    assigned_outcomes: list[int],
    assigned_rates: list[float],
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
        target_outcome = assigned_outcomes[skill]
        rollout = None
        fallback = None
        matched_offset = -1
        target_rate = assigned_rates[skill]
        attempt_limit = (
            1
            if target_rate <= 0
            else min(10_000, max(32, int(math.ceil(10.0 / target_rate))))
        )
        for offset in range(attempt_limit):
            candidate = _rollout_policy(
                q_table,
                config,
                skill,
                seed=config.seed + 990_000 + skill * 100_000 + offset,
                render=True,
            )
            if fallback is None:
                fallback = candidate
            if int(candidate["outcome"]) == target_outcome:
                rollout = candidate
                matched_offset = offset
                break
        if rollout is None:
            assert fallback is not None
            rollout = fallback
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
                "assigned_outcome": FROZENLAKE_OUTCOMES[target_outcome],
                "actual_outcome": FROZENLAKE_OUTCOMES[outcome],
                "matched_seed_offset": matched_offset,
                "assigned_outcome_found": matched_offset >= 0,
                **{key: value for key, value in rollout.items() if key != "outcome"},
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
    visit_counts = np.zeros_like(q_table, dtype=np.int64)
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
                visit_counts[skill, int(state), action] += 1
                step_size = _step_size(
                    config,
                    int(visit_counts[skill, int(state), action]),
                )
                q_table[skill, int(state), action] += step_size * (
                    target - q_table[skill, int(state), action]
                )
                state = next_state
            if (episode + 1) % config.eval_interval == 0:
                evaluation = evaluate_q_table(
                    q_table,
                    config,
                    seed=config.seed + 100_000,
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
    manifest = _write_rollout_audit(
        output_dir,
        q_table,
        config,
        final_evaluation["outcome_assignment"],
        final_evaluation["matched_outcome_rates"],
    )
    output = {
        "config": asdict(config),
        "version": {"gymnasium": importlib.metadata.version("gymnasium")},
        "elapsed_seconds": elapsed,
        "reward_model": reward_model.state_dict(),
        "training_outcome_counts": training_counts.tolist(),
        "reward_sums": reward_sums,
        "visit_counts": {
            "nonzero": int(np.count_nonzero(visit_counts)),
            "maximum": int(visit_counts.max()),
            "by_skill": visit_counts.sum(axis=(1, 2)).tolist(),
        },
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
    parser.add_argument(
        "--learning-rate-schedule",
        choices=("constant", "visit_decay"),
        default="constant",
    )
    parser.add_argument("--eval-interval", type=int, default=1_000)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--epsilon-decay-fraction", type=float, default=0.90)
    parser.add_argument("--slippery", action="store_true")
    parser.add_argument(
        "--outcome-rate-gates",
        type=float,
        nargs=3,
        metavar=("SAFE", "HOLE", "GOAL"),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = FrozenLakeTrainConfig(
        objective=args.objective,
        seed=args.seed,
        episodes=args.episodes,
        learning_rate_schedule=args.learning_rate_schedule,
        eval_interval=args.eval_interval,
        eval_episodes_per_skill=args.eval_episodes,
        epsilon_decay_fraction=args.epsilon_decay_fraction,
        outcome_rate_gates=tuple(args.outcome_rate_gates)
        if args.outcome_rate_gates is not None
        else None,
        lake=FrozenLakeConfig(is_slippery=args.slippery),
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
