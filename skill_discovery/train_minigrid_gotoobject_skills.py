"""Train mission-free tabular skills on MiniGrid GoToObject."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from skill_discovery.generate_point_cup_dataset import _write_png
from skill_discovery.minigrid_gotoobject import (
    GOTOOBJECT_STAGES,
    floor_objects,
    make_gotoobject,
    semantic_stage,
)
from skill_discovery.train_pusher_diayn import _best_class_assignment


POLICY_ACTIONS = (0, 1, 2, 3, 4)
RelationKey = tuple[int, int, int, int, int, int, int, int]


@dataclass(frozen=True)
class GoToObjectTrainConfig:
    objective: str = "semantic_balanced"
    frozen_reward_matrix: tuple[tuple[float, ...], ...] | None = None
    frozen_reward_source: str | None = None
    frozen_reward_calibration: str = "none"
    frozen_reward_target_order: tuple[int, ...] | None = None
    reward_timing: str = "exact_terminal"
    seed: int = 7
    num_skills: int = 3
    episodes: int = 100_000
    horizon: int = 64
    gamma: float = 0.99
    learning_rate_mode: str = "visit_power"
    constant_learning_rate: float = 0.1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.0
    epsilon_decay_fraction: float = 0.8
    pseudocount: float = 2.0
    semantic_decay: float = 0.9995
    semantic_coverage_weight: float = 1.0
    eval_interval: int = 5_000
    eval_episodes_per_skill: int = 1_024
    stability_checkpoints: int = 5
    stage_rate_gates: tuple[float, float, float] = (0.90, 0.90, 0.90)

    def __post_init__(self) -> None:
        if self.objective not in {
            "random",
            "semantic",
            "semantic_spread",
            "semantic_balanced",
            "frozen_matrix",
        }:
            raise ValueError("unknown objective")
        if self.objective == "frozen_matrix":
            matrix = np.asarray(self.frozen_reward_matrix, dtype=np.float64)
            if matrix.shape != (self.num_skills, len(GOTOOBJECT_STAGES)):
                raise ValueError("frozen reward matrix must have shape (3, 3)")
            if not np.isfinite(matrix).all():
                raise ValueError("frozen reward matrix must be finite")
        elif self.frozen_reward_matrix is not None:
            raise ValueError("frozen reward matrix requires frozen_matrix objective")
        if self.frozen_reward_calibration not in {"none", "runner_up_unit"}:
            raise ValueError("unknown frozen reward calibration")
        if (
            self.objective != "frozen_matrix"
            and (
                self.frozen_reward_calibration != "none"
                or self.frozen_reward_target_order is not None
            )
        ):
            raise ValueError(
                "frozen reward transforms require frozen_matrix objective"
            )
        if self.frozen_reward_target_order is not None and sorted(
            self.frozen_reward_target_order
        ) != list(range(self.num_skills)):
            raise ValueError("frozen reward target order must be a stage permutation")
        if self.reward_timing not in {"exact_terminal", "occupancy"}:
            raise ValueError("unknown reward timing")
        if self.learning_rate_mode not in {"visit_power", "constant"}:
            raise ValueError("unknown learning rate mode")
        if not 0 < self.constant_learning_rate <= 1:
            raise ValueError("constant learning rate must be in (0, 1]")
        if self.num_skills != len(GOTOOBJECT_STAGES):
            raise ValueError("GoToObject probe is fixed to three skills")
        if self.episodes <= 0 or self.horizon <= 0 or self.eval_interval <= 0:
            raise ValueError("episode, horizon, and eval interval must be positive")
        if self.stability_checkpoints <= 0:
            raise ValueError("stability checkpoints must be positive")
        if not 0 < self.epsilon_decay_fraction <= 1:
            raise ValueError("epsilon decay fraction must be in (0, 1]")
        if len(self.stage_rate_gates) != len(GOTOOBJECT_STAGES):
            raise ValueError("one rate gate is required per stage")


def compact_relation_key(env: object) -> RelationKey:
    base = env.unwrapped
    positions = sorted(obj.position for obj in floor_objects(env))
    while len(positions) < 2:
        positions.append((-1, -1))
    if len(positions) != 2:
        raise ValueError("GoToObject compact key expects two total objects")
    return (
        int(base.agent_pos[0]),
        int(base.agent_pos[1]),
        int(base.agent_dir),
        int(positions[0][0]),
        int(positions[0][1]),
        int(positions[1][0]),
        int(positions[1][1]),
        int(base.carrying is not None),
    )


class GoToObjectReward:
    def __init__(self, config: GoToObjectTrainConfig):
        self.config = config
        self.counts = np.full(
            (config.num_skills, len(GOTOOBJECT_STAGES)),
            config.pseudocount,
            dtype=np.float64,
        )
        self.reward_calls_by_skill = np.zeros(config.num_skills, dtype=np.int64)
        self.stage_counts = np.zeros(len(GOTOOBJECT_STAGES), dtype=np.int64)
        self.balanced_targets = np.random.default_rng(
            config.seed + 70_000
        ).permutation(config.num_skills)

    def reward(self, skill: int, stage: int) -> tuple[float, dict[str, float]]:
        self.reward_calls_by_skill[skill] += 1
        self.stage_counts[stage] += 1
        if self.config.objective == "random":
            return 0.0, {"diayn_reward": 0.0, "coverage_reward": 0.0}
        if self.config.objective == "frozen_matrix":
            assert self.config.frozen_reward_matrix is not None
            value = float(self.config.frozen_reward_matrix[skill][stage])
            return value, {"diayn_reward": value, "coverage_reward": 0.0}
        if self.config.objective == "semantic_balanced":
            value = float(stage == self.balanced_targets[skill])
            return value, {"diayn_reward": value, "coverage_reward": 0.0}
        self.counts *= self.config.semantic_decay
        self.counts[skill, stage] += 1.0
        posterior = self.counts[skill, stage] / self.counts[:, stage].sum()
        diayn = math.log(max(posterior * self.config.num_skills, 1.0e-8))
        coverage = 0.0
        if self.config.objective == "semantic_spread":
            totals = self.counts.sum(axis=0)
            probability = totals[stage] / totals.sum()
            coverage = -math.log(
                max(len(GOTOOBJECT_STAGES) * probability, 1.0e-8)
            )
        total = diayn + self.config.semantic_coverage_weight * coverage
        return total, {"diayn_reward": diayn, "coverage_reward": coverage}

    def state_dict(self) -> dict[str, object]:
        return {
            "objective": self.config.objective,
            "semantic_counts": self.counts.tolist(),
            "reward_calls_by_skill": self.reward_calls_by_skill.tolist(),
            "episode_counts": self.reward_calls_by_skill.tolist()
            if self.config.reward_timing == "exact_terminal"
            else None,
            "stage_counts": self.stage_counts.tolist(),
            "balanced_targets": self.balanced_targets.tolist()
            if self.config.objective == "semantic_balanced"
            else None,
            "frozen_reward_matrix": self.config.frozen_reward_matrix,
            "frozen_reward_source": self.config.frozen_reward_source,
        }


def frozen_reward_matrix_from_metrics(
    metrics: dict[str, object],
    *,
    calibration: str = "none",
    target_stage_order: tuple[int, ...] | None = None,
) -> tuple[tuple[float, ...], ...]:
    config = metrics["config"]
    reward_model = metrics["reward_model"]
    if not isinstance(config, dict) or not isinstance(reward_model, dict):
        raise ValueError("source metrics config and reward_model must be mappings")
    objective = reward_model["objective"]
    if objective not in {"semantic", "semantic_spread"}:
        raise ValueError("frozen reward source must be semantic or semantic_spread")
    counts = np.asarray(reward_model["semantic_counts"], dtype=np.float64)
    if counts.shape != (3, 3) or not np.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("source semantic counts must be a finite positive 3x3 matrix")
    totals = counts.sum(axis=0)
    posterior = counts / totals[None, :]
    reward = np.log(np.maximum(posterior * 3, 1.0e-8))
    if objective == "semantic_spread":
        probabilities = totals / totals.sum()
        coverage = -np.log(np.maximum(3 * probabilities, 1.0e-8))
        reward += float(config["semantic_coverage_weight"]) * coverage[None, :]
    if calibration == "runner_up_unit":
        calibrated = np.empty_like(reward)
        for skill, row in enumerate(reward):
            ordered = np.sort(row)
            runner_up, top = ordered[-2], ordered[-1]
            gap = top - runner_up
            if gap <= 1.0e-8:
                raise ValueError("frozen reward row has no unique top stage")
            calibrated[skill] = (row - runner_up) / gap
        reward = calibrated
    elif calibration != "none":
        raise ValueError("unknown frozen reward calibration")
    if target_stage_order is not None:
        if sorted(target_stage_order) != list(range(reward.shape[0])):
            raise ValueError("target stage order must be a permutation")
        top_stage_by_row = np.argmax(reward, axis=1)
        if sorted(int(stage) for stage in top_stage_by_row) != list(
            range(reward.shape[1])
        ):
            raise ValueError("frozen reward rows do not have unique top stages")
        row_by_stage = {
            int(stage): row for row, stage in enumerate(top_stage_by_row)
        }
        reward = np.stack([reward[row_by_stage[stage]] for stage in target_stage_order])
    return tuple(tuple(float(value) for value in row) for row in reward)


def _epsilon(config: GoToObjectTrainConfig, episode: int) -> float:
    decay = max(int(config.episodes * config.epsilon_decay_fraction), 1)
    progress = min(episode / decay, 1.0)
    return config.epsilon_start + progress * (
        config.epsilon_end - config.epsilon_start
    )


def _training_action(
    values: np.ndarray,
    rng: np.random.Generator,
) -> int:
    maximum = values.max()
    candidates = np.flatnonzero(np.isclose(values, maximum))
    return int(rng.choice(candidates))


def _transition_reward(
    reward_model: GoToObjectReward,
    config: GoToObjectTrainConfig,
    skill: int,
    stage: int,
    *,
    terminal: bool,
) -> tuple[float, dict[str, float]]:
    if config.reward_timing == "occupancy" or terminal:
        return reward_model.reward(skill, stage)
    return 0.0, {"diayn_reward": 0.0, "coverage_reward": 0.0}


def _learning_rate(config: GoToObjectTrainConfig, visits: float) -> float:
    if config.learning_rate_mode == "constant":
        return config.constant_learning_rate
    return float(visits**-0.6)


def _values(
    table: dict[RelationKey, np.ndarray],
    key: RelationKey,
    *,
    create: bool,
) -> np.ndarray:
    values = table.get(key)
    if values is None:
        if not create:
            return np.zeros((3, len(POLICY_ACTIONS)), dtype=np.float32)
        values = np.zeros((3, len(POLICY_ACTIONS)), dtype=np.float32)
        table[key] = values
    return values


def _rollout(
    q_table: dict[RelationKey, np.ndarray],
    config: GoToObjectTrainConfig,
    skill: int,
    *,
    seed: int,
    render: bool = False,
) -> dict[str, object]:
    env = make_gotoobject(render_mode="rgb_array" if render else None)
    try:
        observation, _ = env.reset(seed=seed)
        mission = str(observation["mission"])
        frames = [env.render()] if render else []
        actions = []
        keys = [compact_relation_key(env)]
        terminated = truncated = False
        for _ in range(config.horizon):
            key = compact_relation_key(env)
            action_index = int(np.argmax(_values(q_table, key, create=False)[skill]))
            action = POLICY_ACTIONS[action_index]
            _, _, terminated, truncated, _ = env.step(action)
            actions.append(action)
            keys.append(compact_relation_key(env))
            if render:
                frames.append(env.render())
            if terminated or truncated:
                break
        return {
            "stage": semantic_stage(env),
            "stage_name": GOTOOBJECT_STAGES[semantic_stage(env)],
            "steps": len(actions),
            "actions": actions,
            "relation_keys": [list(key) for key in keys],
            "mission": mission,
            "carrying": None
            if env.unwrapped.carrying is None
            else [env.unwrapped.carrying.type, env.unwrapped.carrying.color],
            "floor_object_count": len(floor_objects(env)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "frames": frames,
        }
    finally:
        env.close()


def evaluate_q_table(
    q_table: dict[RelationKey, np.ndarray],
    config: GoToObjectTrainConfig,
    *,
    seed: int,
) -> dict[str, object]:
    rates = np.zeros((config.num_skills, len(GOTOOBJECT_STAGES)), dtype=np.float64)
    mean_steps = []
    for skill in range(config.num_skills):
        stages = []
        steps = []
        for episode in range(config.eval_episodes_per_skill):
            rollout = _rollout(
                q_table,
                config,
                skill,
                seed=seed + episode,
            )
            stages.append(int(rollout["stage"]))
            steps.append(int(rollout["steps"]))
        rates[skill] = np.bincount(stages, minlength=3) / len(stages)
        mean_steps.append(float(np.mean(steps)))
    assignment, matched_rates = _best_class_assignment(rates)
    matched_by_stage = np.zeros(3, dtype=np.float64)
    for skill, stage in enumerate(assignment):
        matched_by_stage[stage] = matched_rates[skill]
    return {
        "episodes_per_skill": config.eval_episodes_per_skill,
        "outcome_rates": rates.tolist(),
        "stage_assignment": assignment,
        "stage_assignment_names": [GOTOOBJECT_STAGES[index] for index in assignment],
        "matched_stage_rates": matched_rates,
        "matched_rate_by_stage": {
            name: float(matched_by_stage[index])
            for index, name in enumerate(GOTOOBJECT_STAGES)
        },
        "mean_episode_steps": mean_steps,
        "specialization_gate_passed": bool(
            all(
                matched_by_stage[index] >= config.stage_rate_gates[index]
                for index in range(3)
            )
        ),
    }


def _write_curves(path: Path, evaluations: list[dict[str, object]]) -> None:
    width, height = 900, 420
    left, top, chart_width, chart_height = 65, 55, 770, 290
    colors = ("#2875a4", "#c6483a", "#2b895f")
    lines = []
    for stage in range(3):
        points = []
        for index, row in enumerate(evaluations):
            assignment = row["stage_assignment"]
            skill = assignment.index(stage)
            value = row["matched_stage_rates"][skill]
            x = left + chart_width * index / max(len(evaluations) - 1, 1)
            y = top + chart_height * (1.0 - value)
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{colors[stage]}" stroke-width="2"/>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fbfcfd"/>
<text x="30" y="30" font-family="sans-serif" font-size="19" fill="#172b3a">GoToObject mission-free relation rates</text>
<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#83919a"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#83919a"/>
<text x="22" y="{top + 5}" font-family="sans-serif" font-size="12">1.0</text>
<text x="22" y="{top + chart_height + 5}" font-family="sans-serif" font-size="12">0.0</text>
{''.join(lines)}
</svg>'''
    path.write_text(svg, encoding="utf-8")


def _write_rollout_audit(
    output_dir: Path,
    q_table: dict[RelationKey, np.ndarray],
    config: GoToObjectTrainConfig,
    assignment: list[int],
    rates: list[float],
) -> list[dict[str, object]]:
    frame_size = 192
    columns = 4
    gap = 6
    marker = 9
    sheet = np.full(
        (
            3 * frame_size + 2 * gap,
            marker + columns * frame_size + (columns - 1) * gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    colors = ((48, 116, 173), (198, 72, 58), (42, 137, 94))
    manifest = []
    for skill in range(3):
        target = assignment[skill]
        target_rate = rates[skill]
        attempts = 1 if target_rate <= 0 else min(10_000, max(32, int(10 / target_rate)))
        rollout = None
        fallback = None
        matched_offset = -1
        for offset in range(attempts):
            candidate = _rollout(
                q_table,
                config,
                skill,
                seed=config.seed + 990_000 + offset,
                render=True,
            )
            fallback = fallback or candidate
            if int(candidate["stage"]) == target:
                rollout = candidate
                matched_offset = offset
                break
        if rollout is None:
            assert fallback is not None
            rollout = fallback
        frames = rollout.pop("frames")
        indices = np.linspace(0, len(frames) - 1, columns).astype(np.int64)
        y = skill * (frame_size + gap)
        actual = int(rollout["stage"])
        sheet[y : y + frame_size, :marker] = colors[actual]
        for column, index in enumerate(indices):
            x = marker + column * (frame_size + gap)
            sheet[y : y + frame_size, x : x + frame_size] = frames[int(index)]
        manifest.append(
            {
                "skill": skill,
                "assigned_stage": GOTOOBJECT_STAGES[target],
                "actual_stage": GOTOOBJECT_STAGES[actual],
                "assigned_stage_found": matched_offset >= 0,
                "matched_seed_offset": matched_offset,
                **{key: value for key, value in rollout.items() if key != "stage"},
            }
        )
    _write_png(output_dir / "policy_rollout_audit.png", sheet)
    (output_dir / "policy_rollout_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def train_run(
    config: GoToObjectTrainConfig,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(config.seed)
    q_table: dict[RelationKey, np.ndarray] = {}
    visits: dict[RelationKey, np.ndarray] = {}
    reward_model = GoToObjectReward(config)
    training_counts = np.zeros((3, 3), dtype=np.int64)
    reward_sums = {"diayn_reward": 0.0, "coverage_reward": 0.0}
    evaluations = []
    env = make_gotoobject()
    started = time.monotonic()
    try:
        for episode in range(config.episodes):
            skill = episode % 3
            env.reset(seed=config.seed * 1_000_000 + episode)
            epsilon = _epsilon(config, episode)
            for step in range(config.horizon):
                key = compact_relation_key(env)
                q_values = _values(q_table, key, create=True)
                visit_values = _values(visits, key, create=True)
                if rng.random() < epsilon:
                    action_index = int(rng.integers(len(POLICY_ACTIONS)))
                else:
                    action_index = _training_action(q_values[skill], rng)
                _, _, terminated, truncated, _ = env.step(POLICY_ACTIONS[action_index])
                terminal = bool(terminated or truncated or step + 1 == config.horizon)
                stage = semantic_stage(env)
                reward, parts = _transition_reward(
                    reward_model,
                    config,
                    skill,
                    stage,
                    terminal=terminal,
                )
                for name in reward_sums:
                    reward_sums[name] += parts[name]
                if terminal:
                    target = reward
                    training_counts[skill, stage] += 1
                else:
                    next_key = compact_relation_key(env)
                    target = reward + config.gamma * _values(
                        q_table, next_key, create=True
                    )[skill].max()
                visit_values[skill, action_index] += 1
                step_size = _learning_rate(
                    config,
                    float(visit_values[skill, action_index]),
                )
                q_values[skill, action_index] += step_size * (
                    target - q_values[skill, action_index]
                )
                if terminal:
                    break
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
    final = evaluate_q_table(q_table, config, seed=config.seed + 900_000)
    recent = evaluations[-min(config.stability_checkpoints, len(evaluations)) :]
    stability = bool(
        len(recent) >= config.stability_checkpoints
        and all(row["specialization_gate_passed"] for row in recent)
    )
    manifest = _write_rollout_audit(
        output_dir,
        q_table,
        config,
        final["stage_assignment"],
        final["matched_stage_rates"],
    )
    keys = sorted(q_table)
    np.savez_compressed(
        output_dir / "q_table.npz",
        relation_keys=np.asarray(keys, dtype=np.int16),
        q_values=np.stack([q_table[key] for key in keys]),
        visits=np.stack([visits[key] for key in keys]),
    )
    output = {
        "config": asdict(config),
        "elapsed_seconds": elapsed,
        "policy_state_fields": [
            "agent_x",
            "agent_y",
            "agent_dir",
            "object_0_x",
            "object_0_y",
            "object_1_x",
            "object_1_y",
            "carrying_bit",
        ],
        "excluded_policy_fields": [
            "mission",
            "targetType",
            "target_color",
            "target_pos",
            "object_type",
            "object_color",
            "native_reward",
        ],
        "q_state_count": len(q_table),
        "reward_model": reward_model.state_dict(),
        "training_stage_counts": training_counts.tolist(),
        "reward_sums": reward_sums,
        "evaluations": evaluations,
        "final_evaluation": final,
        "checkpoint_stability_passed": stability,
        "signal_gate_passed": bool(final["specialization_gate_passed"] and stability),
        "policy_rollout_manifest": manifest,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    _write_curves(output_dir / "relation_curves.svg", evaluations)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objective",
        choices=(
            "random",
            "semantic",
            "semantic_spread",
            "semantic_balanced",
            "frozen_matrix",
        ),
        default="semantic_balanced",
    )
    parser.add_argument("--frozen-reward-metrics", type=Path)
    parser.add_argument(
        "--frozen-reward-calibration",
        choices=("none", "runner_up_unit"),
        default="none",
    )
    parser.add_argument(
        "--frozen-reward-target-order",
        help="Comma-separated top-stage order for matrix rows, for example 2,0,1",
    )
    parser.add_argument(
        "--reward-timing",
        choices=("exact_terminal", "occupancy"),
        default="exact_terminal",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument(
        "--learning-rate-mode",
        choices=("visit_power", "constant"),
        default="visit_power",
    )
    parser.add_argument("--constant-learning-rate", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=1_024)
    parser.add_argument("--stability-checkpoints", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    frozen_matrix = None
    frozen_source = None
    frozen_target_order = None
    if args.frozen_reward_target_order is not None:
        try:
            frozen_target_order = tuple(
                int(part) for part in args.frozen_reward_target_order.split(",")
            )
        except ValueError:
            parser.error("--frozen-reward-target-order must contain integers")
        if sorted(frozen_target_order) != list(range(len(GOTOOBJECT_STAGES))):
            parser.error("--frozen-reward-target-order must be a stage permutation")
    if args.objective == "frozen_matrix":
        if args.frozen_reward_metrics is None:
            parser.error("frozen_matrix requires --frozen-reward-metrics")
        source_metrics = json.loads(
            args.frozen_reward_metrics.read_text(encoding="utf-8")
        )
        frozen_matrix = frozen_reward_matrix_from_metrics(
            source_metrics,
            calibration=args.frozen_reward_calibration,
            target_stage_order=frozen_target_order,
        )
        frozen_source = str(args.frozen_reward_metrics.resolve())
    elif args.frozen_reward_metrics is not None:
        parser.error("--frozen-reward-metrics requires frozen_matrix objective")
    elif args.frozen_reward_calibration != "none" or frozen_target_order is not None:
        parser.error("frozen reward transforms require frozen_matrix objective")
    config = GoToObjectTrainConfig(
        objective=args.objective,
        frozen_reward_matrix=frozen_matrix,
        frozen_reward_source=frozen_source,
        frozen_reward_calibration=args.frozen_reward_calibration,
        frozen_reward_target_order=frozen_target_order,
        reward_timing=args.reward_timing,
        seed=args.seed,
        episodes=args.episodes,
        horizon=args.horizon,
        learning_rate_mode=args.learning_rate_mode,
        constant_learning_rate=args.constant_learning_rate,
        eval_interval=args.eval_interval,
        eval_episodes_per_skill=args.eval_episodes,
        stability_checkpoints=args.stability_checkpoints,
    )
    run_id = (
        f"gotoobject_{config.objective}_{config.reward_timing}_seed{config.seed}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/minigrid_gotoobject_training"
    ) / run_id
    output = train_run(config, output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "elapsed_seconds": output["elapsed_seconds"],
                "q_state_count": output["q_state_count"],
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
