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
    seed: int = 7
    num_skills: int = 3
    episodes: int = 100_000
    horizon: int = 64
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.0
    epsilon_decay_fraction: float = 0.8
    pseudocount: float = 2.0
    semantic_decay: float = 0.9995
    semantic_coverage_weight: float = 1.0
    eval_interval: int = 5_000
    eval_episodes_per_skill: int = 1_024
    stage_rate_gates: tuple[float, float, float] = (0.90, 0.90, 0.90)

    def __post_init__(self) -> None:
        if self.objective not in {
            "random",
            "semantic",
            "semantic_spread",
            "semantic_balanced",
        }:
            raise ValueError("unknown objective")
        if self.num_skills != len(GOTOOBJECT_STAGES):
            raise ValueError("GoToObject probe is fixed to three skills")
        if self.episodes <= 0 or self.horizon <= 0 or self.eval_interval <= 0:
            raise ValueError("episode, horizon, and eval interval must be positive")
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
        self.episode_counts = np.zeros(config.num_skills, dtype=np.int64)
        self.stage_counts = np.zeros(len(GOTOOBJECT_STAGES), dtype=np.int64)
        self.balanced_targets = np.random.default_rng(
            config.seed + 70_000
        ).permutation(config.num_skills)

    def reward(self, skill: int, stage: int) -> tuple[float, dict[str, float]]:
        self.episode_counts[skill] += 1
        self.stage_counts[stage] += 1
        if self.config.objective == "random":
            return 0.0, {"diayn_reward": 0.0, "coverage_reward": 0.0}
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
            "episode_counts": self.episode_counts.tolist(),
            "stage_counts": self.stage_counts.tolist(),
            "balanced_targets": self.balanced_targets.tolist()
            if self.config.objective == "semantic_balanced"
            else None,
        }


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
                if terminal:
                    stage = semantic_stage(env)
                    reward, parts = reward_model.reward(skill, stage)
                    target = reward
                    training_counts[skill, stage] += 1
                    for name in reward_sums:
                        reward_sums[name] += parts[name]
                else:
                    next_key = compact_relation_key(env)
                    target = config.gamma * _values(
                        q_table,
                        next_key,
                        create=True,
                    )[skill].max()
                visit_values[skill, action_index] += 1
                step_size = float(visit_values[skill, action_index] ** -0.6)
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
    recent = evaluations[-min(5, len(evaluations)) :]
    stability = bool(
        len(recent) >= 5
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
        choices=("random", "semantic", "semantic_spread", "semantic_balanced"),
        default="semantic_balanced",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=1_024)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = GoToObjectTrainConfig(
        objective=args.objective,
        seed=args.seed,
        episodes=args.episodes,
        horizon=args.horizon,
        eval_interval=args.eval_interval,
        eval_episodes_per_skill=args.eval_episodes,
    )
    run_id = (
        f"gotoobject_{config.objective}_seed{config.seed}_"
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
