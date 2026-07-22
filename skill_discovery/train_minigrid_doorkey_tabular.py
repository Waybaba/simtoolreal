"""Train tabular compositional controls on official MiniGrid DoorKey."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from minigrid.core.actions import Actions

from skill_discovery.generate_point_cup_dataset import _write_png
from skill_discovery.minigrid_doorkey import DOORKEY_STAGES, semantic_stage


DOORKEY_POLICY_ACTIONS = (
    int(Actions.left),
    int(Actions.right),
    int(Actions.forward),
    int(Actions.pickup),
    int(Actions.toggle),
)
DOORKEY_CONTROL_MATRIX = (
    (1.0, -1.0, -1.0, -1.0),
    (0.0, 1.0, -1.0, -1.0),
    (0.0, 0.0, 1.0, -1.0),
    (0.0, 0.0, 0.0, 1.0),
)
DoorKeyState = tuple[int, ...]


@dataclass(frozen=True)
class DoorKeyTabularConfig:
    env_id: str = "MiniGrid-DoorKey-5x5-v0"
    seed: int = 7
    num_skills: int = 4
    episodes: int = 20_000
    horizon: int = 64
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.0
    epsilon_decay_fraction: float = 0.8
    evaluation_checkpoints: tuple[int, ...] = (
        4_000,
        8_000,
        12_000,
        18_000,
        19_000,
        20_000,
    )
    eval_episodes_per_skill: int = 512
    stability_checkpoints: int = 3
    stage_rate_gate: float = 0.80

    def __post_init__(self) -> None:
        if self.num_skills != len(DOORKEY_STAGES):
            raise ValueError("DoorKey control is fixed to four skills")
        if self.episodes <= 0 or self.episodes % self.num_skills != 0:
            raise ValueError("episodes must be positive and divisible by four")
        if self.horizon <= 0 or self.eval_episodes_per_skill <= 0:
            raise ValueError("horizon and evaluation episodes must be positive")
        if not 0 < self.epsilon_decay_fraction <= 1:
            raise ValueError("epsilon decay fraction must be in (0, 1]")
        if tuple(sorted(set(self.evaluation_checkpoints))) != (
            self.evaluation_checkpoints
        ):
            raise ValueError("evaluation checkpoints must be sorted and unique")
        if not self.evaluation_checkpoints:
            raise ValueError("at least one evaluation checkpoint is required")
        if self.evaluation_checkpoints[-1] > self.episodes:
            raise ValueError("evaluation checkpoint exceeds episode budget")
        if len(self.evaluation_checkpoints) < self.stability_checkpoints:
            raise ValueError("not enough checkpoints for stability gate")
        if not 0 <= self.stage_rate_gate <= 1:
            raise ValueError("stage rate gate must be in [0, 1]")


def common_layout_seed(seed: int, episode: int) -> int:
    return seed * 1_000_000 + episode // len(DOORKEY_STAGES)


def compact_doorkey_state(env: gym.Env) -> DoorKeyState:
    base = env.unwrapped
    key_position = (-1, -1)
    door_position = None
    goal_position = None
    door_open = 0
    door_locked = 0
    for x in range(base.width):
        for y in range(base.height):
            obj = base.grid.get(x, y)
            if obj is None:
                continue
            if obj.type == "key":
                key_position = (x, y)
            elif obj.type == "door":
                door_position = (x, y)
                door_open = int(bool(obj.is_open))
                door_locked = int(bool(obj.is_locked))
            elif obj.type == "goal":
                goal_position = (x, y)
    if door_position is None or goal_position is None:
        raise ValueError("DoorKey state requires one door and one goal")
    carrying_key = int(
        base.carrying is not None and base.carrying.type == "key"
    )
    return (
        int(base.agent_pos[0]),
        int(base.agent_pos[1]),
        int(base.agent_dir),
        key_position[0],
        key_position[1],
        carrying_key,
        door_position[0],
        door_position[1],
        door_open,
        door_locked,
        goal_position[0],
        goal_position[1],
    )


def _values(
    table: dict[DoorKeyState, np.ndarray],
    key: DoorKeyState,
    *,
    create: bool,
) -> np.ndarray:
    values = table.get(key)
    if values is None:
        shape = (len(DOORKEY_STAGES), len(DOORKEY_POLICY_ACTIONS))
        values = np.zeros(shape, dtype=np.float32)
        if create:
            table[key] = values
    return values


def _epsilon(config: DoorKeyTabularConfig, episode: int) -> float:
    decay = max(int(config.episodes * config.epsilon_decay_fraction), 1)
    progress = min(episode / decay, 1.0)
    return config.epsilon_start + progress * (
        config.epsilon_end - config.epsilon_start
    )


def _training_action(values: np.ndarray, rng: np.random.Generator) -> int:
    candidates = np.flatnonzero(np.isclose(values, values.max()))
    return int(rng.choice(candidates))


def _rollout(
    q_table: dict[DoorKeyState, np.ndarray],
    config: DoorKeyTabularConfig,
    skill: int,
    *,
    seed: int,
    render: bool = False,
) -> dict[str, object]:
    env = gym.make(config.env_id, render_mode="rgb_array" if render else None)
    frames = []
    try:
        env.reset(seed=seed)
        if render:
            frames.append(env.render())
        furthest_stage = semantic_stage(env)
        actions = []
        keys = []
        native_reward = 0.0
        terminated = truncated = False
        for _ in range(config.horizon):
            key = compact_doorkey_state(env)
            keys.append(key)
            values = _values(q_table, key, create=False)
            action_index = int(np.argmax(values[skill]))
            action = DOORKEY_POLICY_ACTIONS[action_index]
            _, reward, terminated, truncated, _ = env.step(action)
            native_reward = float(reward)
            stage = semantic_stage(
                env,
                terminated=bool(terminated),
                reward=native_reward,
            )
            furthest_stage = max(furthest_stage, stage)
            actions.append(action)
            if render:
                frames.append(env.render())
            if terminated or truncated:
                break
        base = env.unwrapped
        carrying = None
        if base.carrying is not None:
            carrying = [base.carrying.type, base.carrying.color]
        objects = (
            base.grid.get(x, y)
            for x in range(base.width)
            for y in range(base.height)
        )
        door_open = any(
            obj is not None and obj.type == "door" and bool(obj.is_open)
            for obj in objects
        )
        return {
            "stage": furthest_stage,
            "stage_name": DOORKEY_STAGES[furthest_stage],
            "steps": len(actions),
            "actions": actions,
            "state_keys": keys,
            "native_success": bool(terminated and native_reward > 0),
            "native_reward": native_reward,
            "carrying": carrying,
            "door_open": bool(door_open),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "frames": frames,
        }
    finally:
        env.close()


def evaluate_q_table(
    q_table: dict[DoorKeyState, np.ndarray],
    config: DoorKeyTabularConfig,
    *,
    seed: int,
) -> dict[str, object]:
    rates = np.zeros((4, 4), dtype=np.float64)
    mean_steps = []
    native_success_rates = []
    for skill in range(config.num_skills):
        stages = []
        steps = []
        successes = []
        for episode in range(config.eval_episodes_per_skill):
            rollout = _rollout(q_table, config, skill, seed=seed + episode)
            stages.append(int(rollout["stage"]))
            steps.append(int(rollout["steps"]))
            successes.append(bool(rollout["native_success"]))
        rates[skill] = np.bincount(stages, minlength=4) / len(stages)
        mean_steps.append(float(np.mean(steps)))
        native_success_rates.append(float(np.mean(successes)))
    target_rates = np.diag(rates).astype(float).tolist()
    by_stage = {
        stage: target_rates[index] for index, stage in enumerate(DOORKEY_STAGES)
    }
    gate = bool(
        all(rate >= config.stage_rate_gate for rate in target_rates)
        and native_success_rates[3] >= config.stage_rate_gate
    )
    return {
        "episodes_per_skill": config.eval_episodes_per_skill,
        "outcome_rates": rates.tolist(),
        "target_stage_rates": target_rates,
        "target_rate_by_stage": by_stage,
        "native_success_rates": native_success_rates,
        "goal_native_success_rate": native_success_rates[3],
        "mean_episode_steps": mean_steps,
        "specialization_gate_passed": gate,
    }


def _save_q_table(
    path: Path,
    q_table: dict[DoorKeyState, np.ndarray],
    visits: dict[DoorKeyState, np.ndarray],
) -> None:
    keys = sorted(q_table)
    np.savez_compressed(
        path,
        state_keys=np.asarray(keys, dtype=np.int16),
        q_values=np.stack([q_table[key] for key in keys]),
        visits=np.stack([visits[key] for key in keys]),
    )


def _write_curves(path: Path, evaluations: list[dict[str, object]]) -> None:
    width, height = 900, 420
    left, top, chart_width, chart_height = 65, 55, 770, 290
    colors = ("#68757d", "#2875a4", "#d17031", "#2b895f")
    lines = []
    for stage, name in enumerate(DOORKEY_STAGES):
        points = []
        for index, row in enumerate(evaluations):
            value = row["target_rate_by_stage"][name]
            x = left + chart_width * index / max(len(evaluations) - 1, 1)
            y = top + chart_height * (1.0 - value)
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{colors[stage]}" stroke-width="2"/>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#fbfcfd"/>'
        '<text x="30" y="30" font-family="sans-serif" font-size="19" '
        'fill="#172b3a">DoorKey tabular target-stage rates</text>'
        f'<line x1="{left}" y1="{top + chart_height}" '
        f'x2="{left + chart_width}" y2="{top + chart_height}" '
        'stroke="#83919a"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{top + chart_height}" stroke="#83919a"/>'
        f'<text x="22" y="{top + 5}" font-family="sans-serif" '
        'font-size="12">1.0</text>'
        f'<text x="22" y="{top + chart_height + 5}" '
        'font-family="sans-serif" font-size="12">0.0</text>'
        f'{"".join(lines)}</svg>'
    )
    path.write_text(svg, encoding="utf-8")


def _write_rollout_audit(
    output_dir: Path,
    q_table: dict[DoorKeyState, np.ndarray],
    config: DoorKeyTabularConfig,
    target_rates: list[float],
) -> list[dict[str, object]]:
    frame_size = 160
    columns = 4
    gap = 6
    marker = 9
    sheet = np.full(
        (
            4 * frame_size + 3 * gap,
            marker + columns * frame_size + (columns - 1) * gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    colors = ((104, 117, 125), (40, 117, 164), (209, 112, 49), (43, 137, 95))
    manifest = []
    for skill in range(4):
        rate = target_rates[skill]
        attempts = 1 if rate <= 0 else min(10_000, max(32, int(10 / rate)))
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
            if int(candidate["stage"]) == skill:
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
                "target_stage": DOORKEY_STAGES[skill],
                "actual_stage": DOORKEY_STAGES[actual],
                "target_stage_found": matched_offset >= 0,
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
    config: DoorKeyTabularConfig,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    env = gym.make(config.env_id)
    q_table: dict[DoorKeyState, np.ndarray] = {}
    visits: dict[DoorKeyState, np.ndarray] = {}
    rng = np.random.default_rng(config.seed)
    terminal_counts = np.zeros((4, 4), dtype=np.int64)
    native_success_counts = np.zeros(4, dtype=np.int64)
    evaluations: list[dict[str, object]] = []
    started = time.monotonic()
    try:
        for episode in range(config.episodes):
            skill = episode % config.num_skills
            env.reset(seed=common_layout_seed(config.seed, episode))
            furthest_stage = semantic_stage(env)
            epsilon = _epsilon(config, episode)
            for step in range(config.horizon):
                key = compact_doorkey_state(env)
                q_values = _values(q_table, key, create=True)
                visit_values = _values(visits, key, create=True)
                if rng.random() < epsilon:
                    action_index = int(rng.integers(len(DOORKEY_POLICY_ACTIONS)))
                else:
                    action_index = _training_action(q_values[skill], rng)
                _, native_reward, terminated, truncated, _ = env.step(
                    DOORKEY_POLICY_ACTIONS[action_index]
                )
                stage = semantic_stage(
                    env,
                    terminated=bool(terminated),
                    reward=float(native_reward),
                )
                furthest_stage = max(furthest_stage, stage)
                reward = DOORKEY_CONTROL_MATRIX[skill][furthest_stage]
                terminal = bool(terminated or truncated or step + 1 == config.horizon)
                if terminal:
                    target = reward
                    terminal_counts[skill, furthest_stage] += 1
                    native_success_counts[skill] += int(
                        bool(terminated) and native_reward > 0
                    )
                else:
                    next_key = compact_doorkey_state(env)
                    target = reward + config.gamma * _values(
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
            if episode + 1 in config.evaluation_checkpoints:
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
    recent = evaluations[-config.stability_checkpoints :]
    stability = bool(
        len(recent) == config.stability_checkpoints
        and all(row["specialization_gate_passed"] for row in recent)
    )
    _save_q_table(output_dir / "q_table.npz", q_table, visits)
    manifest = _write_rollout_audit(
        output_dir,
        q_table,
        config,
        final["target_stage_rates"],
    )
    _write_curves(output_dir / "stage_curves.svg", evaluations)
    output = {
        "config": asdict(config),
        "reward_matrix": DOORKEY_CONTROL_MATRIX,
        "policy_actions": DOORKEY_POLICY_ACTIONS,
        "elapsed_seconds": elapsed,
        "training_layout_seed_mode": "common_per_four_skill_cycle",
        "terminal_counts": terminal_counts.tolist(),
        "native_success_counts": native_success_counts.tolist(),
        "q_state_count": len(q_table),
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
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--evaluation-checkpoints")
    parser.add_argument("--eval-episodes", type=int, default=512)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    checkpoints = (
        tuple(int(value) for value in args.evaluation_checkpoints.split(","))
        if args.evaluation_checkpoints
        else DoorKeyTabularConfig.evaluation_checkpoints
    )
    config = DoorKeyTabularConfig(
        seed=args.seed,
        episodes=args.episodes,
        horizon=args.horizon,
        evaluation_checkpoints=checkpoints,
        eval_episodes_per_skill=args.eval_episodes,
    )
    run_id = (
        f"doorkey5_tabular_balanced_control_seed{config.seed}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/minigrid_doorkey_training"
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
