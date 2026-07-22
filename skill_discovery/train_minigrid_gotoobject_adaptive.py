"""Train frozen GoToObject skills with reward-deficit allocation."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from skill_discovery.minigrid_gotoobject import (
    GOTOOBJECT_STAGES,
    make_gotoobject,
    semantic_stage,
)
from skill_discovery.train_minigrid_gotoobject_blockwise import _save_q_table
from skill_discovery.train_minigrid_gotoobject_skills import (
    POLICY_ACTIONS,
    GoToObjectReward,
    GoToObjectTrainConfig,
    RelationKey,
    _learning_rate,
    _training_action,
    _values,
    _write_curves,
    _write_rollout_audit,
    compact_relation_key,
    evaluate_q_table,
)


@dataclass(frozen=True)
class AdaptiveAllocationConfig:
    seed: int = 7
    num_skills: int = 3
    policy_episodes: int = 15_000
    horizon: int = 64
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.0
    epsilon_decay_fraction: float = 0.8
    ema_alpha: float = 0.05
    scheduler_signal: str = "raw_reward"
    eval_interval: int = 3_000
    eval_episodes_per_skill: int = 512
    stability_checkpoints: int = 3
    stage_rate_gates: tuple[float, float, float] = (0.90, 0.90, 0.90)

    def __post_init__(self) -> None:
        if self.num_skills != len(GOTOOBJECT_STAGES):
            raise ValueError("GoToObject adaptive probe is fixed to three skills")
        if self.policy_episodes <= 0 or self.policy_episodes % 4 != 0:
            raise ValueError("policy episodes must be positive and divisible by four")
        if self.horizon <= 0 or self.eval_interval <= 0:
            raise ValueError("horizon and eval interval must be positive")
        if self.eval_interval % 4 != 0:
            raise ValueError("eval interval must end on a four-episode cycle")
        if self.policy_episodes % self.eval_interval != 0:
            raise ValueError("policy episodes must be divisible by eval interval")
        if not 0 < self.epsilon_decay_fraction <= 1:
            raise ValueError("epsilon decay fraction must be in (0, 1]")
        if not 0 < self.ema_alpha <= 1:
            raise ValueError("EMA alpha must be in (0, 1]")
        if self.scheduler_signal not in {"raw_reward", "top_reward_indicator"}:
            raise ValueError("unknown scheduler signal")
        if self.stability_checkpoints <= 0:
            raise ValueError("stability checkpoints must be positive")
        if len(self.stage_rate_gates) != len(GOTOOBJECT_STAGES):
            raise ValueError("one rate gate is required per stage")


def load_bootstrap_matrix(path: Path) -> tuple[tuple[float, ...], ...]:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if not metrics.get("bootstrap_gate_passed"):
        raise ValueError("source bootstrap gate did not pass")
    matrix = np.asarray(metrics.get("bootstrap_calibrated_matrix"), dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("source bootstrap matrix must be a finite 3x3 matrix")
    top_stages = np.argmax(matrix, axis=1).astype(int).tolist()
    if sorted(top_stages) != list(range(3)):
        raise ValueError("source bootstrap rows must have distinct top stages")
    return tuple(tuple(float(value) for value in row) for row in matrix)


def select_deficit_skill(
    ema_scores: np.ndarray,
    rng: np.random.Generator,
) -> int:
    minimum = float(ema_scores.min())
    candidates = np.flatnonzero(np.isclose(ema_scores, minimum, atol=1.0e-12))
    return int(rng.choice(candidates))


def scheduler_observation(
    matrix: tuple[tuple[float, ...], ...],
    skill: int,
    terminal_reward: float,
    signal: str,
) -> float:
    if signal == "raw_reward":
        return terminal_reward
    if signal == "top_reward_indicator":
        return float(
            np.isclose(terminal_reward, max(matrix[skill]), atol=1.0e-12)
        )
    raise ValueError("unknown scheduler signal")


def _epsilon(config: AdaptiveAllocationConfig, episode: int) -> float:
    decay = max(int(config.policy_episodes * config.epsilon_decay_fraction), 1)
    progress = min(episode / decay, 1.0)
    return config.epsilon_start + progress * (
        config.epsilon_end - config.epsilon_start
    )


def _policy_config(
    config: AdaptiveAllocationConfig,
    matrix: tuple[tuple[float, ...], ...],
    source_metrics: Path,
) -> GoToObjectTrainConfig:
    return GoToObjectTrainConfig(
        objective="frozen_matrix",
        frozen_reward_matrix=matrix,
        frozen_reward_source=str(source_metrics.resolve()),
        frozen_reward_calibration="runner_up_unit",
        reward_timing="occupancy",
        seed=config.seed,
        episodes=config.policy_episodes,
        horizon=config.horizon,
        gamma=config.gamma,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay_fraction=config.epsilon_decay_fraction,
        eval_interval=config.eval_interval,
        eval_episodes_per_skill=config.eval_episodes_per_skill,
        stability_checkpoints=config.stability_checkpoints,
        stage_rate_gates=config.stage_rate_gates,
    )


def _update_ema(current: float, value: float, alpha: float) -> float:
    return (1.0 - alpha) * current + alpha * value


def _train_episode(
    env: object,
    config: AdaptiveAllocationConfig,
    policy_config: GoToObjectTrainConfig,
    reward_model: GoToObjectReward,
    q_table: dict[RelationKey, np.ndarray],
    visits: dict[RelationKey, np.ndarray],
    action_rng: np.random.Generator,
    *,
    skill: int,
    layout_seed: int,
    episode: int,
) -> dict[str, float | int]:
    env.reset(seed=layout_seed)
    epsilon = _epsilon(config, episode)
    reward_sum = 0.0
    terminal_reward = 0.0
    stage = 0
    for step in range(config.horizon):
        key = compact_relation_key(env)
        q_values = _values(q_table, key, create=True)
        visit_values = _values(visits, key, create=True)
        if action_rng.random() < epsilon:
            action_index = int(action_rng.integers(len(POLICY_ACTIONS)))
        else:
            action_index = _training_action(q_values[skill], action_rng)
        _, _, terminated, truncated, _ = env.step(POLICY_ACTIONS[action_index])
        terminal = bool(terminated or truncated or step + 1 == config.horizon)
        stage = semantic_stage(env)
        reward, _ = reward_model.reward(skill, stage)
        reward_sum += reward
        if terminal:
            target = reward
            terminal_reward = reward
        else:
            next_key = compact_relation_key(env)
            target = reward + config.gamma * _values(
                q_table, next_key, create=True
            )[skill].max()
        visit_values[skill, action_index] += 1
        step_size = _learning_rate(
            policy_config,
            float(visit_values[skill, action_index]),
        )
        q_values[skill, action_index] += step_size * (
            target - q_values[skill, action_index]
        )
        if terminal:
            return {
                "terminal_reward": float(terminal_reward),
                "reward_sum": float(reward_sum),
                "terminal_stage": int(stage),
                "steps": step + 1,
            }
    raise AssertionError("positive horizon must produce a terminal update")


def train_adaptive_run(
    config: AdaptiveAllocationConfig,
    source_metrics: Path,
    output_dir: Path,
) -> dict[str, object]:
    matrix = load_bootstrap_matrix(source_metrics)
    output_dir.mkdir(parents=True, exist_ok=False)
    policy_config = _policy_config(config, matrix, source_metrics)
    reward_model = GoToObjectReward(policy_config)
    q_table: dict[RelationKey, np.ndarray] = {}
    visits: dict[RelationKey, np.ndarray] = {}
    action_rng = np.random.default_rng(config.seed)
    scheduler_rng = np.random.default_rng(config.seed + 800_000)
    ema_scores = np.zeros(config.num_skills, dtype=np.float64)
    core_counts = np.zeros(config.num_skills, dtype=np.int64)
    extra_counts = np.zeros(config.num_skills, dtype=np.int64)
    terminal_reward_sums = np.zeros(config.num_skills, dtype=np.float64)
    terminal_counts = np.zeros((3, 3), dtype=np.int64)
    ema_trace: list[dict[str, object]] = []
    evaluations: list[dict[str, object]] = []
    total_episode = 0
    started = time.monotonic()
    env = make_gotoobject()
    try:
        for cycle in range(config.policy_episodes // 4):
            layout_seed = config.seed * 1_000_000 + 500_000 + cycle
            for skill in range(config.num_skills):
                result = _train_episode(
                    env,
                    config,
                    policy_config,
                    reward_model,
                    q_table,
                    visits,
                    action_rng,
                    skill=skill,
                    layout_seed=layout_seed,
                    episode=total_episode,
                )
                core_counts[skill] += 1
                terminal_reward_sums[skill] += result["terminal_reward"]
                terminal_counts[skill, result["terminal_stage"]] += 1
                observation = scheduler_observation(
                    matrix,
                    skill,
                    float(result["terminal_reward"]),
                    config.scheduler_signal,
                )
                ema_scores[skill] = _update_ema(
                    ema_scores[skill],
                    observation,
                    config.ema_alpha,
                )
                total_episode += 1
            selected = select_deficit_skill(ema_scores, scheduler_rng)
            result = _train_episode(
                env,
                config,
                policy_config,
                reward_model,
                q_table,
                visits,
                action_rng,
                skill=selected,
                layout_seed=layout_seed,
                episode=total_episode,
            )
            extra_counts[selected] += 1
            terminal_reward_sums[selected] += result["terminal_reward"]
            terminal_counts[selected, result["terminal_stage"]] += 1
            selected_observation = scheduler_observation(
                matrix,
                selected,
                float(result["terminal_reward"]),
                config.scheduler_signal,
            )
            ema_scores[selected] = _update_ema(
                ema_scores[selected],
                selected_observation,
                config.ema_alpha,
            )
            total_episode += 1
            ema_trace.append(
                {
                    "cycle": cycle + 1,
                    "total_policy_episodes": total_episode,
                    "ema_scheduler_scores": ema_scores.tolist(),
                    "selected_extra_skill": selected,
                    "selected_terminal_reward": result["terminal_reward"],
                    "selected_scheduler_observation": selected_observation,
                }
            )
            if total_episode % config.eval_interval == 0:
                evaluation = evaluate_q_table(
                    q_table,
                    policy_config,
                    seed=config.seed + 100_000,
                )
                evaluations.append(
                    {"policy_episodes": total_episode, **evaluation}
                )
    finally:
        env.close()

    elapsed = time.monotonic() - started
    final = evaluate_q_table(q_table, policy_config, seed=config.seed + 900_000)
    recent = evaluations[-min(config.stability_checkpoints, len(evaluations)) :]
    stability = bool(
        len(recent) >= config.stability_checkpoints
        and all(row["specialization_gate_passed"] for row in recent)
    )
    manifest = _write_rollout_audit(
        output_dir,
        q_table,
        policy_config,
        final["stage_assignment"],
        final["matched_stage_rates"],
    )
    _save_q_table(output_dir / "q_table.npz", q_table, visits)
    output = {
        "config": asdict(config),
        "source_metrics": str(source_metrics.resolve()),
        "frozen_reward_matrix": matrix,
        "elapsed_seconds": elapsed,
        "training_layout_seed_mode": "three_skill_core_plus_selected_extra_same_layout",
        "scheduler_signal": config.scheduler_signal,
        "scheduler_reads_semantic_stage": False,
        "core_episode_counts": core_counts.tolist(),
        "extra_episode_counts": extra_counts.tolist(),
        "total_episode_counts": (core_counts + extra_counts).tolist(),
        "terminal_reward_sums": terminal_reward_sums.tolist(),
        "terminal_counts_audit_only": terminal_counts.tolist(),
        "final_ema_scheduler_scores": ema_scores.tolist(),
        "ema_trace": ema_trace,
        "evaluations": evaluations,
        "final_evaluation": final,
        "checkpoint_stability_passed": stability,
        "signal_gate_passed": bool(final["specialization_gate_passed"] and stability),
        "q_state_count": len(q_table),
        "reward_model": reward_model.state_dict(),
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
    parser.add_argument("--source-metrics", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy-episodes", type=int, default=15_000)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--ema-alpha", type=float, default=0.05)
    parser.add_argument(
        "--scheduler-signal",
        choices=("raw_reward", "top_reward_indicator"),
        default="raw_reward",
    )
    parser.add_argument("--eval-interval", type=int, default=3_000)
    parser.add_argument("--eval-episodes", type=int, default=512)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = AdaptiveAllocationConfig(
        seed=args.seed,
        policy_episodes=args.policy_episodes,
        horizon=args.horizon,
        ema_alpha=args.ema_alpha,
        scheduler_signal=args.scheduler_signal,
        eval_interval=args.eval_interval,
        eval_episodes_per_skill=args.eval_episodes,
    )
    run_id = (
        f"gotoobject_adaptive_deficit_seed{config.seed}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/minigrid_gotoobject_training"
    ) / run_id
    output = train_adaptive_run(config, args.source_metrics, output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "elapsed_seconds": output["elapsed_seconds"],
                "extra_episode_counts": output["extra_episode_counts"],
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
