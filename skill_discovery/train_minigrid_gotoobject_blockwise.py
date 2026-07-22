"""Train GoToObject skills with a two-timescale spread reward."""

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
    frozen_reward_matrix_from_metrics,
)


ReplayTransition = tuple[RelationKey, int, int, int, RelationKey | None, bool]
ReplayBuffer = list[list[ReplayTransition]]


@dataclass(frozen=True)
class BlockwiseSpreadConfig:
    seed: int = 7
    num_skills: int = 3
    bootstrap_episodes: int = 5_000
    policy_episodes: int = 15_000
    horizon: int = 64
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.0
    epsilon_decay_fraction: float = 0.8
    pseudocount: float = 2.0
    semantic_decay: float = 0.9995
    semantic_coverage_weight: float = 1.0
    bootstrap_replay: str = "none"
    eval_interval: int = 3_000
    eval_episodes_per_skill: int = 512
    stability_checkpoints: int = 3
    stage_rate_gates: tuple[float, float, float] = (0.90, 0.90, 0.90)

    def __post_init__(self) -> None:
        if self.num_skills != len(GOTOOBJECT_STAGES):
            raise ValueError("GoToObject blockwise probe is fixed to three skills")
        if self.bootstrap_episodes <= 0 or self.policy_episodes <= 0:
            raise ValueError("both training phases require positive episode budgets")
        if self.horizon <= 0 or self.eval_interval <= 0:
            raise ValueError("horizon and eval interval must be positive")
        if self.policy_episodes % self.eval_interval != 0:
            raise ValueError("policy episodes must be divisible by eval interval")
        if not 0 < self.epsilon_decay_fraction <= 1:
            raise ValueError("epsilon decay fraction must be in (0, 1]")
        if self.bootstrap_replay not in {"none", "reverse_once"}:
            raise ValueError("unknown bootstrap replay mode")
        if self.stability_checkpoints <= 0:
            raise ValueError("stability checkpoints must be positive")
        if len(self.stage_rate_gates) != len(GOTOOBJECT_STAGES):
            raise ValueError("one rate gate is required per stage")


def common_layout_seed(seed: int, phase_offset: int, episode: int) -> int:
    return seed * 1_000_000 + phase_offset + episode // len(GOTOOBJECT_STAGES)


def _phase_epsilon(config: BlockwiseSpreadConfig, episode: int, total: int) -> float:
    decay = max(int(total * config.epsilon_decay_fraction), 1)
    progress = min(episode / decay, 1.0)
    return config.epsilon_start + progress * (
        config.epsilon_end - config.epsilon_start
    )


def _online_config(config: BlockwiseSpreadConfig) -> GoToObjectTrainConfig:
    return GoToObjectTrainConfig(
        objective="semantic_spread",
        reward_timing="occupancy",
        seed=config.seed,
        episodes=config.bootstrap_episodes,
        horizon=config.horizon,
        gamma=config.gamma,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay_fraction=config.epsilon_decay_fraction,
        pseudocount=config.pseudocount,
        semantic_decay=config.semantic_decay,
        semantic_coverage_weight=config.semantic_coverage_weight,
        eval_interval=config.eval_interval,
        eval_episodes_per_skill=config.eval_episodes_per_skill,
        stability_checkpoints=config.stability_checkpoints,
        stage_rate_gates=config.stage_rate_gates,
    )


def snapshot_spread_matrix(
    reward_model: GoToObjectReward,
    config: BlockwiseSpreadConfig,
    *,
    calibration: str,
) -> tuple[tuple[float, ...], ...]:
    metrics = {
        "config": {"semantic_coverage_weight": config.semantic_coverage_weight},
        "reward_model": reward_model.state_dict(),
    }
    return frozen_reward_matrix_from_metrics(metrics, calibration=calibration)


def _policy_config(
    config: BlockwiseSpreadConfig,
    matrix: tuple[tuple[float, ...], ...],
) -> GoToObjectTrainConfig:
    return GoToObjectTrainConfig(
        objective="frozen_matrix",
        frozen_reward_matrix=matrix,
        frozen_reward_source="phase5q_bootstrap_snapshot",
        frozen_reward_calibration="runner_up_unit",
        reward_timing="occupancy",
        seed=config.seed,
        episodes=config.policy_episodes,
        horizon=config.horizon,
        gamma=config.gamma,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay_fraction=config.epsilon_decay_fraction,
        pseudocount=config.pseudocount,
        semantic_decay=config.semantic_decay,
        semantic_coverage_weight=config.semantic_coverage_weight,
        eval_interval=config.eval_interval,
        eval_episodes_per_skill=config.eval_episodes_per_skill,
        stability_checkpoints=config.stability_checkpoints,
        stage_rate_gates=config.stage_rate_gates,
    )


def _save_q_table(
    path: Path,
    q_table: dict[RelationKey, np.ndarray],
    visits: dict[RelationKey, np.ndarray],
) -> None:
    keys = sorted(q_table)
    np.savez_compressed(
        path,
        relation_keys=np.asarray(keys, dtype=np.int16),
        q_values=np.stack([q_table[key] for key in keys]),
        visits=np.stack([visits[key] for key in keys]),
    )


def _save_replay_buffer(path: Path, replay_buffer: ReplayBuffer) -> None:
    transitions = [transition for episode in replay_buffer for transition in episode]
    offsets = np.zeros(len(replay_buffer) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(episode) for episode in replay_buffer])
    np.savez_compressed(
        path,
        episode_offsets=offsets,
        relation_keys=np.asarray([row[0] for row in transitions], dtype=np.int16),
        skills=np.asarray([row[1] for row in transitions], dtype=np.int8),
        action_indices=np.asarray([row[2] for row in transitions], dtype=np.int8),
        stages=np.asarray([row[3] for row in transitions], dtype=np.int8),
        next_relation_keys=np.asarray(
            [row[4] if row[4] is not None else (0,) * 8 for row in transitions],
            dtype=np.int16,
        ),
        terminals=np.asarray([row[5] for row in transitions], dtype=np.bool_),
    )


def _replay_frozen_transitions(
    replay_buffer: ReplayBuffer,
    policy_config: GoToObjectTrainConfig,
    q_table: dict[RelationKey, np.ndarray],
    visits: dict[RelationKey, np.ndarray],
) -> dict[str, float | int]:
    matrix = policy_config.frozen_reward_matrix
    assert matrix is not None
    transition_count = 0
    reward_sum = 0.0
    for episode in replay_buffer:
        for key, skill, action_index, stage, next_key, terminal in reversed(episode):
            q_values = _values(q_table, key, create=True)
            visit_values = _values(visits, key, create=True)
            reward = float(matrix[skill][stage])
            if terminal:
                target = reward
            else:
                assert next_key is not None
                target = reward + policy_config.gamma * _values(
                    q_table,
                    next_key,
                    create=True,
                )[skill].max()
            visit_values[skill, action_index] += 1
            step_size = _learning_rate(
                policy_config,
                float(visit_values[skill, action_index]),
            )
            q_values[skill, action_index] += step_size * (
                target - q_values[skill, action_index]
            )
            transition_count += 1
            reward_sum += reward
    return {
        "episodes": len(replay_buffer),
        "transitions": transition_count,
        "relabelled_reward_sum": reward_sum,
        "q_state_count": len(q_table),
    }


def _train_phase(
    env: object,
    config: BlockwiseSpreadConfig,
    policy_config: GoToObjectTrainConfig,
    reward_model: GoToObjectReward,
    q_table: dict[RelationKey, np.ndarray],
    visits: dict[RelationKey, np.ndarray],
    rng: np.random.Generator,
    *,
    episodes: int,
    phase_offset: int,
    shadow_model: GoToObjectReward | None = None,
    replay_buffer: ReplayBuffer | None = None,
    evaluate: bool = False,
) -> tuple[np.ndarray, dict[str, float], list[dict[str, object]]]:
    terminal_counts = np.zeros((3, 3), dtype=np.int64)
    reward_sums = {"diayn_reward": 0.0, "coverage_reward": 0.0}
    evaluations: list[dict[str, object]] = []
    for episode in range(episodes):
        replay_episode: list[ReplayTransition] = []
        skill = episode % config.num_skills
        env.reset(seed=common_layout_seed(config.seed, phase_offset, episode))
        epsilon = _phase_epsilon(config, episode, episodes)
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
            reward, parts = reward_model.reward(skill, stage)
            if shadow_model is not None:
                shadow_model.reward(skill, stage)
            for name in reward_sums:
                reward_sums[name] += parts[name]
            next_key = None
            if terminal:
                target = reward
                terminal_counts[skill, stage] += 1
            else:
                next_key = compact_relation_key(env)
                target = reward + config.gamma * _values(
                    q_table, next_key, create=True
                )[skill].max()
            if replay_buffer is not None:
                replay_episode.append(
                    (key, skill, action_index, stage, next_key, terminal)
                )
            visit_values[skill, action_index] += 1
            step_size = _learning_rate(
                policy_config,
                float(visit_values[skill, action_index]),
            )
            q_values[skill, action_index] += step_size * (
                target - q_values[skill, action_index]
            )
            if terminal:
                break
        if replay_buffer is not None:
            replay_buffer.append(replay_episode)
        if evaluate and (episode + 1) % config.eval_interval == 0:
            evaluation = evaluate_q_table(
                q_table,
                policy_config,
                seed=config.seed + 100_000,
            )
            evaluations.append(
                {
                    "policy_episodes": episode + 1,
                    "total_episodes": config.bootstrap_episodes + episode + 1,
                    **evaluation,
                }
            )
    return terminal_counts, reward_sums, evaluations


def _copy_reward_model(source: GoToObjectReward) -> GoToObjectReward:
    target = GoToObjectReward(source.config)
    target.counts = source.counts.copy()
    target.reward_calls_by_skill = source.reward_calls_by_skill.copy()
    target.stage_counts = source.stage_counts.copy()
    return target


def train_blockwise_run(
    config: BlockwiseSpreadConfig,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(config.seed)
    env = make_gotoobject()
    online_config = _online_config(config)
    bootstrap_reward = GoToObjectReward(online_config)
    bootstrap_q: dict[RelationKey, np.ndarray] = {}
    bootstrap_visits: dict[RelationKey, np.ndarray] = {}
    replay_buffer: ReplayBuffer | None = (
        [] if config.bootstrap_replay == "reverse_once" else None
    )
    started = time.monotonic()
    try:
        bootstrap_counts, bootstrap_reward_sums, _ = _train_phase(
            env,
            config,
            online_config,
            bootstrap_reward,
            bootstrap_q,
            bootstrap_visits,
            rng,
            episodes=config.bootstrap_episodes,
            phase_offset=0,
            replay_buffer=replay_buffer,
        )
        raw_matrix = snapshot_spread_matrix(
            bootstrap_reward,
            config,
            calibration="none",
        )
        top_stages = np.argmax(np.asarray(raw_matrix), axis=1).astype(int).tolist()
        bootstrap_gate = sorted(top_stages) == list(range(config.num_skills))
        calibrated_matrix = (
            snapshot_spread_matrix(
                bootstrap_reward,
                config,
                calibration="runner_up_unit",
            )
            if bootstrap_gate
            else None
        )
        _save_q_table(
            output_dir / "bootstrap_q_table.npz",
            bootstrap_q,
            bootstrap_visits,
        )
        if replay_buffer is not None:
            _save_replay_buffer(
                output_dir / "bootstrap_replay_buffer.npz",
                replay_buffer,
            )
        if not bootstrap_gate:
            elapsed = time.monotonic() - started
            output = {
                "config": asdict(config),
                "elapsed_seconds": elapsed,
                "bootstrap_gate_passed": False,
                "bootstrap_top_stages": top_stages,
                "bootstrap_raw_matrix": raw_matrix,
                "bootstrap_calibrated_matrix": None,
                "bootstrap_reward_model": bootstrap_reward.state_dict(),
                "bootstrap_terminal_counts": bootstrap_counts.tolist(),
                "bootstrap_reward_sums": bootstrap_reward_sums,
                "bootstrap_q_state_count": len(bootstrap_q),
                "bootstrap_replay_mode": config.bootstrap_replay,
                "bootstrap_replay_summary": None,
                "policy_phase_ran": False,
                "checkpoint_stability_passed": False,
                "signal_gate_passed": False,
            }
            (output_dir / "metrics.json").write_text(
                json.dumps(output, indent=2),
                encoding="utf-8",
            )
            return output

        assert calibrated_matrix is not None
        policy_config = _policy_config(config, calibrated_matrix)
        active_reward = GoToObjectReward(policy_config)
        shadow_reward = _copy_reward_model(bootstrap_reward)
        policy_q: dict[RelationKey, np.ndarray] = {}
        policy_visits: dict[RelationKey, np.ndarray] = {}
        replay_summary = None
        if replay_buffer is not None:
            replay_summary = _replay_frozen_transitions(
                replay_buffer,
                policy_config,
                policy_q,
                policy_visits,
            )
            _save_q_table(
                output_dir / "relabelled_bootstrap_q_table.npz",
                policy_q,
                policy_visits,
            )
        policy_counts, policy_reward_sums, evaluations = _train_phase(
            env,
            config,
            policy_config,
            active_reward,
            policy_q,
            policy_visits,
            rng,
            episodes=config.policy_episodes,
            phase_offset=500_000,
            shadow_model=shadow_reward,
            evaluate=True,
        )
    finally:
        env.close()

    elapsed = time.monotonic() - started
    final = evaluate_q_table(policy_q, policy_config, seed=config.seed + 900_000)
    recent = evaluations[-min(config.stability_checkpoints, len(evaluations)) :]
    stability = bool(
        len(recent) >= config.stability_checkpoints
        and all(row["specialization_gate_passed"] for row in recent)
    )
    manifest = _write_rollout_audit(
        output_dir,
        policy_q,
        policy_config,
        final["stage_assignment"],
        final["matched_stage_rates"],
    )
    _save_q_table(output_dir / "q_table.npz", policy_q, policy_visits)
    shadow_raw = snapshot_spread_matrix(shadow_reward, config, calibration="none")
    shadow_calibrated = snapshot_spread_matrix(
        shadow_reward,
        config,
        calibration="runner_up_unit",
    )
    output = {
        "config": asdict(config),
        "elapsed_seconds": elapsed,
        "training_layout_seed_mode": "common_per_three_skill_cycle",
        "bootstrap_gate_passed": True,
        "bootstrap_top_stages": top_stages,
        "bootstrap_raw_matrix": raw_matrix,
        "bootstrap_calibrated_matrix": calibrated_matrix,
        "bootstrap_reward_model": bootstrap_reward.state_dict(),
        "bootstrap_terminal_counts": bootstrap_counts.tolist(),
        "bootstrap_reward_sums": bootstrap_reward_sums,
        "bootstrap_q_state_count": len(bootstrap_q),
        "bootstrap_replay_mode": config.bootstrap_replay,
        "bootstrap_replay_summary": replay_summary,
        "policy_phase_ran": True,
        "policy_q_reset_after_bootstrap": True,
        "policy_terminal_counts": policy_counts.tolist(),
        "policy_reward_sums": policy_reward_sums,
        "policy_q_state_count": len(policy_q),
        "shadow_reward_model": shadow_reward.state_dict(),
        "shadow_final_raw_matrix": shadow_raw,
        "shadow_final_calibrated_matrix": shadow_calibrated,
        "shadow_final_top_stages": np.argmax(
            np.asarray(shadow_raw), axis=1
        ).astype(int).tolist(),
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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bootstrap-episodes", type=int, default=5_000)
    parser.add_argument("--policy-episodes", type=int, default=15_000)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=3_000)
    parser.add_argument("--eval-episodes", type=int, default=512)
    parser.add_argument(
        "--bootstrap-replay",
        choices=("none", "reverse_once"),
        default="none",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = BlockwiseSpreadConfig(
        seed=args.seed,
        bootstrap_episodes=args.bootstrap_episodes,
        policy_episodes=args.policy_episodes,
        horizon=args.horizon,
        bootstrap_replay=args.bootstrap_replay,
        eval_interval=args.eval_interval,
        eval_episodes_per_skill=args.eval_episodes,
    )
    replay_tag = "_replay" if config.bootstrap_replay != "none" else ""
    run_id = (
        f"gotoobject_blockwise_spread{replay_tag}_seed{config.seed}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/minigrid_gotoobject_training"
    ) / run_id
    output = train_blockwise_run(config, output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "elapsed_seconds": output["elapsed_seconds"],
                "bootstrap_gate_passed": output["bootstrap_gate_passed"],
                "bootstrap_top_stages": output["bootstrap_top_stages"],
                "policy_phase_ran": output["policy_phase_ran"],
                "final_evaluation": output.get("final_evaluation"),
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
