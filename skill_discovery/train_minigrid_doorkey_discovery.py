"""Train DoorKey skills with online spread discovery and frozen option rewards."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np

from skill_discovery.audit_gotoobject_balanced_transition import (
    maximum_weight_assignment,
    supported_predecessors,
    transitive_ancestors,
    transition_aware_matrix,
    transition_counts_from_stage_episodes,
)
from skill_discovery.minigrid_doorkey import DOORKEY_STAGES, semantic_stage
from skill_discovery.train_minigrid_doorkey_tabular import (
    DOORKEY_POLICY_ACTIONS,
    DoorKeyState,
    DoorKeyTabularConfig,
    _save_q_table,
    _training_action,
    _values,
    _write_curves,
    _write_rollout_audit,
    compact_doorkey_state,
    evaluate_q_table,
    state_changing_action_indices,
)


ReplayTransition = tuple[
    DoorKeyState,
    int,
    int,
    int,
    DoorKeyState | None,
    tuple[int, ...],
    bool,
]
ReplayBuffer = list[list[ReplayTransition]]


class TrainingStageTracker(Protocol):
    def observe(
        self,
        env: gym.Env,
        *,
        terminated: bool = False,
        reward: float = 0.0,
    ) -> int: ...


TrainingStageTrackerFactory = Callable[[], TrainingStageTracker]


@dataclass(frozen=True)
class DoorKeyDiscoveryConfig:
    env_id: str = "MiniGrid-DoorKey-5x5-v0"
    seed: int = 7
    num_skills: int = 4
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
    evaluation_checkpoints: tuple[int, ...] = (
        3_000,
        6_000,
        9_000,
        13_000,
        14_000,
        15_000,
    )
    eval_episodes_per_skill: int = 512
    stability_checkpoints: int = 3
    stage_rate_gate: float = 0.80
    minimum_transition_count: int = 25
    minimum_transition_share: float = 0.01

    def __post_init__(self) -> None:
        if self.num_skills != len(DOORKEY_STAGES):
            raise ValueError("DoorKey discovery is fixed to four skills")
        if self.bootstrap_episodes <= 0 or self.policy_episodes <= 0:
            raise ValueError("both training phases require positive budgets")
        if (
            self.bootstrap_episodes % self.num_skills != 0
            or self.policy_episodes % self.num_skills != 0
        ):
            raise ValueError("phase budgets must be divisible by four")
        if self.horizon <= 0 or self.eval_episodes_per_skill <= 0:
            raise ValueError("horizon and evaluation episodes must be positive")
        if not 0 < self.epsilon_decay_fraction <= 1:
            raise ValueError("epsilon decay fraction must be in (0, 1]")
        if not 0 < self.semantic_decay <= 1:
            raise ValueError("semantic decay must be in (0, 1]")
        if self.pseudocount <= 0:
            raise ValueError("pseudocount must be positive")
        if tuple(sorted(set(self.evaluation_checkpoints))) != (
            self.evaluation_checkpoints
        ):
            raise ValueError("evaluation checkpoints must be sorted and unique")
        if not self.evaluation_checkpoints:
            raise ValueError("at least one evaluation checkpoint is required")
        if self.evaluation_checkpoints[-1] > self.policy_episodes:
            raise ValueError("evaluation checkpoint exceeds policy budget")
        if len(self.evaluation_checkpoints) < self.stability_checkpoints:
            raise ValueError("not enough checkpoints for stability gate")
        if not 0 <= self.stage_rate_gate <= 1:
            raise ValueError("stage rate gate must be in [0, 1]")
        if self.minimum_transition_count < 0:
            raise ValueError("minimum transition count must be nonnegative")
        if not 0 <= self.minimum_transition_share <= 1:
            raise ValueError("minimum transition share must be in [0, 1]")


class DoorKeySpreadReward:
    def __init__(self, config: DoorKeyDiscoveryConfig):
        self.config = config
        self.counts = np.full(
            (config.num_skills, len(DOORKEY_STAGES)),
            config.pseudocount,
            dtype=np.float64,
        )
        self.reward_calls_by_skill = np.zeros(config.num_skills, dtype=np.int64)
        self.stage_counts = np.zeros(len(DOORKEY_STAGES), dtype=np.int64)

    def reward(self, skill: int, stage: int) -> tuple[float, dict[str, float]]:
        self.counts *= self.config.semantic_decay
        self.counts[skill, stage] += 1.0
        self.reward_calls_by_skill[skill] += 1
        self.stage_counts[stage] += 1
        posterior = self.counts[skill, stage] / self.counts[:, stage].sum()
        diayn = math.log(max(posterior * self.config.num_skills, 1.0e-8))
        totals = self.counts.sum(axis=0)
        probability = totals[stage] / totals.sum()
        coverage = -math.log(
            max(len(DOORKEY_STAGES) * probability, 1.0e-8)
        )
        total = diayn + self.config.semantic_coverage_weight * coverage
        return total, {"diayn_reward": diayn, "coverage_reward": coverage}

    def raw_matrix(self) -> tuple[tuple[float, ...], ...]:
        totals = self.counts.sum(axis=0)
        posterior = self.counts / totals[None, :]
        reward = np.log(
            np.maximum(posterior * self.config.num_skills, 1.0e-8)
        )
        probabilities = totals / totals.sum()
        coverage = -np.log(
            np.maximum(len(DOORKEY_STAGES) * probabilities, 1.0e-8)
        )
        reward += self.config.semantic_coverage_weight * coverage[None, :]
        return tuple(tuple(float(value) for value in row) for row in reward)

    def state_dict(self) -> dict[str, object]:
        return {
            "objective": "semantic_spread",
            "semantic_counts": self.counts.tolist(),
            "reward_calls_by_skill": self.reward_calls_by_skill.tolist(),
            "stage_counts": self.stage_counts.tolist(),
        }


def common_layout_seed(seed: int, phase_offset: int, episode: int) -> int:
    return seed * 1_000_000 + phase_offset + episode // len(DOORKEY_STAGES)


def _epsilon(
    config: DoorKeyDiscoveryConfig,
    episode: int,
    total_episodes: int,
) -> float:
    decay = max(int(total_episodes * config.epsilon_decay_fraction), 1)
    progress = min(episode / decay, 1.0)
    return config.epsilon_start + progress * (
        config.epsilon_end - config.epsilon_start
    )


def _step_size(visits: float) -> float:
    return float(visits**-0.6)


def _training_stage(
    env: gym.Env,
    tracker: TrainingStageTracker | None,
    *,
    terminated: bool = False,
    reward: float = 0.0,
) -> int:
    if tracker is None:
        return semantic_stage(env, terminated=terminated, reward=reward)
    stage = int(tracker.observe(env, terminated=terminated, reward=reward))
    if stage < 0 or stage >= len(DOORKEY_STAGES):
        raise ValueError("training stage tracker returned an invalid stage")
    return stage


def _save_replay_buffer(path: Path, replay_buffer: ReplayBuffer) -> None:
    transitions = [transition for episode in replay_buffer for transition in episode]
    offsets = np.zeros(len(replay_buffer) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(episode) for episode in replay_buffer])
    next_action_masks = np.zeros(
        (len(transitions), len(DOORKEY_POLICY_ACTIONS)),
        dtype=np.bool_,
    )
    for index, transition in enumerate(transitions):
        next_action_masks[index, list(transition[5])] = True
    np.savez_compressed(
        path,
        episode_offsets=offsets,
        state_keys=np.asarray([row[0] for row in transitions], dtype=np.int16),
        skills=np.asarray([row[1] for row in transitions], dtype=np.int8),
        action_indices=np.asarray([row[2] for row in transitions], dtype=np.int8),
        stages=np.asarray([row[3] for row in transitions], dtype=np.int8),
        next_state_keys=np.asarray(
            [row[4] if row[4] is not None else (0,) * 12 for row in transitions],
            dtype=np.int16,
        ),
        next_action_masks=next_action_masks,
        terminals=np.asarray([row[6] for row in transitions], dtype=np.bool_),
    )


def resolve_bootstrap(
    raw_matrix: tuple[tuple[float, ...], ...],
    replay_buffer: ReplayBuffer,
    config: DoorKeyDiscoveryConfig,
) -> dict[str, object]:
    matrix = np.asarray(raw_matrix, dtype=np.float64)
    assignment, scores = maximum_weight_assignment(matrix)
    independent_tops = tuple(int(value) for value in np.argmax(matrix, axis=1))
    stage_episodes = [
        [0, *[transition[3] for transition in episode]]
        for episode in replay_buffer
    ]
    transition_counts = transition_counts_from_stage_episodes(
        stage_episodes,
        num_stages=len(DOORKEY_STAGES),
    )
    direct_predecessors = supported_predecessors(
        transition_counts,
        minimum_count=config.minimum_transition_count,
        minimum_share=config.minimum_transition_share,
    )
    try:
        ancestors = transitive_ancestors(direct_predecessors)
        acyclic = True
    except ValueError:
        ancestors = tuple(() for _ in DOORKEY_STAGES)
        acyclic = False
    transition_gate = all(direct_predecessors[target] for target in range(1, 4))
    structural_gate = bool(
        sorted(assignment) == list(range(4)) and transition_gate and acyclic
    )
    reward_matrix = (
        transition_aware_matrix(assignment, ancestors)
        if structural_gate
        else None
    )
    return {
        "gate_passed": structural_gate,
        "independent_top_stages": independent_tops,
        "independent_assignment_collision": sorted(independent_tops)
        != list(range(4)),
        "assigned_stages": assignment,
        "permutation_scores": scores,
        "transition_counts": transition_counts.tolist(),
        "direct_predecessors": direct_predecessors,
        "transitive_ancestors": ancestors,
        "reward_matrix": reward_matrix,
    }


def _replay_frozen_transitions(
    replay_buffer: ReplayBuffer,
    policy_config: DoorKeyTabularConfig,
    q_table: dict[DoorKeyState, np.ndarray],
    visits: dict[DoorKeyState, np.ndarray],
) -> dict[str, int | float]:
    transition_count = 0
    reward_sum = 0.0
    option_truncated_episodes = 0
    for episode in replay_buffer:
        if not episode:
            continue
        skill = episode[0][1]
        target_stage = policy_config.target_assignment[skill]
        relabelled = []
        for transition in episode:
            reached_option_target = bool(
                target_stage in {1, 2} and transition[3] == target_stage
            )
            if reached_option_target:
                relabelled.append(
                    (
                        *transition[:4],
                        None,
                        (),
                        True,
                    )
                )
                option_truncated_episodes += 1
                break
            relabelled.append(transition)
        for (
            key,
            skill,
            action_index,
            stage,
            next_key,
            next_valid_indices,
            terminal,
        ) in reversed(relabelled):
            q_values = _values(q_table, key, create=True)
            visit_values = _values(visits, key, create=True)
            reward = float(policy_config.reward_matrix[skill][stage])
            if terminal:
                target = reward
            else:
                assert next_key is not None and next_valid_indices
                next_values = _values(q_table, next_key, create=True)
                target = reward + policy_config.gamma * next_values[
                    skill, list(next_valid_indices)
                ].max()
            visit_values[skill, action_index] += 1
            q_values[skill, action_index] += _step_size(
                float(visit_values[skill, action_index])
            ) * (target - q_values[skill, action_index])
            transition_count += 1
            reward_sum += reward
    return {
        "episodes": len(replay_buffer),
        "transitions": transition_count,
        "option_truncated_episodes": option_truncated_episodes,
        "relabelled_reward_sum": reward_sum,
        "q_state_count": len(q_table),
    }


def _train_bootstrap(
    env: gym.Env,
    config: DoorKeyDiscoveryConfig,
    reward_model: DoorKeySpreadReward,
    q_table: dict[DoorKeyState, np.ndarray],
    visits: dict[DoorKeyState, np.ndarray],
    rng: np.random.Generator,
    replay_buffer: ReplayBuffer,
    stage_tracker_factory: TrainingStageTrackerFactory | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    terminal_counts = np.zeros((4, 4), dtype=np.int64)
    reward_sums = {"diayn_reward": 0.0, "coverage_reward": 0.0}
    for episode in range(config.bootstrap_episodes):
        skill = episode % config.num_skills
        env.reset(seed=common_layout_seed(config.seed, 0, episode))
        stage_tracker = (
            stage_tracker_factory() if stage_tracker_factory is not None else None
        )
        furthest_stage = _training_stage(env, stage_tracker)
        epsilon = _epsilon(config, episode, config.bootstrap_episodes)
        replay_episode = []
        for step in range(config.horizon):
            key = compact_doorkey_state(env)
            q_values = _values(q_table, key, create=True)
            visit_values = _values(visits, key, create=True)
            valid_indices = state_changing_action_indices(env)
            if rng.random() < epsilon:
                action_index = int(rng.choice(valid_indices))
            else:
                action_index = _training_action(
                    q_values[skill],
                    rng,
                    valid_indices,
                )
            _, native_reward, terminated, truncated, _ = env.step(
                DOORKEY_POLICY_ACTIONS[action_index]
            )
            stage = _training_stage(
                env,
                stage_tracker,
                terminated=bool(terminated),
                reward=float(native_reward),
            )
            furthest_stage = max(furthest_stage, stage)
            reward, parts = reward_model.reward(skill, furthest_stage)
            for name in reward_sums:
                reward_sums[name] += parts[name]
            terminal = bool(terminated or truncated or step + 1 == config.horizon)
            next_key = None
            next_valid_indices: tuple[int, ...] = ()
            if terminal:
                target = reward
                terminal_counts[skill, furthest_stage] += 1
            else:
                next_key = compact_doorkey_state(env)
                next_valid_indices = state_changing_action_indices(env)
                target = reward + config.gamma * _values(
                    q_table,
                    next_key,
                    create=True,
                )[skill, list(next_valid_indices)].max()
            replay_episode.append(
                (
                    key,
                    skill,
                    action_index,
                    furthest_stage,
                    next_key,
                    next_valid_indices,
                    terminal,
                )
            )
            visit_values[skill, action_index] += 1
            q_values[skill, action_index] += _step_size(
                float(visit_values[skill, action_index])
            ) * (target - q_values[skill, action_index])
            if terminal:
                break
        replay_buffer.append(replay_episode)
    return terminal_counts, reward_sums


def _train_policy(
    env: gym.Env,
    config: DoorKeyDiscoveryConfig,
    policy_config: DoorKeyTabularConfig,
    q_table: dict[DoorKeyState, np.ndarray],
    visits: dict[DoorKeyState, np.ndarray],
    rng: np.random.Generator,
    stage_tracker_factory: TrainingStageTrackerFactory | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    terminal_counts = np.zeros((4, 4), dtype=np.int64)
    evaluations = []
    for episode in range(config.policy_episodes):
        skill = episode % config.num_skills
        target_stage = policy_config.target_assignment[skill]
        env.reset(seed=common_layout_seed(config.seed, 500_000, episode))
        stage_tracker = (
            stage_tracker_factory() if stage_tracker_factory is not None else None
        )
        furthest_stage = _training_stage(env, stage_tracker)
        epsilon = _epsilon(config, episode, config.policy_episodes)
        for step in range(config.horizon):
            key = compact_doorkey_state(env)
            q_values = _values(q_table, key, create=True)
            visit_values = _values(visits, key, create=True)
            valid_indices = state_changing_action_indices(env)
            if rng.random() < epsilon:
                action_index = int(rng.choice(valid_indices))
            else:
                action_index = _training_action(
                    q_values[skill],
                    rng,
                    valid_indices,
                )
            _, native_reward, terminated, truncated, _ = env.step(
                DOORKEY_POLICY_ACTIONS[action_index]
            )
            stage = _training_stage(
                env,
                stage_tracker,
                terminated=bool(terminated),
                reward=float(native_reward),
            )
            furthest_stage = max(furthest_stage, stage)
            reward = policy_config.reward_matrix[skill][furthest_stage]
            option_terminated = bool(
                target_stage in {1, 2} and furthest_stage == target_stage
            )
            terminal = bool(
                terminated
                or truncated
                or option_terminated
                or step + 1 == config.horizon
            )
            if terminal:
                target = reward
                terminal_counts[skill, furthest_stage] += 1
            else:
                next_key = compact_doorkey_state(env)
                next_valid_indices = state_changing_action_indices(env)
                target = reward + config.gamma * _values(
                    q_table,
                    next_key,
                    create=True,
                )[skill, list(next_valid_indices)].max()
            visit_values[skill, action_index] += 1
            q_values[skill, action_index] += _step_size(
                float(visit_values[skill, action_index])
            ) * (target - q_values[skill, action_index])
            if terminal:
                break
        if episode + 1 in config.evaluation_checkpoints:
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
    return terminal_counts, evaluations


def _policy_config(
    config: DoorKeyDiscoveryConfig,
    assignment: tuple[int, ...],
    reward_matrix: tuple[tuple[float, ...], ...],
) -> DoorKeyTabularConfig:
    return DoorKeyTabularConfig(
        env_id=config.env_id,
        seed=config.seed,
        episodes=config.policy_episodes,
        horizon=config.horizon,
        gamma=config.gamma,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay_fraction=config.epsilon_decay_fraction,
        evaluation_checkpoints=config.evaluation_checkpoints,
        eval_episodes_per_skill=config.eval_episodes_per_skill,
        stability_checkpoints=config.stability_checkpoints,
        stage_rate_gate=config.stage_rate_gate,
        valid_action_mask=True,
        terminate_on_target=True,
        target_assignment=assignment,
        reward_matrix=reward_matrix,
    )


def train_discovery_run(
    config: DoorKeyDiscoveryConfig,
    output_dir: Path,
    *,
    stage_tracker_factory: TrainingStageTrackerFactory | None = None,
    render_mode: str | None = None,
    training_stage_metric_summary: Callable[[], dict[str, object]] | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(config.seed)
    env = gym.make(config.env_id, render_mode=render_mode)
    bootstrap_reward = DoorKeySpreadReward(config)
    bootstrap_q: dict[DoorKeyState, np.ndarray] = {}
    bootstrap_visits: dict[DoorKeyState, np.ndarray] = {}
    replay_buffer: ReplayBuffer = []
    started = time.monotonic()
    try:
        bootstrap_counts, bootstrap_reward_sums = _train_bootstrap(
            env,
            config,
            bootstrap_reward,
            bootstrap_q,
            bootstrap_visits,
            rng,
            replay_buffer,
            stage_tracker_factory,
        )
        raw_matrix = bootstrap_reward.raw_matrix()
        resolution = resolve_bootstrap(raw_matrix, replay_buffer, config)
        _save_q_table(
            output_dir / "bootstrap_q_table.npz",
            bootstrap_q,
            bootstrap_visits,
        )
        _save_replay_buffer(
            output_dir / "bootstrap_replay_buffer.npz",
            replay_buffer,
        )
        base_output = {
            "config": asdict(config),
            "bootstrap_gate_passed": resolution["gate_passed"],
            "bootstrap_raw_matrix": raw_matrix,
            "bootstrap_independent_top_stages": resolution[
                "independent_top_stages"
            ],
            "bootstrap_independent_assignment_collision": resolution[
                "independent_assignment_collision"
            ],
            "bootstrap_assigned_stages": resolution["assigned_stages"],
            "bootstrap_assigned_stage_names": [
                DOORKEY_STAGES[target]
                for target in resolution["assigned_stages"]
            ],
            "bootstrap_permutation_scores": resolution["permutation_scores"],
            "bootstrap_transition_counts": resolution["transition_counts"],
            "bootstrap_direct_predecessors": resolution[
                "direct_predecessors"
            ],
            "bootstrap_transitive_ancestors": resolution[
                "transitive_ancestors"
            ],
            "bootstrap_reward_matrix": resolution["reward_matrix"],
            "bootstrap_reward_model": bootstrap_reward.state_dict(),
            "bootstrap_terminal_counts": bootstrap_counts.tolist(),
            "bootstrap_reward_sums": bootstrap_reward_sums,
            "bootstrap_q_state_count": len(bootstrap_q),
            "bootstrap_replay_episodes": len(replay_buffer),
            "training_stage_metric": training_stage_metric_summary()
            if training_stage_metric_summary is not None
            else {"name": "oracle_semantic_stage"},
        }
        if not resolution["gate_passed"]:
            output = {
                **base_output,
                "elapsed_seconds": time.monotonic() - started,
                "policy_phase_ran": False,
                "checkpoint_stability_passed": False,
                "signal_gate_passed": False,
            }
            (output_dir / "metrics.json").write_text(
                json.dumps(output, indent=2),
                encoding="utf-8",
            )
            return output

        assignment = tuple(int(value) for value in resolution["assigned_stages"])
        reward_matrix = resolution["reward_matrix"]
        assert reward_matrix is not None
        policy_config = _policy_config(config, assignment, reward_matrix)
        policy_q: dict[DoorKeyState, np.ndarray] = {}
        policy_visits: dict[DoorKeyState, np.ndarray] = {}
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
        policy_counts, evaluations = _train_policy(
            env,
            config,
            policy_config,
            policy_q,
            policy_visits,
            rng,
            stage_tracker_factory,
        )
    finally:
        env.close()

    final = evaluate_q_table(
        policy_q,
        policy_config,
        seed=config.seed + 900_000,
    )
    recent = evaluations[-config.stability_checkpoints :]
    stability = bool(
        len(recent) == config.stability_checkpoints
        and all(row["specialization_gate_passed"] for row in recent)
    )
    _save_q_table(output_dir / "q_table.npz", policy_q, policy_visits)
    manifest = _write_rollout_audit(
        output_dir,
        policy_q,
        policy_config,
        final["target_stage_rates"],
    )
    _write_curves(output_dir / "stage_curves.svg", evaluations)
    output = {
        **base_output,
        "elapsed_seconds": time.monotonic() - started,
        "policy_phase_ran": True,
        "policy_q_reset_after_bootstrap": True,
        "bootstrap_replay_summary": replay_summary,
        "policy_terminal_counts": policy_counts.tolist(),
        "policy_q_state_count": len(policy_q),
        "evaluations": evaluations,
        "final_evaluation": final,
        "checkpoint_stability_passed": stability,
        "signal_gate_passed": bool(final["specialization_gate_passed"] and stability),
        "policy_rollout_manifest": manifest,
        "training_stage_metric": training_stage_metric_summary()
        if training_stage_metric_summary is not None
        else {"name": "oracle_semantic_stage"},
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bootstrap-episodes", type=int, default=5_000)
    parser.add_argument("--policy-episodes", type=int, default=15_000)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--evaluation-checkpoints")
    parser.add_argument("--eval-episodes", type=int, default=512)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    checkpoints = (
        tuple(int(value) for value in args.evaluation_checkpoints.split(","))
        if args.evaluation_checkpoints
        else DoorKeyDiscoveryConfig.evaluation_checkpoints
    )
    config = DoorKeyDiscoveryConfig(
        seed=args.seed,
        bootstrap_episodes=args.bootstrap_episodes,
        policy_episodes=args.policy_episodes,
        horizon=args.horizon,
        evaluation_checkpoints=checkpoints,
        eval_episodes_per_skill=args.eval_episodes,
    )
    run_id = (
        f"doorkey5_online_spread_balanced_transition_replay_seed{config.seed}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/minigrid_doorkey_training"
    ) / run_id
    output = train_discovery_run(config, output_dir)
    final = output.get("final_evaluation")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "elapsed_seconds": output["elapsed_seconds"],
                "bootstrap_gate_passed": output["bootstrap_gate_passed"],
                "bootstrap_assigned_stages": output[
                    "bootstrap_assigned_stages"
                ],
                "policy_phase_ran": output["policy_phase_ran"],
                "final_target_stage_rates": final["target_stage_rates"]
                if final
                else None,
                "final_state_rates": final["final_state_rates"]
                if final
                else None,
                "goal_native_success_rate": final["goal_native_success_rate"]
                if final
                else None,
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
