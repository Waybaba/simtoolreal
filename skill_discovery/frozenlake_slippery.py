"""Finite-horizon outcome control for stochastic FrozenLake dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skill_discovery.frozenlake import (
    FrozenLakeConfig,
    classify_outcome,
    make_frozenlake,
)


@dataclass(frozen=True)
class OutcomePolicy:
    target_outcome: int
    probability_upper_bound: float
    actions_by_remaining_steps: np.ndarray
    values_by_remaining_steps: np.ndarray


def _terminal_outcome(desc: np.ndarray, state: int) -> int | None:
    tile = desc.ravel()[state].decode("ascii")
    if tile == "H":
        return 1
    if tile == "G":
        return 2
    return None


def finite_horizon_outcome_policy(
    config: FrozenLakeConfig,
    target_outcome: int,
) -> OutcomePolicy:
    if not config.is_slippery:
        raise ValueError("finite-horizon stochastic audit requires slippery dynamics")
    if target_outcome not in {0, 1, 2}:
        raise ValueError("target outcome must be safe, hole, or goal")
    env = make_frozenlake(config)
    try:
        transitions = env.unwrapped.P
        desc = env.unwrapped.desc
        state_count = int(env.observation_space.n)
        action_count = int(env.action_space.n)
        horizon = config.max_episode_steps
        values = np.zeros((horizon + 1, state_count), dtype=np.float64)
        actions = np.zeros((horizon + 1, state_count), dtype=np.int64)
        if target_outcome == 0:
            for state in range(state_count):
                if _terminal_outcome(desc, state) is None:
                    values[0, state] = 1.0

        for remaining in range(1, horizon + 1):
            for state in range(state_count):
                if _terminal_outcome(desc, state) is not None:
                    continue
                action_values = np.zeros(action_count, dtype=np.float64)
                for action in range(action_count):
                    for probability, next_state, _, terminated in transitions[state][action]:
                        terminal_outcome = (
                            _terminal_outcome(desc, int(next_state))
                            if terminated
                            else None
                        )
                        if terminal_outcome is not None:
                            continuation = float(terminal_outcome == target_outcome)
                        else:
                            continuation = values[remaining - 1, int(next_state)]
                        action_values[action] += float(probability) * continuation
                actions[remaining, state] = int(np.argmax(action_values))
                values[remaining, state] = float(action_values.max())
        start_state = int(np.flatnonzero(desc.ravel() == b"S")[0])
        return OutcomePolicy(
            target_outcome=target_outcome,
            probability_upper_bound=float(values[horizon, start_state]),
            actions_by_remaining_steps=actions,
            values_by_remaining_steps=values,
        )
    finally:
        env.close()


def rollout_outcome_policy(
    config: FrozenLakeConfig,
    policy: OutcomePolicy,
    *,
    seed: int,
    render: bool = False,
) -> dict[str, object]:
    env = make_frozenlake(
        config,
        render_mode="rgb_array" if render else None,
    )
    try:
        state, _ = env.reset(seed=seed)
        states = [int(state)]
        actions = []
        frames = [env.render()] if render else []
        terminated = truncated = False
        native_reward = 0.0
        remaining = config.max_episode_steps
        while not (terminated or truncated):
            action = int(policy.actions_by_remaining_steps[remaining, int(state)])
            state, native_reward, terminated, truncated, _ = env.step(action)
            states.append(int(state))
            actions.append(action)
            remaining -= 1
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
