"""Outcome helpers for the official Gymnasium FrozenLake environment."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import DOWN, LEFT, RIGHT


FROZENLAKE_OUTCOMES = ("safe_timeout", "hole_terminal", "goal_terminal")
SCRIPTED_ACTIONS = {
    "safe_timeout": (LEFT,) * 32,
    "hole_terminal": (DOWN, RIGHT),
    "goal_terminal": (RIGHT, RIGHT, DOWN, DOWN, DOWN, RIGHT),
}


@dataclass(frozen=True)
class FrozenLakeConfig:
    map_name: str = "4x4"
    is_slippery: bool = False
    max_episode_steps: int = 32


@dataclass(frozen=True)
class FrozenLakeRollout:
    outcome: int
    final_state: int
    native_reward: float
    terminated: bool
    truncated: bool
    actions: tuple[int, ...]
    states: tuple[int, ...]
    frames: tuple[np.ndarray, ...]


def make_frozenlake(
    config: FrozenLakeConfig,
    *,
    render_mode: str | None = None,
) -> gym.Env:
    return gym.make(
        "FrozenLake-v1",
        map_name=config.map_name,
        is_slippery=config.is_slippery,
        max_episode_steps=config.max_episode_steps,
        render_mode=render_mode,
    )


def classify_outcome(
    env: gym.Env,
    state: int,
    *,
    terminated: bool,
    truncated: bool,
) -> int:
    tile = env.unwrapped.desc.ravel()[state].decode("ascii")
    if terminated and tile == "G":
        return 2
    if terminated and tile == "H":
        return 1
    if truncated:
        return 0
    raise ValueError("outcome requested before a recognized episode boundary")


def rollout_actions(
    config: FrozenLakeConfig,
    actions: tuple[int, ...],
    *,
    seed: int,
    render: bool = False,
) -> FrozenLakeRollout:
    env = make_frozenlake(config, render_mode="rgb_array" if render else None)
    try:
        state, _ = env.reset(seed=seed)
        states = [int(state)]
        frames = [env.render()] if render else []
        executed = []
        terminated = truncated = False
        native_reward = 0.0
        for action in actions:
            state, native_reward, terminated, truncated, _ = env.step(int(action))
            states.append(int(state))
            executed.append(int(action))
            if render:
                frames.append(env.render())
            if terminated or truncated:
                break
        if not (terminated or truncated):
            raise ValueError("scripted action sequence did not finish the episode")
        outcome = classify_outcome(
            env,
            int(state),
            terminated=terminated,
            truncated=truncated,
        )
        return FrozenLakeRollout(
            outcome=outcome,
            final_state=int(state),
            native_reward=float(native_reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            actions=tuple(executed),
            states=tuple(states),
            frames=tuple(frames),
        )
    finally:
        env.close()
