"""Semantic auditing helpers for the official MiniGrid DoorKey environments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from minigrid.core.actions import Actions


DOORKEY_STAGES = ("navigation_only", "key_acquired", "door_opened", "goal_reached")
DIR_TO_VEC = ((1, 0), (0, 1), (-1, 0), (0, -1))


@dataclass(frozen=True)
class SolverResult:
    actions: tuple[int, ...]
    stage_sequence: tuple[int, ...]
    reward: float
    terminated: bool
    truncated: bool
    frames: tuple[np.ndarray, ...]


def object_positions(env: gym.Env) -> dict[str, tuple[int, int]]:
    base = env.unwrapped
    positions: dict[str, tuple[int, int]] = {}
    for x in range(base.width):
        for y in range(base.height):
            obj = base.grid.get(x, y)
            if obj is not None and obj.type in {"key", "door", "goal"}:
                positions[obj.type] = (x, y)
    if base.carrying is not None and base.carrying.type == "key":
        positions["key"] = tuple(int(value) for value in base.carrying.cur_pos)
    return positions


def semantic_stage(env: gym.Env, *, terminated: bool = False, reward: float = 0.0) -> int:
    if terminated and reward > 0:
        return 3
    base = env.unwrapped
    for x in range(base.width):
        for y in range(base.height):
            obj = base.grid.get(x, y)
            if obj is not None and obj.type == "door" and bool(obj.is_open):
                return 2
    if base.carrying is not None and base.carrying.type == "key":
        return 1
    return 0


def reproducibility_signature(env: gym.Env, seed: int) -> tuple[bytes, tuple[int, int], int]:
    env.reset(seed=seed)
    base = env.unwrapped
    return (
        base.grid.encode().tobytes(),
        tuple(int(value) for value in base.agent_pos),
        int(base.agent_dir),
    )


def _front_position(state: tuple[int, int, int]) -> tuple[int, int]:
    x, y, direction = state
    dx, dy = DIR_TO_VEC[direction]
    return x + dx, y + dy


def _can_enter(env: gym.Env, position: tuple[int, int]) -> bool:
    base = env.unwrapped
    x, y = position
    if x < 0 or y < 0 or x >= base.width or y >= base.height:
        return False
    obj = base.grid.get(x, y)
    return obj is None or bool(obj.can_overlap())


def plan_to_face(env: gym.Env, target: tuple[int, int]) -> list[int]:
    base = env.unwrapped
    start = (
        int(base.agent_pos[0]),
        int(base.agent_pos[1]),
        int(base.agent_dir),
    )
    queue = deque([start])
    parents: dict[tuple[int, int, int], tuple[tuple[int, int, int], int] | None] = {
        start: None
    }
    goal_state: tuple[int, int, int] | None = None
    while queue:
        state = queue.popleft()
        if _front_position(state) == target:
            goal_state = state
            break
        x, y, direction = state
        candidates = (
            ((x, y, (direction - 1) % 4), int(Actions.left)),
            ((x, y, (direction + 1) % 4), int(Actions.right)),
        )
        forward_position = _front_position(state)
        if _can_enter(env, forward_position):
            candidates += (
                ((forward_position[0], forward_position[1], direction), int(Actions.forward)),
            )
        for next_state, action in candidates:
            if next_state not in parents:
                parents[next_state] = (state, action)
                queue.append(next_state)
    if goal_state is None:
        raise RuntimeError(f"no path found to face target {target}")

    actions = []
    state = goal_state
    while parents[state] is not None:
        previous, action = parents[state]
        actions.append(action)
        state = previous
    actions.reverse()
    return actions


def solve_doorkey_episode(env: gym.Env, seed: int) -> SolverResult:
    env.reset(seed=seed)
    frames = [env.render()]
    actions: list[int] = []
    stages = [semantic_stage(env)]
    terminated = False
    truncated = False
    reward = 0.0

    def run(next_actions: list[int]) -> None:
        nonlocal terminated, truncated, reward
        for action in next_actions:
            _, step_reward, terminated, truncated, _ = env.step(action)
            reward = float(step_reward)
            actions.append(int(action))
            stages.append(
                semantic_stage(env, terminated=terminated, reward=reward)
            )
            if terminated or truncated:
                break

    positions = object_positions(env)
    run(plan_to_face(env, positions["key"]))
    run([int(Actions.pickup)])
    frames.append(env.render())
    if semantic_stage(env) != 1:
        raise RuntimeError("scripted solver failed to acquire the key")

    positions = object_positions(env)
    run(plan_to_face(env, positions["door"]))
    run([int(Actions.toggle)])
    frames.append(env.render())
    if semantic_stage(env) != 2:
        raise RuntimeError("scripted solver failed to open the door")

    positions = object_positions(env)
    run(plan_to_face(env, positions["goal"]))
    run([int(Actions.forward)])
    frames.append(env.render())
    if not terminated or reward <= 0 or max(stages) != 3:
        raise RuntimeError("scripted solver failed to reach the goal")
    return SolverResult(
        actions=tuple(actions),
        stage_sequence=tuple(stages),
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        frames=tuple(frames),
    )
