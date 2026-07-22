"""Mission-independent object-relation helpers for MiniGrid GoToObject."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from minigrid.core.actions import Actions
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

from skill_discovery.minigrid_doorkey import DIR_TO_VEC, plan_to_face


GOTOOBJECT_STAGES = ("object_far", "object_adjacent", "object_carried")
OBJECT_TYPES = {"key", "ball", "box"}


@dataclass(frozen=True)
class ObjectRecord:
    object_type: str
    color: str
    position: tuple[int, int]


@dataclass(frozen=True)
class GoToObjectScriptResult:
    seed: int
    mission: str
    selected_object: ObjectRecord
    selected_is_mission_target: bool
    stages: tuple[int, ...]
    actions: tuple[int, ...]
    floor_counts: tuple[int, ...]
    carrying: tuple[str, str] | None
    frames: tuple[np.ndarray, ...]


def make_gotoobject(
    *,
    render_mode: str | None = None,
    mission_free: bool = False,
) -> gym.Env:
    env = gym.make(
        "MiniGrid-GoToObject-6x6-N2-v0",
        render_mode=render_mode,
    )
    if mission_free:
        env = ImgObsWrapper(FullyObsWrapper(env))
    return env


def floor_objects(env: gym.Env) -> list[ObjectRecord]:
    base = env.unwrapped
    output = []
    for x in range(base.width):
        for y in range(base.height):
            obj = base.grid.get(x, y)
            if obj is not None and obj.type in OBJECT_TYPES:
                output.append(ObjectRecord(obj.type, obj.color, (x, y)))
    return output


def semantic_stage(env: gym.Env) -> int:
    base = env.unwrapped
    if base.carrying is not None:
        return 2
    agent = tuple(int(value) for value in base.agent_pos)
    if any(
        abs(agent[0] - obj.position[0]) + abs(agent[1] - obj.position[1]) == 1
        for obj in floor_objects(env)
    ):
        return 1
    return 0


def reproducibility_signature(
    env: gym.Env,
    seed: int,
) -> tuple[bytes, tuple[int, int], int, str]:
    observation, _ = env.reset(seed=seed)
    base = env.unwrapped
    mission = observation["mission"] if isinstance(observation, dict) else base.mission
    return (
        base.grid.encode().tobytes(),
        tuple(int(value) for value in base.agent_pos),
        int(base.agent_dir),
        str(mission),
    )


def _can_enter(env: gym.Env, x: int, y: int) -> bool:
    base = env.unwrapped
    if x < 0 or y < 0 or x >= base.width or y >= base.height:
        return False
    obj = base.grid.get(x, y)
    return obj is None or bool(obj.can_overlap())


def plan_to_far(env: gym.Env) -> list[int]:
    base = env.unwrapped
    objects = floor_objects(env)
    start = (
        int(base.agent_pos[0]),
        int(base.agent_pos[1]),
        int(base.agent_dir),
    )

    def is_far(state: tuple[int, int, int]) -> bool:
        return all(
            abs(state[0] - obj.position[0]) + abs(state[1] - obj.position[1]) > 1
            for obj in objects
        )

    queue = deque([start])
    parents: dict[tuple[int, int, int], tuple[tuple[int, int, int], int] | None] = {
        start: None
    }
    goal = None
    while queue:
        state = queue.popleft()
        if is_far(state):
            goal = state
            break
        x, y, direction = state
        candidates = (
            ((x, y, (direction - 1) % 4), int(Actions.left)),
            ((x, y, (direction + 1) % 4), int(Actions.right)),
        )
        dx, dy = DIR_TO_VEC[direction]
        if _can_enter(env, x + dx, y + dy):
            candidates += (((x + dx, y + dy, direction), int(Actions.forward)),)
        for next_state, action in candidates:
            if next_state not in parents:
                parents[next_state] = (state, action)
                queue.append(next_state)
    if goal is None:
        raise RuntimeError("no object-far state is reachable")
    actions = []
    state = goal
    while parents[state] is not None:
        previous, action = parents[state]
        actions.append(action)
        state = previous
    actions.reverse()
    return actions


def scripted_relation_audit(env: gym.Env, seed: int) -> GoToObjectScriptResult:
    observation, _ = env.reset(seed=seed)
    mission = str(observation["mission"])
    actions = []
    terminated = truncated = False

    def run(next_actions: list[int]) -> None:
        nonlocal terminated, truncated
        for action in next_actions:
            _, _, terminated, truncated, _ = env.step(action)
            actions.append(int(action))
            if terminated or truncated:
                raise RuntimeError("scripted relation audit ended unexpectedly")

    run(plan_to_far(env))
    if semantic_stage(env) != 0:
        raise RuntimeError("failed to establish object-far relation")
    frames = [env.render()]
    stages = [semantic_stage(env)]
    counts = [len(floor_objects(env))]

    selected = floor_objects(env)[0]
    run(plan_to_face(env, selected.position))
    if semantic_stage(env) != 1:
        raise RuntimeError("failed to establish object-adjacent relation")
    frames.append(env.render())
    stages.append(semantic_stage(env))
    counts.append(len(floor_objects(env)))

    run([int(Actions.pickup)])
    if semantic_stage(env) != 2 or env.unwrapped.carrying is None:
        raise RuntimeError("failed to establish object-carried relation")
    carried = env.unwrapped.carrying
    frames.append(env.render())
    stages.append(semantic_stage(env))
    counts.append(len(floor_objects(env)))
    selected_is_target = bool(
        selected.object_type == env.unwrapped.targetType
        and selected.color == env.unwrapped.target_color
    )
    return GoToObjectScriptResult(
        seed=seed,
        mission=mission,
        selected_object=selected,
        selected_is_mission_target=selected_is_target,
        stages=tuple(stages),
        actions=tuple(actions),
        floor_counts=tuple(counts),
        carrying=(carried.type, carried.color),
        frames=tuple(frames),
    )
