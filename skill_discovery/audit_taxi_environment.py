"""Audit official Gymnasium Taxi-v4 before semantic skill experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np

from skill_discovery.generate_point_cup_dataset import _write_png


TAXI_STAGES = ("passenger_waiting", "passenger_onboard", "delivered")


@dataclass(frozen=True)
class TaxiEnvironmentAuditConfig:
    env_id: str = "Taxi-v4"
    seeds: tuple[int, ...] = (7, 17, 29, 37, 47)
    random_episodes: int = 4_096
    random_horizon: int = 200

    def __post_init__(self) -> None:
        if len(set(self.seeds)) != len(self.seeds) or not self.seeds:
            raise ValueError("audit seeds must be non-empty and unique")
        if self.random_episodes <= 0 or self.random_horizon <= 0:
            raise ValueError("random audit budgets must be positive")


def taxi_semantic_stage(base: object, state: int) -> int:
    row, col, passenger, destination = base.decode(int(state))
    if passenger == destination and (row, col) == base.locs[destination]:
        return 2
    if passenger == 4:
        return 1
    return 0


def deterministic_transition(base: object, state: int, action: int) -> tuple[int, int, bool]:
    transitions = base.P[int(state)][int(action)]
    if len(transitions) != 1 or float(transitions[0][0]) != 1.0:
        raise ValueError("Taxi dry transition is not deterministic")
    _, next_state, reward, terminated = transitions[0]
    return int(next_state), int(reward), bool(terminated)


def shortest_path(
    base: object,
    start: int,
    target: Callable[[int], bool],
) -> list[int]:
    queue = deque([int(start)])
    parent: dict[int, tuple[int, int] | None] = {int(start): None}
    target_state = None
    while queue:
        state = queue.popleft()
        if target(state):
            target_state = state
            break
        valid_actions = np.flatnonzero(base.action_mask(state))
        for action in valid_actions:
            next_state, _, _ = deterministic_transition(base, state, int(action))
            if next_state not in parent:
                parent[next_state] = (state, int(action))
                queue.append(next_state)
    if target_state is None:
        raise RuntimeError("Taxi target is unreachable from the start state")
    actions = []
    state = target_state
    while parent[state] is not None:
        previous, action = parent[state]
        actions.append(action)
        state = previous
    return list(reversed(actions))


def shortest_delivery_actions(base: object, start: int) -> list[int]:
    pickup_path = shortest_path(
        base,
        start,
        lambda state: base.decode(state)[2] == 4,
    )
    state = int(start)
    for action in pickup_path:
        state, _, _ = deterministic_transition(base, state, action)
    delivery_path = shortest_path(
        base,
        state,
        lambda candidate: taxi_semantic_stage(base, candidate) == 2,
    )
    actions = [*pickup_path, *delivery_path]
    if actions.count(4) != 1 or actions.count(5) != 1:
        raise RuntimeError("shortest Taxi script is not one pickup and one dropoff")
    return actions


def enumerate_reachable_states(base: object) -> dict[str, object]:
    initial_states = np.flatnonzero(base.initial_state_distrib > 0)
    queue = deque(int(state) for state in initial_states)
    reachable = set(queue)
    terminal_states = set()
    deterministic = True
    while queue:
        state = queue.popleft()
        for action in range(6):
            transitions = base.P[state][action]
            deterministic &= len(transitions) == 1 and float(transitions[0][0]) == 1.0
            for probability, next_state, _, terminated in transitions:
                deterministic &= float(probability) == 1.0
                next_state = int(next_state)
                if terminated:
                    terminal_states.add(next_state)
                    reachable.add(next_state)
                elif next_state not in reachable:
                    reachable.add(next_state)
                    queue.append(next_state)
    return {
        "nominal_state_count": int(base.observation_space.n),
        "initial_state_count": len(initial_states),
        "reachable_state_count": len(reachable),
        "terminal_state_count": len(terminal_states),
        "terminal_states": sorted(terminal_states),
        "dry_transitions_deterministic": bool(deterministic),
    }


def _frame_hash(frame: np.ndarray) -> str:
    return hashlib.blake2b(frame.tobytes(), digest_size=16).hexdigest()


def _orientation_hashes(base: object, state: int) -> list[str]:
    hashes = []
    for orientation in range(4):
        base.s = int(state)
        base.lastaction = orientation
        hashes.append(_frame_hash(base.render()))
    return hashes


def _invalid_action_audit(env_id: str, seed: int) -> dict[str, object]:
    rows = {}
    for action, name in ((4, "illegal_pickup"), (5, "illegal_dropoff")):
        env = gym.make(env_id)
        try:
            state, _ = env.reset(seed=seed)
            base = env.unwrapped
            if action == 4 and base.action_mask(state)[4]:
                movement = int(np.flatnonzero(base.action_mask(state)[:4])[0])
                state, _, _, _, _ = env.step(movement)
            before_stage = taxi_semantic_stage(base, state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            rows[name] = {
                "action": action,
                "state_before": int(state),
                "state_after": int(next_state),
                "reward": int(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "stage_before": before_stage,
                "stage_after": taxi_semantic_stage(base, next_state),
                "passed": bool(
                    reward == -10
                    and next_state == state
                    and not terminated
                    and not truncated
                    and taxi_semantic_stage(base, next_state) == before_stage == 0
                ),
            }
        finally:
            env.close()
    return rows


def scripted_audit(env_id: str, seed: int) -> dict[str, object]:
    env = gym.make(env_id, render_mode="rgb_array")
    try:
        state, _ = env.reset(seed=seed)
        base = env.unwrapped
        actions = shortest_delivery_actions(base, state)
        frames = {"passenger_waiting": env.render().copy()}
        states = {"passenger_waiting": int(state)}
        pickup_count = 0
        dropoff_count = 0
        rewards = []
        terminated = truncated = False
        for action in actions:
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(int(reward))
            if action == 4:
                pickup_count += 1
                states["passenger_onboard"] = int(state)
                frames["passenger_onboard"] = env.render().copy()
            if action == 5:
                dropoff_count += 1
                states["delivered"] = int(state)
                frames["delivered"] = env.render().copy()
        stage_sequence = [taxi_semantic_stage(base, states[name]) for name in TAXI_STAGES]
        orientation_hashes = {
            name: _orientation_hashes(base, states[name]) for name in TAXI_STAGES
        }
        image_shapes = {name: list(frame.shape) for name, frame in frames.items()}
        frame_hashes = {name: _frame_hash(frame) for name, frame in frames.items()}
        passed = bool(
            pickup_count == 1
            and dropoff_count == 1
            and rewards[-1] == 20
            and terminated
            and not truncated
            and stage_sequence == [0, 1, 2]
            and len(set(frame_hashes.values())) == 3
            and all(len(set(hashes)) == 4 for hashes in orientation_hashes.values())
            and all(frame.shape == (350, 550, 3) for frame in frames.values())
            and all(frame.dtype == np.uint8 for frame in frames.values())
        )
        return {
            "seed": seed,
            "initial_state": states["passenger_waiting"],
            "actions": actions,
            "steps": len(actions),
            "pickup_count": pickup_count,
            "dropoff_count": dropoff_count,
            "rewards": rewards,
            "native_terminated": bool(terminated),
            "truncated": bool(truncated),
            "critical_states": states,
            "critical_stage_sequence": stage_sequence,
            "critical_frame_hashes": frame_hashes,
            "orientation_hashes": orientation_hashes,
            "image_shapes": image_shapes,
            "invalid_actions": _invalid_action_audit(env_id, seed),
            "scripted_gate_passed": passed,
            "frames": frames,
        }
    finally:
        env.close()


def random_reachability(config: TaxiEnvironmentAuditConfig) -> dict[str, object]:
    env = gym.make(config.env_id)
    rng = np.random.default_rng(20260722)
    pickup_episodes = 0
    onboard_episodes = 0
    delivered_episodes = 0
    transition_count = 0
    try:
        for episode in range(config.random_episodes):
            state, info = env.reset(seed=1_000_000 + episode)
            base = env.unwrapped
            picked_up = onboard = delivered = False
            previous_passenger = base.decode(state)[2]
            for _ in range(config.random_horizon):
                actions = np.flatnonzero(info["action_mask"])
                action = int(rng.choice(actions))
                state, reward, terminated, truncated, info = env.step(action)
                passenger = base.decode(state)[2]
                picked_up |= previous_passenger < 4 and passenger == 4
                onboard |= passenger == 4
                delivered |= bool(terminated and reward == 20)
                previous_passenger = passenger
                transition_count += 1
                if terminated or truncated:
                    break
            pickup_episodes += int(picked_up)
            onboard_episodes += int(onboard)
            delivered_episodes += int(delivered)
    finally:
        env.close()
    return {
        "episodes": config.random_episodes,
        "horizon": config.random_horizon,
        "transition_count": transition_count,
        "pickup_episode_count": pickup_episodes,
        "onboard_episode_count": onboard_episodes,
        "delivered_episode_count": delivered_episodes,
        "pickup_episode_rate": pickup_episodes / config.random_episodes,
        "onboard_episode_rate": onboard_episodes / config.random_episodes,
        "delivered_episode_rate": delivered_episodes / config.random_episodes,
        "reachability_gate_passed": bool(
            onboard_episodes > 0 and delivered_episodes > 0
        ),
    }


def _write_contact_sheet(path: Path, scripts: list[dict[str, object]]) -> list[dict[str, object]]:
    columns = 4
    gap = 6
    marker = 10
    sample = scripts[0]["frames"][TAXI_STAGES[0]][::2, ::2]
    height, width = sample.shape[:2]
    sheet = np.full(
        (3 * height + 2 * gap, marker + columns * width + 3 * gap, 3),
        255,
        dtype=np.uint8,
    )
    colors = ((104, 117, 125), (40, 117, 164), (43, 137, 95))
    manifest = []
    for stage, name in enumerate(TAXI_STAGES):
        y = stage * (height + gap)
        sheet[y : y + height, :marker] = colors[stage]
        for column, script in enumerate(scripts[:columns]):
            x = marker + column * (width + gap)
            frame = script["frames"][name][::2, ::2]
            sheet[y : y + height, x : x + width] = frame
            manifest.append(
                {
                    "stage": name,
                    "column": column,
                    "seed": int(script["seed"]),
                    "state": int(script["critical_states"][name]),
                }
            )
    _write_png(path, sheet)
    return manifest


def audit_taxi_environment(
    config: TaxiEnvironmentAuditConfig,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    env = gym.make(config.env_id)
    try:
        base = env.unwrapped
        graph = enumerate_reachable_states(base)
        api = {
            "gymnasium_version": gym.__version__,
            "observation_count": int(env.observation_space.n),
            "action_count": int(env.action_space.n),
            "render_modes": list(env.metadata["render_modes"]),
            "max_episode_steps": int(env.spec.max_episode_steps),
        }
    finally:
        env.close()
    scripts = [scripted_audit(config.env_id, seed) for seed in config.seeds]
    image_path = output_dir / "taxi_scripted_stage_audit.png"
    manifest = _write_contact_sheet(image_path, scripts)
    manifest_path = output_dir / "taxi_scripted_stage_audit.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    random = random_reachability(config)
    serializable_scripts = []
    for script in scripts:
        row = {key: value for key, value in script.items() if key != "frames"}
        serializable_scripts.append(row)
    gate = bool(
        api["observation_count"] == 500
        and api["action_count"] == 6
        and graph["initial_state_count"] == 300
        and graph["reachable_state_count"] == 404
        and graph["terminal_state_count"] == 4
        and graph["dry_transitions_deterministic"]
        and all(row["scripted_gate_passed"] for row in serializable_scripts)
        and all(
            all(item["passed"] for item in row["invalid_actions"].values())
            for row in serializable_scripts
        )
        and random["reachability_gate_passed"]
    )
    output = {
        "config": asdict(config),
        "stage_names": TAXI_STAGES,
        "api": api,
        "state_graph": graph,
        "scripted_audits": serializable_scripts,
        "random_reachability": random,
        "manual_audit_image": str(image_path.resolve()),
        "manual_audit_manifest": str(manifest_path.resolve()),
        "environment_gate_passed": gate,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["metrics"] = str(metrics_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-episodes", type=int, default=4_096)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = TaxiEnvironmentAuditConfig(random_episodes=args.random_episodes)
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/taxi"
    ) / f"taxi_environment_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = audit_taxi_environment(config, output_dir)
    print(
        json.dumps(
            {
                "metrics": output["metrics"],
                "manual_audit_image": output["manual_audit_image"],
                "state_graph": output["state_graph"],
                "scripted_gates": [
                    row["scripted_gate_passed"]
                    for row in output["scripted_audits"]
                ],
                "invalid_action_gates": [
                    {
                        name: item["passed"]
                        for name, item in row["invalid_actions"].items()
                    }
                    for row in output["scripted_audits"]
                ],
                "random_reachability": output["random_reachability"],
                "environment_gate_passed": output["environment_gate_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
