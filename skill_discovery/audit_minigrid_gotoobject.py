"""Audit MiniGrid GoToObject as a mission-independent skill environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from minigrid.core.actions import Actions

from skill_discovery.generate_point_cup_dataset import _write_png
from skill_discovery.minigrid_gotoobject import (
    GOTOOBJECT_STAGES,
    floor_objects,
    make_gotoobject,
    reproducibility_signature,
    scripted_relation_audit,
    semantic_stage,
)


RANDOM_ACTIONS = tuple(
    int(action)
    for action in (Actions.left, Actions.right, Actions.forward, Actions.pickup, Actions.drop)
)


def _random_seed_audit(seed: int, episodes: int, horizon: int) -> dict[str, object]:
    env = make_gotoobject()
    rng = np.random.default_rng(seed + 280_000)
    counts = np.zeros(len(GOTOOBJECT_STAGES), dtype=np.int64)
    stage_visits = np.zeros(len(GOTOOBJECT_STAGES), dtype=np.int64)
    try:
        for episode in range(episodes):
            env.reset(seed=seed * 1_000_000 + episode)
            furthest = semantic_stage(env)
            visited = {furthest}
            for _ in range(horizon):
                action = int(rng.choice(RANDOM_ACTIONS))
                _, _, terminated, truncated, _ = env.step(action)
                stage = semantic_stage(env)
                furthest = max(furthest, stage)
                visited.add(stage)
                if terminated or truncated:
                    break
            counts[furthest] += 1
            for stage in visited:
                stage_visits[stage] += 1
    finally:
        env.close()
    return {
        "seed": seed,
        "episodes": episodes,
        "horizon": horizon,
        "furthest_stage_counts": counts.tolist(),
        "episode_stage_visit_counts": stage_visits.tolist(),
    }


def _write_contact_sheet(path: Path, scripted: list[dict[str, object]]) -> None:
    frame_size = 192
    gap = 6
    marker = 9
    sheet = np.full(
        (
            len(scripted) * frame_size + (len(scripted) - 1) * gap,
            marker + 3 * frame_size + 2 * gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    for row, output in enumerate(scripted):
        y = row * (frame_size + gap)
        sheet[y : y + frame_size, :marker] = (
            (43, 137, 94) if output["selected_is_mission_target"] else (198, 72, 58)
        )
        for column, frame in enumerate(output.pop("frames")):
            x = marker + column * (frame_size + gap)
            sheet[y : y + frame_size, x : x + frame_size] = frame
    _write_png(path, sheet)


def run_audit(
    output_dir: Path,
    *,
    seeds: tuple[int, ...] = (7, 17, 27, 37, 47),
    random_episodes_per_seed: int = 2_048,
    random_horizon: int = 64,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    environment = make_gotoobject(render_mode="rgb_array")
    policy_env = make_gotoobject(mission_free=True)
    try:
        raw_observation, _ = environment.reset(seed=seeds[0])
        frame = environment.render()
        policy_observation, _ = policy_env.reset(seed=seeds[0])
        signature_a = reproducibility_signature(environment, seeds[0])
        signature_b = reproducibility_signature(environment, seeds[0])
        environment_audit = {
            "env_id": "MiniGrid-GoToObject-6x6-N2-v0",
            "max_steps": int(environment.unwrapped.max_steps),
            "action_count": int(environment.action_space.n),
            "raw_observation_keys": sorted(raw_observation),
            "raw_image_shape": list(raw_observation["image"].shape),
            "render_shape": list(frame.shape),
            "mission_free_observation_type": type(policy_observation).__name__,
            "mission_free_observation_shape": list(policy_observation.shape),
            "seed_reproducible": signature_a == signature_b,
        }
    finally:
        environment.close()
        policy_env.close()

    scripted_outputs = []
    for seed in seeds:
        env = make_gotoobject(render_mode="rgb_array")
        try:
            result = scripted_relation_audit(env, seed)
            scripted_outputs.append(
                {
                    "seed": seed,
                    "mission": result.mission,
                    "selected_object": {
                        "type": result.selected_object.object_type,
                        "color": result.selected_object.color,
                        "position": list(result.selected_object.position),
                    },
                    "selected_is_mission_target": result.selected_is_mission_target,
                    "stages": list(result.stages),
                    "stage_names": [GOTOOBJECT_STAGES[index] for index in result.stages],
                    "action_count": len(result.actions),
                    "floor_counts": list(result.floor_counts),
                    "carrying": list(result.carrying) if result.carrying else None,
                    "frames": list(result.frames),
                }
            )
        finally:
            env.close()

    image_path = output_dir / "gotoobject_relation_audit.png"
    _write_contact_sheet(image_path, scripted_outputs)
    with ProcessPoolExecutor(max_workers=len(seeds)) as executor:
        random_outputs = list(
            executor.map(
                _random_seed_audit,
                seeds,
                [random_episodes_per_seed] * len(seeds),
                [random_horizon] * len(seeds),
            )
        )
    random_furthest = np.sum(
        np.asarray([row["furthest_stage_counts"] for row in random_outputs]),
        axis=0,
    )
    random_visits = np.sum(
        np.asarray([row["episode_stage_visit_counts"] for row in random_outputs]),
        axis=0,
    )
    scripted_passed = bool(
        all(
            row["stages"] == [0, 1, 2]
            and row["floor_counts"] == [2, 2, 1]
            and row["carrying"]
            == [row["selected_object"]["type"], row["selected_object"]["color"]]
            for row in scripted_outputs
        )
        and any(not row["selected_is_mission_target"] for row in scripted_outputs)
    )
    random_passed = bool(random_visits[1] > 0 and random_visits[2] > 0)
    passed = bool(
        environment_audit["seed_reproducible"]
        and environment_audit["mission_free_observation_type"] == "ndarray"
        and scripted_passed
        and random_passed
    )
    output = {
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("minigrid", "gymnasium", "pygame-ce")
        },
        "environment": environment_audit,
        "stage_names": GOTOOBJECT_STAGES,
        "semantic_label_usage": "mission target is ignored",
        "scripted": scripted_outputs,
        "scripted_gate_passed": scripted_passed,
        "random": {
            "actions": list(RANDOM_ACTIONS),
            "episodes_per_seed": random_episodes_per_seed,
            "horizon": random_horizon,
            "runs": random_outputs,
            "total_furthest_stage_counts": random_furthest.tolist(),
            "total_episode_stage_visit_counts": random_visits.tolist(),
            "gate_passed": random_passed,
        },
        "image": str(image_path.resolve()),
        "passed": passed,
    }
    metrics_path = output_dir / "summary.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["summary"] = str(metrics_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 17, 27, 37, 47))
    parser.add_argument("--random-episodes-per-seed", type=int, default=2_048)
    parser.add_argument("--random-horizon", type=int, default=64)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_id = f"gotoobject_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/minigrid_gotoobject"
    ) / run_id
    output = run_audit(
        output_dir,
        seeds=tuple(args.seeds),
        random_episodes_per_seed=args.random_episodes_per_seed,
        random_horizon=args.random_horizon,
    )
    print(
        json.dumps(
            {
                "summary": output["summary"],
                "image": output["image"],
                "environment": output["environment"],
                "scripted_gate_passed": output["scripted_gate_passed"],
                "random_total_furthest_stage_counts": output["random"][
                    "total_furthest_stage_counts"
                ],
                "random_total_episode_stage_visit_counts": output["random"][
                    "total_episode_stage_visit_counts"
                ],
                "passed": output["passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
