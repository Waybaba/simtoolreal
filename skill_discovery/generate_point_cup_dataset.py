"""Generate random and balanced trajectories for the Point-Cup metric probe."""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from skill_discovery.point_cup import LABEL_TO_ID, PointCupEnv, ShapeWorldConfig


AUDIT_CLASSES = ("outside", "inside", "entering", "leaving")


def _trajectory_class(labels: np.ndarray) -> np.ndarray:
    final_inside = np.isin(labels[:, -1], [LABEL_TO_ID["inside"], LABEL_TO_ID["entering"]])
    entered = np.any(labels == LABEL_TO_ID["entering"], axis=1)
    left = np.any(labels == LABEL_TO_ID["leaving"], axis=1)
    inside_fraction = np.mean(
        np.isin(labels, [LABEL_TO_ID["inside"], LABEL_TO_ID["entering"]]),
        axis=1,
    )

    classes = np.zeros(labels.shape[0], dtype=np.int64)
    classes[inside_fraction >= 0.8] = 1
    classes[entered & final_inside] = 2
    classes[left & ~final_inside] = 3
    return classes


def _record_random(config: ShapeWorldConfig, steps: int, seed: int) -> dict[str, np.ndarray]:
    env = PointCupEnv(config)
    observation = env.reset(seed=seed)
    states = np.empty((config.num_envs, steps + 1, 2), dtype=np.float32)
    labels = np.empty((config.num_envs, steps + 1), dtype=np.int8)
    actions = np.empty((config.num_envs, steps, 2), dtype=np.float32)
    states[:, 0] = observation["state"]
    labels[:, 0] = observation["semantic_label"]

    rng = np.random.default_rng(seed + 1)
    momentum = np.zeros((config.num_envs, 2), dtype=np.float32)
    # Random policies span both near-static and highly mobile behavior. This
    # keeps rare semantic modes present without balancing or labeling the pool.
    motion_scale = np.exp(
        rng.uniform(np.log(0.02), np.log(1.0), size=(config.num_envs, 1))
    ).astype(np.float32)
    for step in range(steps):
        momentum = 0.82 * momentum + 0.42 * rng.normal(size=momentum.shape)
        action = np.clip(momentum * motion_scale, -1.0, 1.0).astype(np.float32)
        observation, semantic_label, _ = env.step(action)
        actions[:, step] = action
        states[:, step + 1] = observation["state"]
        labels[:, step + 1] = semantic_label

    return {
        "states": states,
        "actions": actions,
        "semantic_labels": labels,
        "episode_class": _trajectory_class(labels),
    }


def _record_balanced(
    config: ShapeWorldConfig,
    steps: int,
    per_class: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], PointCupEnv]:
    num_envs = per_class * len(AUDIT_CLASSES)
    audit_config = ShapeWorldConfig(**{**asdict(config), "num_envs": num_envs})
    env = PointCupEnv(audit_config)
    rng = np.random.default_rng(seed)
    class_ids = np.repeat(np.arange(len(AUDIT_CLASSES), dtype=np.int64), per_class)
    angles = rng.uniform(-np.pi, np.pi, size=num_envs)
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32)

    starts = np.zeros((num_envs, 2), dtype=np.float32)
    targets = np.zeros((num_envs, 2), dtype=np.float32)

    outside = class_ids == 0
    inside = class_ids == 1
    entering = class_ids == 2
    leaving = class_ids == 3
    starts[outside] = directions[outside] * rng.uniform(0.35, 0.9, size=(outside.sum(), 1))
    targets[outside] = starts[outside]
    starts[inside] = directions[inside] * rng.uniform(0.0, 0.055, size=(inside.sum(), 1))
    targets[inside] = np.zeros((inside.sum(), 2), dtype=np.float32)
    starts[entering] = directions[entering] * rng.uniform(0.48, 0.82, size=(entering.sum(), 1))
    targets[entering] = np.zeros((entering.sum(), 2), dtype=np.float32)
    starts[leaving] = directions[leaving] * rng.uniform(0.0, 0.055, size=(leaving.sum(), 1))
    targets[leaving] = directions[leaving] * rng.uniform(0.48, 0.82, size=(leaving.sum(), 1))

    observation = env.reset(positions=starts)
    states = np.empty((num_envs, steps + 1, 2), dtype=np.float32)
    labels = np.empty((num_envs, steps + 1), dtype=np.int8)
    actions = np.empty((num_envs, steps, 2), dtype=np.float32)
    states[:, 0] = observation["state"]
    labels[:, 0] = observation["semantic_label"]

    for step in range(steps):
        delta = targets - env.positions
        action = delta / config.action_scale
        settled = np.linalg.norm(delta, axis=-1) < 0.018
        action[settled] = 0.04 * rng.normal(size=(settled.sum(), 2))
        action[outside] += 0.025 * rng.normal(size=(outside.sum(), 2))
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        observation, semantic_label, _ = env.step(action)
        actions[:, step] = action
        states[:, step + 1] = observation["state"]
        labels[:, step + 1] = semantic_label

    return (
        {
            "states": states,
            "actions": actions,
            "semantic_labels": labels,
            "episode_class": class_ids,
            "observed_class": _trajectory_class(labels),
        },
        env,
    )


def _record_nuisance(
    config: ShapeWorldConfig,
    steps: int,
    layout_count: int,
    per_class: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Generate controlled layout changes with matched semantic behaviors."""

    class_count = len(AUDIT_CLASSES)
    num_envs = layout_count * class_count * per_class
    nuisance_config = ShapeWorldConfig(**{**asdict(config), "num_envs": num_envs})
    env = PointCupEnv(nuisance_config)
    rng = np.random.default_rng(seed)

    layout_ids = np.repeat(np.arange(layout_count, dtype=np.int64), class_count * per_class)
    class_ids = np.tile(np.repeat(np.arange(class_count, dtype=np.int64), per_class), layout_count)
    instance_ids = np.tile(np.arange(per_class, dtype=np.int64), layout_count * class_count)

    layout_centers = rng.uniform(-0.38, 0.38, size=(layout_count, 2)).astype(np.float32)
    layout_axes = rng.uniform(0.085, 0.19, size=(layout_count, 2)).astype(np.float32)
    centers = layout_centers[layout_ids]
    axes = layout_axes[layout_ids]
    env.set_layout(centers=centers, axes=axes)

    base_angles = rng.uniform(-np.pi, np.pi, size=per_class)
    directions = np.stack(
        [np.cos(base_angles[instance_ids]), np.sin(base_angles[instance_ids])],
        axis=-1,
    ).astype(np.float32)

    def relative_position(radius: float) -> np.ndarray:
        return centers + axes * directions * radius

    starts = np.empty((num_envs, 2), dtype=np.float32)
    targets = np.empty((num_envs, 2), dtype=np.float32)
    outside = class_ids == 0
    inside = class_ids == 1
    entering = class_ids == 2
    leaving = class_ids == 3
    starts[outside] = relative_position(1.22)[outside]
    targets[outside] = starts[outside]
    starts[inside] = relative_position(0.58)[inside]
    targets[inside] = starts[inside]
    starts[entering] = relative_position(1.22)[entering]
    targets[entering] = relative_position(0.58)[entering]
    starts[leaving] = relative_position(0.58)[leaving]
    targets[leaving] = relative_position(1.22)[leaving]

    observation = env.reset(positions=starts)
    states = np.empty((num_envs, steps + 1, 2), dtype=np.float32)
    labels = np.empty((num_envs, steps + 1), dtype=np.int8)
    actions = np.empty((num_envs, steps, 2), dtype=np.float32)
    states[:, 0] = observation["state"]
    labels[:, 0] = observation["semantic_label"]

    for step in range(steps):
        delta = targets - env.positions
        action = np.clip(delta / config.action_scale, -1.0, 1.0).astype(np.float32)
        observation, semantic_label, _ = env.step(action)
        actions[:, step] = action
        states[:, step + 1] = observation["state"]
        labels[:, step + 1] = semantic_label

    return {
        "states": states,
        "actions": actions,
        "semantic_labels": labels,
        "episode_class": class_ids,
        "observed_class": _trajectory_class(labels),
        "layout_id": layout_ids,
        "instance_id": instance_ids,
        "cup_centers": centers,
        "cup_axes": axes,
        "layout_centers": layout_centers,
        "layout_axes": layout_axes,
    }


def _image_grid(images: np.ndarray, columns: int = 4, gap: int = 3) -> np.ndarray:
    rows = int(np.ceil(len(images) / columns))
    height, width = images.shape[1:3]
    grid = np.full(
        (rows * height + (rows - 1) * gap, columns * width + (columns - 1) * gap, 3),
        255,
        dtype=np.uint8,
    )
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        y = row * (height + gap)
        x = column * (width + gap)
        grid[y : y + height, x : x + width] = image
    return grid


def _write_png(path: Path, image: np.ndarray) -> None:
    height, width = image.shape[:2]

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += chunk(b"IDAT", zlib.compress(rows, level=9))
    payload += chunk(b"IEND", b"")
    path.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-episodes", type=int, default=4096)
    parser.add_argument("--audit-per-class", type=int, default=256)
    parser.add_argument("--nuisance-layouts", type=int, default=12)
    parser.add_argument("--nuisance-per-class", type=int, default=16)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if min(
        args.random_episodes,
        args.audit_per_class,
        args.nuisance_layouts,
        args.nuisance_per_class,
        args.steps,
    ) <= 0:
        raise ValueError("episode counts and steps must be positive")
    run_id = datetime.now().strftime("point_cup_probe_%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("outputs/skill_discovery/point_cup") / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    base_config = ShapeWorldConfig(num_envs=args.random_episodes, episode_length=args.steps, seed=args.seed)
    discovery = _record_random(base_config, args.steps, args.seed)
    audit, audit_env = _record_balanced(base_config, args.steps, args.audit_per_class, args.seed + 100)
    nuisance = _record_nuisance(
        base_config,
        args.steps,
        args.nuisance_layouts,
        args.nuisance_per_class,
        args.seed + 200,
    )

    np.savez_compressed(
        output_dir / "dataset.npz",
        discovery_states=discovery["states"],
        discovery_actions=discovery["actions"],
        discovery_semantic_labels=discovery["semantic_labels"],
        discovery_episode_class=discovery["episode_class"],
        audit_states=audit["states"],
        audit_actions=audit["actions"],
        audit_semantic_labels=audit["semantic_labels"],
        audit_episode_class=audit["episode_class"],
        audit_observed_class=audit["observed_class"],
        nuisance_states=nuisance["states"],
        nuisance_actions=nuisance["actions"],
        nuisance_semantic_labels=nuisance["semantic_labels"],
        nuisance_episode_class=nuisance["episode_class"],
        nuisance_observed_class=nuisance["observed_class"],
        nuisance_layout_id=nuisance["layout_id"],
        nuisance_instance_id=nuisance["instance_id"],
        nuisance_cup_centers=nuisance["cup_centers"],
        nuisance_cup_axes=nuisance["cup_axes"],
        nuisance_layout_centers=nuisance["layout_centers"],
        nuisance_layout_axes=nuisance["layout_axes"],
        class_names=np.asarray(AUDIT_CLASSES),
    )
    config_payload = {
        "run_id": run_id,
        "environment": asdict(base_config),
        "random_episodes": args.random_episodes,
        "audit_per_class": args.audit_per_class,
        "nuisance_layouts": args.nuisance_layouts,
        "nuisance_per_class": args.nuisance_per_class,
        "steps": args.steps,
        "seed": args.seed,
    }
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    preview_ids = np.concatenate(
        [np.arange(class_id * args.audit_per_class, class_id * args.audit_per_class + 4) for class_id in range(4)]
    )
    preview = _image_grid(audit_env.render(preview_ids, size=72))
    _write_png(output_dir / "preview.png", preview)
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
