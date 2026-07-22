"""Generate balanced GoToObject RGB reference and random-exploration splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from minigrid.core.actions import Actions

from skill_discovery.generate_point_cup_dataset import _write_png
from skill_discovery.minigrid_doorkey import plan_to_face
from skill_discovery.minigrid_gotoobject import (
    GOTOOBJECT_STAGES,
    floor_objects,
    make_gotoobject,
    plan_to_far,
    semantic_stage,
)


SPLIT_NAMES = ("scripted_reference", "random_exploration_audit")
RANDOM_ACTIONS = tuple(
    int(action)
    for action in (Actions.left, Actions.right, Actions.forward, Actions.pickup, Actions.drop)
)


@dataclass(frozen=True)
class GoToObjectVisualDatasetConfig:
    samples_per_stage: int = 512
    reference_seed_start: int = 200_000
    audit_seed_start: int = 1_300_000
    random_seed: int = 7
    random_horizon: int = 64
    max_reference_episodes: int = 20_000
    max_audit_episodes: int = 20_000

    def __post_init__(self) -> None:
        if self.samples_per_stage <= 0:
            raise ValueError("samples per stage must be positive")
        if self.reference_seed_start == self.audit_seed_start:
            raise ValueError("reference and audit seed ranges must differ")
        if self.random_horizon <= 0:
            raise ValueError("random horizon must be positive")
        if self.max_reference_episodes <= 0 or self.max_audit_episodes <= 0:
            raise ValueError("collection episode limits must be positive")


def frame_hash(frame: np.ndarray) -> str:
    return hashlib.blake2b(frame.tobytes(), digest_size=16).hexdigest()


def _capture(env: object, *, env_seed: int, step: int, source: str) -> dict[str, object]:
    base = env.unwrapped
    frame = env.render().copy()
    objects = sorted(floor_objects(env), key=lambda item: item.position)
    positions = [list(item.position) for item in objects]
    object_types = [item.object_type for item in objects]
    object_colors = [item.color for item in objects]
    while len(positions) < 2:
        positions.append([-1, -1])
        object_types.append("")
        object_colors.append("")
    if len(positions) != 2:
        raise RuntimeError("GoToObject sample must contain at most two floor objects")
    carrying = base.carrying
    return {
        "frame": frame,
        "hash": frame_hash(frame),
        "stage": semantic_stage(env),
        "env_seed": env_seed,
        "step": step,
        "source": source,
        "agent_position": [int(value) for value in base.agent_pos],
        "agent_direction": int(base.agent_dir),
        "floor_count": len(objects),
        "floor_positions": positions,
        "floor_types": object_types,
        "floor_colors": object_colors,
        "carrying": [carrying.type, carrying.color] if carrying is not None else ["", ""],
    }


def _append_if_needed(
    samples: list[list[dict[str, object]]],
    seen_hashes: set[str],
    sample: dict[str, object],
    samples_per_stage: int,
    *,
    forbidden_hashes: set[str] | None = None,
) -> bool:
    stage = int(sample["stage"])
    digest = str(sample["hash"])
    if len(samples[stage]) >= samples_per_stage or digest in seen_hashes:
        return False
    if forbidden_hashes is not None and digest in forbidden_hashes:
        return False
    samples[stage].append(sample)
    seen_hashes.add(digest)
    return True


def _complete(samples: list[list[dict[str, object]]], count: int) -> bool:
    return all(len(rows) >= count for rows in samples)


def _step_many(env: object, actions: list[int]) -> int:
    steps = 0
    for action in actions:
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            raise RuntimeError("scripted reference trajectory ended unexpectedly")
    return steps


def collect_scripted_reference(
    config: GoToObjectVisualDatasetConfig,
) -> tuple[list[dict[str, object]], set[str], int, int]:
    samples = [[] for _ in GOTOOBJECT_STAGES]
    hashes: set[str] = set()
    env = make_gotoobject(render_mode="rgb_array")
    attempts = 0
    skipped_no_far = 0
    try:
        while not _complete(samples, config.samples_per_stage):
            if attempts >= config.max_reference_episodes:
                raise RuntimeError("reference collection did not reach balanced unique quotas")
            env_seed = config.reference_seed_start + attempts
            attempts += 1
            env.reset(seed=env_seed)
            try:
                far_actions = plan_to_far(env)
            except RuntimeError as error:
                if str(error) != "no object-far state is reachable":
                    raise
                skipped_no_far += 1
                continue
            step = _step_many(env, far_actions)
            sample = _capture(env, env_seed=env_seed, step=step, source="scripted_far")
            if sample["stage"] != 0:
                raise RuntimeError("script failed to reach far stage")
            _append_if_needed(samples, hashes, sample, config.samples_per_stage)

            selected = floor_objects(env)[0]
            step += _step_many(env, plan_to_face(env, selected.position))
            sample = _capture(
                env,
                env_seed=env_seed,
                step=step,
                source="scripted_adjacent",
            )
            if sample["stage"] != 1:
                raise RuntimeError("script failed to reach adjacent stage")
            _append_if_needed(samples, hashes, sample, config.samples_per_stage)

            step += _step_many(env, [int(Actions.pickup)])
            sample = _capture(env, env_seed=env_seed, step=step, source="scripted_carried")
            if sample["stage"] != 2:
                raise RuntimeError("script failed to reach carried stage")
            _append_if_needed(samples, hashes, sample, config.samples_per_stage)
    finally:
        env.close()
    return [sample for rows in samples for sample in rows], hashes, attempts, skipped_no_far


def collect_random_audit(
    config: GoToObjectVisualDatasetConfig,
    forbidden_hashes: set[str],
) -> tuple[list[dict[str, object]], set[str], int]:
    samples = [[] for _ in GOTOOBJECT_STAGES]
    hashes: set[str] = set()
    rng = np.random.default_rng(config.random_seed + 950_000)
    env = make_gotoobject(render_mode="rgb_array")
    episodes = 0
    try:
        while not _complete(samples, config.samples_per_stage):
            if episodes >= config.max_audit_episodes:
                raise RuntimeError("random audit did not reach balanced unique quotas")
            env_seed = config.audit_seed_start + episodes
            env.reset(seed=env_seed)
            sample = _capture(env, env_seed=env_seed, step=0, source="random_reset")
            _append_if_needed(
                samples,
                hashes,
                sample,
                config.samples_per_stage,
                forbidden_hashes=forbidden_hashes,
            )
            for step in range(1, config.random_horizon + 1):
                action = int(rng.choice(RANDOM_ACTIONS))
                _, _, terminated, truncated, _ = env.step(action)
                sample = _capture(
                    env,
                    env_seed=env_seed,
                    step=step,
                    source="random_step",
                )
                _append_if_needed(
                    samples,
                    hashes,
                    sample,
                    config.samples_per_stage,
                    forbidden_hashes=forbidden_hashes,
                )
                if terminated or truncated or _complete(samples, config.samples_per_stage):
                    break
            episodes += 1
    finally:
        env.close()
    return [sample for rows in samples for sample in rows], hashes, episodes


def _write_contact_sheet(
    path: Path,
    samples: list[dict[str, object]],
    samples_per_cell: int = 4,
) -> list[dict[str, object]]:
    frame_size = 192
    gap = 6
    marker = 10
    columns = len(SPLIT_NAMES) * samples_per_cell
    sheet = np.full(
        (
            len(GOTOOBJECT_STAGES) * frame_size + (len(GOTOOBJECT_STAGES) - 1) * gap,
            marker + columns * frame_size + (columns - 1) * gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    colors = ((104, 117, 125), (40, 117, 164), (43, 137, 95))
    manifest = []
    for stage, stage_name in enumerate(GOTOOBJECT_STAGES):
        y = stage * (frame_size + gap)
        sheet[y : y + frame_size, :marker] = colors[stage]
        for split, split_name in enumerate(SPLIT_NAMES):
            candidates = [
                sample
                for sample in samples
                if sample["split"] == split and sample["stage"] == stage
            ][:samples_per_cell]
            if len(candidates) != samples_per_cell:
                raise RuntimeError("contact sheet is missing a split-stage sample")
            for offset, sample in enumerate(candidates):
                column = split * samples_per_cell + offset
                x = marker + column * (frame_size + gap)
                sheet[y : y + frame_size, x : x + frame_size] = sample["frame"]
                manifest.append(
                    {
                        "stage": stage_name,
                        "split": split_name,
                        "env_seed": sample["env_seed"],
                        "step": sample["step"],
                        "frame_hash": sample["hash"],
                    }
                )
    _write_png(path, sheet)
    return manifest


def generate_dataset(
    config: GoToObjectVisualDatasetConfig,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    (
        reference,
        reference_hashes,
        reference_episodes,
        reference_no_far_skips,
    ) = collect_scripted_reference(config)
    audit, audit_hashes, audit_episodes = collect_random_audit(config, reference_hashes)
    for sample in reference:
        sample["split"] = 0
    for sample in audit:
        sample["split"] = 1
    samples = [*reference, *audit]
    frames = np.stack([sample["frame"] for sample in samples]).astype(np.uint8)
    stages = np.asarray([sample["stage"] for sample in samples], dtype=np.int8)
    splits = np.asarray([sample["split"] for sample in samples], dtype=np.int8)
    hashes = np.asarray([sample["hash"] for sample in samples])
    env_seeds = np.asarray([sample["env_seed"] for sample in samples], dtype=np.int64)
    steps = np.asarray([sample["step"] for sample in samples], dtype=np.int16)
    agent_positions = np.asarray(
        [sample["agent_position"] for sample in samples], dtype=np.int8
    )
    agent_directions = np.asarray(
        [sample["agent_direction"] for sample in samples], dtype=np.int8
    )
    floor_counts = np.asarray(
        [sample["floor_count"] for sample in samples], dtype=np.int8
    )
    floor_positions = np.asarray(
        [sample["floor_positions"] for sample in samples], dtype=np.int8
    )
    floor_types = np.asarray([sample["floor_types"] for sample in samples])
    floor_colors = np.asarray([sample["floor_colors"] for sample in samples])
    carrying = np.asarray([sample["carrying"] for sample in samples])
    split_stage_counts = np.stack(
        [np.bincount(stages[splits == split], minlength=3) for split in range(2)]
    )
    expected = np.full((2, 3), config.samples_per_stage, dtype=np.int64)
    hash_overlap = reference_hashes & audit_hashes
    data_gate = bool(
        frames.shape == (config.samples_per_stage * 6, 192, 192, 3)
        and np.array_equal(split_stage_counts, expected)
        and len(reference_hashes) == config.samples_per_stage * 3
        and len(audit_hashes) == config.samples_per_stage * 3
        and not hash_overlap
        and np.all((floor_counts == 1) == (stages == 2))
        and np.all((floor_counts == 2) == (stages != 2))
    )
    dataset_path = output_dir / "gotoobject_visual_dataset.npz"
    np.savez_compressed(
        dataset_path,
        frames=frames,
        stages=stages,
        splits=splits,
        frame_hashes=hashes,
        env_seeds=env_seeds,
        steps=steps,
        agent_positions=agent_positions,
        agent_directions=agent_directions,
        floor_counts=floor_counts,
        floor_positions=floor_positions,
        floor_types=floor_types,
        floor_colors=floor_colors,
        carrying=carrying,
    )
    image_path = output_dir / "gotoobject_visual_contact_sheet.png"
    contact_manifest = _write_contact_sheet(
        image_path,
        samples,
        samples_per_cell=min(4, config.samples_per_stage),
    )
    manifest_path = output_dir / "gotoobject_visual_contact_sheet.json"
    manifest_path.write_text(json.dumps(contact_manifest, indent=2), encoding="utf-8")
    output = {
        "config": asdict(config),
        "stage_names": GOTOOBJECT_STAGES,
        "split_names": SPLIT_NAMES,
        "frame_count": len(frames),
        "frame_shape": list(frames.shape[1:]),
        "split_stage_counts": {
            SPLIT_NAMES[index]: split_stage_counts[index].tolist()
            for index in range(2)
        },
        "reference_episodes_consumed": reference_episodes,
        "reference_layouts_skipped_without_far_state": reference_no_far_skips,
        "audit_episodes_consumed": audit_episodes,
        "reference_unique_hashes": len(reference_hashes),
        "audit_unique_hashes": len(audit_hashes),
        "cross_split_hash_overlap": len(hash_overlap),
        "random_actions": list(RANDOM_ACTIONS),
        "dataset": str(dataset_path.resolve()),
        "manual_audit_image": str(image_path.resolve()),
        "manual_audit_manifest": str(manifest_path.resolve()),
        "data_gate_passed": data_gate,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["metrics"] = str(metrics_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-stage", type=int, default=512)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = GoToObjectVisualDatasetConfig(samples_per_stage=args.samples_per_stage)
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/minigrid_gotoobject_visual"
    ) / f"gotoobject_visual_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = generate_dataset(config, output_dir)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
