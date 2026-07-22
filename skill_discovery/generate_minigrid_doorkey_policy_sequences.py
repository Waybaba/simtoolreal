"""Generate complete frozen-policy sequences with a unique DoorKey frame cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np

from skill_discovery.audit_minigrid_doorkey_final_state import (
    load_doorkey_q_table,
)
from skill_discovery.generate_point_cup_dataset import _write_png
from skill_discovery.minigrid_doorkey import DOORKEY_STAGES, semantic_stage
from skill_discovery.train_minigrid_doorkey_tabular import (
    DOORKEY_POLICY_ACTIONS,
    DoorKeyState,
    DoorKeyTabularConfig,
    _values,
    compact_doorkey_state,
    state_changing_action_indices,
)


@dataclass(frozen=True)
class PolicySequenceDatasetConfig:
    generation_groups: tuple[int, ...] = (97, 107)
    layouts_per_group: int = 128
    seed_stride: int = 100_000

    def __post_init__(self) -> None:
        if not self.generation_groups:
            raise ValueError("at least one generation group is required")
        if len(set(self.generation_groups)) != len(self.generation_groups):
            raise ValueError("generation groups must be unique")
        if self.layouts_per_group <= 0 or self.seed_stride <= 0:
            raise ValueError("layout count and seed stride must be positive")


class UniqueRenderedStateCache:
    def __init__(self) -> None:
        self.index_by_key: dict[DoorKeyState, int] = {}
        self.keys: list[DoorKeyState] = []
        self.frames: list[np.ndarray] = []
        self.stages: list[int] = []
        self.hashes: list[str] = []

    def add(self, key: DoorKeyState, frame: np.ndarray, stage: int) -> int:
        image = np.asarray(frame, dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("rendered state must be an RGB image")
        if stage < 0 or stage >= len(DOORKEY_STAGES):
            raise ValueError("oracle stage is outside the DoorKey stage range")
        digest = hashlib.blake2b(image.tobytes(), digest_size=16).hexdigest()
        existing = self.index_by_key.get(key)
        if existing is not None:
            if self.hashes[existing] != digest:
                raise ValueError("compact state maps to multiple RGB frames")
            if self.stages[existing] != stage:
                raise ValueError("compact state maps to multiple oracle stages")
            return existing
        index = len(self.keys)
        self.index_by_key[key] = index
        self.keys.append(key)
        self.frames.append(image.copy())
        self.stages.append(stage)
        self.hashes.append(digest)
        return index


def _write_contact_sheet(
    path: Path,
    frames: np.ndarray,
    stages: np.ndarray,
    keys: np.ndarray,
) -> list[dict[str, object]]:
    columns = 3
    frame_height, frame_width = frames.shape[1:3]
    gap = 6
    marker = 9
    sheet = np.full(
        (
            4 * frame_height + 3 * gap,
            marker + columns * frame_width + (columns - 1) * gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    colors = ((104, 117, 125), (40, 117, 164), (209, 112, 49), (43, 137, 95))
    manifest = []
    for stage in range(4):
        selected = np.flatnonzero(stages == stage)[:columns]
        if len(selected) != columns:
            raise RuntimeError("not enough unique states for contact sheet")
        y = stage * (frame_height + gap)
        sheet[y : y + frame_height, :marker] = colors[stage]
        for column, index in enumerate(selected):
            x = marker + column * (frame_width + gap)
            sheet[y : y + frame_height, x : x + frame_width] = frames[index]
            manifest.append(
                {
                    "stage": stage,
                    "column": column,
                    "state_index": int(index),
                    "compact_state": keys[index].tolist(),
                }
            )
    _write_png(path, sheet)
    return manifest


def generate_policy_sequences(
    source_run: Path,
    audit: PolicySequenceDatasetConfig,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    source_metrics = json.loads(
        (source_run / "metrics.json").read_text(encoding="utf-8")
    )
    source = source_metrics["config"]
    config = DoorKeyTabularConfig(
        env_id=str(source["env_id"]),
        seed=int(source["seed"]),
        episodes=int(source["episodes"]),
        horizon=int(source["horizon"]),
        evaluation_checkpoints=tuple(source["evaluation_checkpoints"]),
        eval_episodes_per_skill=1,
        stage_rate_gate=float(source["stage_rate_gate"]),
        valid_action_mask=bool(source["valid_action_mask"]),
        terminate_on_target=bool(source["terminate_on_target"]),
        target_assignment=tuple(int(value) for value in source["target_assignment"]),
        reward_matrix=tuple(
            tuple(float(value) for value in row)
            for row in source["reward_matrix"]
        ),
    )
    if not config.valid_action_mask or not config.terminate_on_target:
        raise ValueError("source policy must use mask and target option termination")
    q_table = load_doorkey_q_table(source_run / "q_table.npz")
    cache = UniqueRenderedStateCache()
    sequence_indices = []
    offsets = [0]
    sequence_skills = []
    sequence_target_stages = []
    sequence_groups = []
    sequence_env_seeds = []
    sequence_native_success = []
    env = gym.make(config.env_id, render_mode="rgb_array")
    try:
        for group in audit.generation_groups:
            for episode in range(audit.layouts_per_group):
                env_seed = group * audit.seed_stride + episode
                for skill in range(config.num_skills):
                    target_stage = config.target_assignment[skill]
                    env.reset(seed=env_seed)
                    furthest_stage = semantic_stage(env)
                    key = compact_doorkey_state(env)
                    sequence_indices.append(
                        cache.add(key, env.render(), furthest_stage)
                    )
                    native_success = False
                    for step in range(config.horizon):
                        values = _values(q_table, key, create=False)[skill]
                        valid_indices = state_changing_action_indices(env)
                        action_index = max(
                            valid_indices,
                            key=lambda index: (float(values[index]), -index),
                        )
                        _, native_reward, terminated, truncated, _ = env.step(
                            DOORKEY_POLICY_ACTIONS[action_index]
                        )
                        stage = semantic_stage(
                            env,
                            terminated=bool(terminated),
                            reward=float(native_reward),
                        )
                        furthest_stage = max(furthest_stage, stage)
                        key = compact_doorkey_state(env)
                        sequence_indices.append(
                            cache.add(key, env.render(), furthest_stage)
                        )
                        native_success = bool(terminated and native_reward > 0)
                        option_terminated = bool(
                            target_stage in {1, 2}
                            and furthest_stage == target_stage
                        )
                        if terminated or truncated or option_terminated:
                            break
                    offsets.append(len(sequence_indices))
                    sequence_skills.append(skill)
                    sequence_target_stages.append(target_stage)
                    sequence_groups.append(group)
                    sequence_env_seeds.append(env_seed)
                    sequence_native_success.append(native_success)
    finally:
        env.close()

    state_keys = np.asarray(cache.keys, dtype=np.int16)
    state_frames = np.asarray(cache.frames, dtype=np.uint8)
    state_stages = np.asarray(cache.stages, dtype=np.int8)
    sequence_indices_array = np.asarray(sequence_indices, dtype=np.int32)
    offsets_array = np.asarray(offsets, dtype=np.int64)
    sequence_skills_array = np.asarray(sequence_skills, dtype=np.int8)
    sequence_target_stages_array = np.asarray(
        sequence_target_stages,
        dtype=np.int8,
    )
    sequence_groups_array = np.asarray(sequence_groups, dtype=np.int16)
    sequence_env_seeds_array = np.asarray(sequence_env_seeds, dtype=np.int64)
    sequence_native_success_array = np.asarray(
        sequence_native_success,
        dtype=np.bool_,
    )
    dataset_path = output_dir / "policy_sequence_visual_dataset.npz"
    np.savez_compressed(
        dataset_path,
        state_keys=state_keys,
        state_frames=state_frames,
        state_stages=state_stages,
        state_hashes=np.asarray(cache.hashes),
        sequence_offsets=offsets_array,
        sequence_state_indices=sequence_indices_array,
        sequence_skills=sequence_skills_array,
        sequence_target_stages=sequence_target_stages_array,
        sequence_generation_groups=sequence_groups_array,
        sequence_env_seeds=sequence_env_seeds_array,
        sequence_native_success=sequence_native_success_array,
    )
    image_path = output_dir / "unique_state_manual_audit.png"
    contact_manifest = _write_contact_sheet(
        image_path,
        state_frames,
        state_stages,
        state_keys,
    )
    (output_dir / "unique_state_manual_audit.json").write_text(
        json.dumps(contact_manifest, indent=2),
        encoding="utf-8",
    )
    unique_stage_counts = np.bincount(state_stages, minlength=4)
    occurrence_stage_counts = np.bincount(
        state_stages[sequence_indices_array],
        minlength=4,
    )
    sequence_lengths = np.diff(offsets_array)
    expected_sequences = (
        len(audit.generation_groups)
        * audit.layouts_per_group
        * config.num_skills
    )
    goal_sequences = sequence_target_stages_array == 3
    output = {
        "source_run": str(source_run.resolve()),
        "source_signal_gate_passed": bool(source_metrics["signal_gate_passed"]),
        "audit_config": asdict(audit),
        "stage_names": DOORKEY_STAGES,
        "target_assignment": list(config.target_assignment),
        "sequence_count": len(sequence_skills),
        "expected_sequence_count": expected_sequences,
        "state_occurrence_count": len(sequence_indices_array),
        "unique_state_count": len(state_keys),
        "compact_state_alias_count": 0,
        "unique_state_count_by_stage": {
            stage: int(unique_stage_counts[index])
            for index, stage in enumerate(DOORKEY_STAGES)
        },
        "occurrence_count_by_stage": {
            stage: int(occurrence_stage_counts[index])
            for index, stage in enumerate(DOORKEY_STAGES)
        },
        "sequence_length_min": int(sequence_lengths.min()),
        "sequence_length_max": int(sequence_lengths.max()),
        "goal_sequence_count": int(np.count_nonzero(goal_sequences)),
        "goal_native_success_count": int(
            np.count_nonzero(sequence_native_success_array[goal_sequences])
        ),
        "dataset": str(dataset_path.resolve()),
        "manual_audit_image": str(image_path.resolve()),
        "manual_audit_manifest": contact_manifest,
        "data_gate_passed": bool(
            source_metrics["signal_gate_passed"]
            and len(sequence_skills) == expected_sequences
            and np.all(unique_stage_counts > 0)
            and np.all(occurrence_stage_counts > 0)
            and np.all(sequence_native_success_array[goal_sequences])
        ),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["metrics"] = str(metrics_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--generation-groups", type=int, nargs="+", default=(97, 107))
    parser.add_argument("--layouts-per-group", type=int, default=128)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    audit = PolicySequenceDatasetConfig(
        generation_groups=tuple(args.generation_groups),
        layouts_per_group=args.layouts_per_group,
    )
    run_id = f"doorkey5_policy_sequences_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/minigrid_doorkey_visual"
    ) / run_id
    output = generate_policy_sequences(args.source_run, audit, output_dir)
    print(
        json.dumps(
            {
                "metrics": output["metrics"],
                "dataset": output["dataset"],
                "manual_audit_image": output["manual_audit_image"],
                "sequence_count": output["sequence_count"],
                "state_occurrence_count": output["state_occurrence_count"],
                "unique_state_count": output["unique_state_count"],
                "unique_state_count_by_stage": output[
                    "unique_state_count_by_stage"
                ],
                "goal_native_success_count": output[
                    "goal_native_success_count"
                ],
                "data_gate_passed": output["data_gate_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
