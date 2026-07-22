"""Sample balanced policy-state frames from frozen DoorKey tabular controls."""

from __future__ import annotations

import argparse
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
    DoorKeyTabularConfig,
    _values,
    compact_doorkey_state,
    state_changing_action_indices,
)


@dataclass(frozen=True)
class PolicyStateDatasetConfig:
    generation_groups: tuple[int, ...] = (57, 67)
    layouts_per_group: int = 256
    samples_per_group_stage: int = 128
    seed_stride: int = 100_000
    reservoir_seed: int = 7

    def __post_init__(self) -> None:
        if len(set(self.generation_groups)) != len(self.generation_groups):
            raise ValueError("generation groups must be unique")
        if not self.generation_groups:
            raise ValueError("at least one generation group is required")
        if self.layouts_per_group <= 0 or self.samples_per_group_stage <= 0:
            raise ValueError("layout and sample counts must be positive")
        if self.seed_stride <= 0:
            raise ValueError("seed stride must be positive")


class BalancedStateReservoir:
    def __init__(
        self,
        groups: tuple[int, ...],
        capacity: int,
        *,
        seed: int,
    ):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.rows = {
            (group, stage): []
            for group in groups
            for stage in range(len(DOORKEY_STAGES))
        }
        self.seen = {key: 0 for key in self.rows}

    def add(self, group: int, stage: int, row: dict[str, object]) -> None:
        key = (group, stage)
        if key not in self.rows:
            raise ValueError("sample does not belong to a configured stratum")
        self.seen[key] += 1
        values = self.rows[key]
        if len(values) < self.capacity:
            values.append(row)
            return
        replacement = int(self.rng.integers(self.seen[key]))
        if replacement < self.capacity:
            values[replacement] = row

    def balanced_rows(self) -> list[dict[str, object]]:
        incomplete = {
            key: len(values)
            for key, values in self.rows.items()
            if len(values) != self.capacity
        }
        if incomplete:
            raise RuntimeError(f"policy-state reservoir is incomplete: {incomplete}")
        return [
            row
            for key in sorted(self.rows)
            for row in self.rows[key]
        ]


def _write_contact_sheet(
    path: Path,
    frames: np.ndarray,
    stages: np.ndarray,
    groups: np.ndarray,
    env_seeds: np.ndarray,
    steps: np.ndarray,
) -> list[dict[str, int]]:
    selected = []
    for stage in range(4):
        selected.extend(np.flatnonzero(stages == stage)[:3].tolist())
    frame_height, frame_width = frames.shape[2:4]
    gap = 6
    marker = 9
    sheet = np.full(
        (
            len(selected) * frame_height + (len(selected) - 1) * gap,
            marker + 2 * frame_width + gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    colors = ((104, 117, 125), (40, 117, 164), (209, 112, 49), (43, 137, 95))
    manifest = []
    for row_index, index in enumerate(selected):
        y = row_index * (frame_height + gap)
        stage = int(stages[index])
        sheet[y : y + frame_height, :marker] = colors[stage]
        sheet[y : y + frame_height, marker : marker + frame_width] = frames[index, 0]
        x = marker + frame_width + gap
        sheet[y : y + frame_height, x : x + frame_width] = frames[index, 1]
        manifest.append(
            {
                "row": row_index,
                "sample_index": int(index),
                "stage": stage,
                "generation_group": int(groups[index]),
                "env_seed": int(env_seeds[index]),
                "step": int(steps[index]),
            }
        )
    _write_png(path, sheet)
    return manifest


def generate_policy_state_dataset(
    source_run: Path,
    audit: PolicyStateDatasetConfig,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    source_metrics = json.loads(
        (source_run / "metrics.json").read_text(encoding="utf-8")
    )
    source_config = source_metrics["config"]
    config = DoorKeyTabularConfig(
        env_id=str(source_config["env_id"]),
        seed=int(source_config["seed"]),
        episodes=int(source_config["episodes"]),
        horizon=int(source_config["horizon"]),
        evaluation_checkpoints=tuple(source_config["evaluation_checkpoints"]),
        eval_episodes_per_skill=1,
        stage_rate_gate=float(source_config["stage_rate_gate"]),
        valid_action_mask=bool(source_config["valid_action_mask"]),
        terminate_on_target=bool(source_config["terminate_on_target"]),
    )
    if not config.valid_action_mask or not config.terminate_on_target:
        raise ValueError("source policy must use mask and target option termination")
    q_table = load_doorkey_q_table(source_run / "q_table.npz")
    reservoir = BalancedStateReservoir(
        audit.generation_groups,
        audit.samples_per_group_stage,
        seed=audit.reservoir_seed,
    )
    env = gym.make(config.env_id, render_mode="rgb_array")
    try:
        for group in audit.generation_groups:
            for episode in range(audit.layouts_per_group):
                env_seed = group * audit.seed_stride + episode
                for skill in range(config.num_skills):
                    env.reset(seed=env_seed)
                    reset_frame = env.render().copy()
                    furthest_stage = semantic_stage(env)
                    reservoir.add(
                        group,
                        furthest_stage,
                        {
                            "frames": (reset_frame, reset_frame),
                            "stage": furthest_stage,
                            "group": group,
                            "env_seed": env_seed,
                            "skill": skill,
                            "step": 0,
                        },
                    )
                    for step in range(config.horizon):
                        key = compact_doorkey_state(env)
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
                        current_frame = env.render().copy()
                        reservoir.add(
                            group,
                            furthest_stage,
                            {
                                "frames": (reset_frame, current_frame),
                                "stage": furthest_stage,
                                "group": group,
                                "env_seed": env_seed,
                                "skill": skill,
                                "step": step + 1,
                            },
                        )
                        option_terminated = bool(
                            skill in {1, 2} and furthest_stage == skill
                        )
                        if terminated or truncated or option_terminated:
                            break
    finally:
        env.close()

    rows = reservoir.balanced_rows()
    frames = np.asarray([row["frames"] for row in rows], dtype=np.uint8)
    stages = np.asarray([row["stage"] for row in rows], dtype=np.int8)
    groups = np.asarray([row["group"] for row in rows], dtype=np.int16)
    env_seeds = np.asarray([row["env_seed"] for row in rows], dtype=np.int64)
    skills = np.asarray([row["skill"] for row in rows], dtype=np.int8)
    steps = np.asarray([row["step"] for row in rows], dtype=np.int16)
    dataset_path = output_dir / "policy_state_visual_dataset.npz"
    np.savez_compressed(
        dataset_path,
        frames=frames,
        stages=stages,
        generation_groups=groups,
        env_seeds=env_seeds,
        skills=skills,
        steps=steps,
    )
    image_path = output_dir / "policy_state_manual_audit.png"
    contact_manifest = _write_contact_sheet(
        image_path,
        frames,
        stages,
        groups,
        env_seeds,
        steps,
    )
    (output_dir / "policy_state_manual_audit.json").write_text(
        json.dumps(contact_manifest, indent=2),
        encoding="utf-8",
    )
    candidate_counts = {
        str(group): {
            DOORKEY_STAGES[stage]: reservoir.seen[(group, stage)]
            for stage in range(4)
        }
        for group in audit.generation_groups
    }
    sample_counts = {
        str(group): {
            DOORKEY_STAGES[stage]: int(
                np.count_nonzero((groups == group) & (stages == stage))
            )
            for stage in range(4)
        }
        for group in audit.generation_groups
    }
    output = {
        "source_run": str(source_run.resolve()),
        "source_signal_gate_passed": bool(source_metrics["signal_gate_passed"]),
        "audit_config": asdict(audit),
        "stage_names": DOORKEY_STAGES,
        "sample_count": len(rows),
        "frame_count": int(frames.shape[0] * frames.shape[1]),
        "frame_shape": list(frames.shape[2:]),
        "candidate_counts": candidate_counts,
        "sample_counts": sample_counts,
        "dataset": str(dataset_path.resolve()),
        "manual_audit_image": str(image_path.resolve()),
        "manual_audit_manifest": contact_manifest,
        "data_gate_passed": bool(
            source_metrics["signal_gate_passed"]
            and all(
                count == audit.samples_per_group_stage
                for by_stage in sample_counts.values()
                for count in by_stage.values()
            )
        ),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["metrics"] = str(metrics_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--layouts-per-group", type=int, default=256)
    parser.add_argument("--samples-per-group-stage", type=int, default=128)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    audit = PolicyStateDatasetConfig(
        layouts_per_group=args.layouts_per_group,
        samples_per_group_stage=args.samples_per_group_stage,
    )
    run_id = (
        f"doorkey5_policy_state_visual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/minigrid_doorkey_visual"
    ) / run_id
    output = generate_policy_state_dataset(args.source_run, audit, output_dir)
    print(
        json.dumps(
            {
                "metrics": output["metrics"],
                "dataset": output["dataset"],
                "manual_audit_image": output["manual_audit_image"],
                "sample_count": output["sample_count"],
                "frame_count": output["frame_count"],
                "candidate_counts": output["candidate_counts"],
                "data_gate_passed": output["data_gate_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
