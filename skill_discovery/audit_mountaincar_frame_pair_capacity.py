"""Audit rendered frame-pair capacity for MountainCar momentum relations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np

from skill_discovery.audit_mountaincar_continuous_environment import (
    expected_transition,
)
from skill_discovery.generate_point_cup_dataset import _write_png


PAIR_CLASSES = (
    "left_momentum",
    "valley_return",
    "right_climb",
    "native_goal",
)
PROPOSAL_RANGES = (
    ((-1.15, -0.74), (-0.07, -0.005)),
    ((-0.75, 0.0), (0.005, 0.07)),
    ((0.0, 0.45), (0.005, 0.07)),
    ((0.38, 0.45), (0.005, 0.07)),
)


@dataclass(frozen=True)
class FramePairCapacityConfig:
    accepted_per_class: int = 8_192
    minimum_unique_pairs: int = 1_024
    minimum_motion_visible_rate: float = 0.95
    random_seed: int = 2_400_007
    max_attempt_multiplier: int = 100

    def __post_init__(self) -> None:
        if self.accepted_per_class <= 0 or self.minimum_unique_pairs <= 0:
            raise ValueError("capacity counts must be positive")
        if self.minimum_unique_pairs > self.accepted_per_class:
            raise ValueError("unique pair gate cannot exceed accepted transitions")
        if not 0.0 <= self.minimum_motion_visible_rate <= 1.0:
            raise ValueError("motion-visible gate must be a rate")
        if self.max_attempt_multiplier <= 0:
            raise ValueError("attempt multiplier must be positive")


def transition_class(
    class_index: int,
    next_state: np.ndarray,
    terminated: bool,
) -> bool:
    position, velocity = (float(value) for value in next_state)
    if class_index == 0:
        return bool(-1.15 <= position <= -0.75 and velocity <= -0.005 and not terminated)
    if class_index == 1:
        return bool(-0.75 < position < 0.0 and velocity >= 0.005 and not terminated)
    if class_index == 2:
        return bool(0.0 <= position < 0.45 and velocity >= 0.005 and not terminated)
    if class_index == 3:
        return bool(position >= 0.45 and velocity >= 0.0 and terminated)
    raise ValueError("unknown frame-pair class")


def frame_hash(frame: np.ndarray) -> str:
    return hashlib.blake2b(frame.tobytes(), digest_size=16).hexdigest()


def pair_hash(before: np.ndarray, after: np.ndarray) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(before.tobytes())
    digest.update(after.tobytes())
    return digest.hexdigest()


def collect_class_capacity(
    config: FramePairCapacityConfig,
    class_index: int,
) -> tuple[dict[str, object], list[tuple[np.ndarray, np.ndarray]]]:
    rng = np.random.default_rng(config.random_seed + class_index * 100_000)
    (position_low, position_high), (velocity_low, velocity_high) = PROPOSAL_RANGES[
        class_index
    ]
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    env.reset(seed=config.random_seed + class_index)
    pair_hashes: set[str] = set()
    before_hashes: set[str] = set()
    after_hashes: set[str] = set()
    examples: list[tuple[np.ndarray, np.ndarray]] = []
    accepted = 0
    attempts = 0
    motion_visible = 0
    finite_count = 0
    max_attempts = config.accepted_per_class * config.max_attempt_multiplier
    try:
        while accepted < config.accepted_per_class:
            if attempts >= max_attempts:
                raise RuntimeError(
                    f"{PAIR_CLASSES[class_index]} accepted {accepted}/"
                    f"{config.accepted_per_class} after {attempts} attempts"
                )
            attempts += 1
            state = np.asarray(
                [
                    rng.uniform(position_low, position_high),
                    rng.uniform(velocity_low, velocity_high),
                ],
                dtype=np.float32,
            )
            action = np.asarray([rng.uniform(-1.0, 1.0)], dtype=np.float32)
            next_state, _, terminated = expected_transition(state, action)
            if not transition_class(class_index, next_state, terminated):
                continue
            finite_count += int(
                np.isfinite(state).all()
                and np.isfinite(action).all()
                and np.isfinite(next_state).all()
            )
            env.unwrapped.state = state.copy()
            before = env.render().copy()
            env.unwrapped.state = next_state.copy()
            after = env.render().copy()
            before_digest = frame_hash(before)
            after_digest = frame_hash(after)
            before_hashes.add(before_digest)
            after_hashes.add(after_digest)
            pair_hashes.add(pair_hash(before, after))
            motion_visible += int(before_digest != after_digest)
            if len(examples) < 3:
                examples.append((before, after))
            accepted += 1
    finally:
        env.close()
    motion_visible_rate = motion_visible / accepted
    summary = {
        "class_index": class_index,
        "class_name": PAIR_CLASSES[class_index],
        "proposal_position_range": [position_low, position_high],
        "proposal_velocity_range": [velocity_low, velocity_high],
        "attempts": attempts,
        "accepted_transitions": accepted,
        "finite_transition_count": finite_count,
        "unique_before_hashes": len(before_hashes),
        "unique_after_hashes": len(after_hashes),
        "unique_pair_hashes": len(pair_hashes),
        "motion_visible_count": motion_visible,
        "motion_visible_rate": motion_visible_rate,
        "capacity_gate_passed": len(pair_hashes) >= config.minimum_unique_pairs,
        "motion_gate_passed": motion_visible_rate
        >= config.minimum_motion_visible_rate,
        "predicate_gate_passed": finite_count == accepted,
    }
    summary["passed"] = bool(
        summary["capacity_gate_passed"]
        and summary["motion_gate_passed"]
        and summary["predicate_gate_passed"]
    )
    return summary, examples


def _write_contact_sheet(
    path: Path,
    examples_by_class: list[list[tuple[np.ndarray, np.ndarray]]],
) -> None:
    frame_height, frame_width = 200, 300
    marker = 10
    gap = 5
    row_gap = 6
    columns = 6
    sheet = np.full(
        (
            len(PAIR_CLASSES) * frame_height + (len(PAIR_CLASSES) - 1) * row_gap,
            marker + columns * frame_width + (columns - 1) * gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    row_colors = np.asarray(
        ([34, 114, 157], [214, 93, 74], [64, 145, 108], [146, 92, 156]),
        dtype=np.uint8,
    )
    for row_index, examples in enumerate(examples_by_class):
        y = row_index * (frame_height + row_gap)
        sheet[y : y + frame_height, :marker] = row_colors[row_index]
        for example_index, (before, after) in enumerate(examples):
            for pair_offset, frame in enumerate((before, after)):
                column = example_index * 2 + pair_offset
                x = marker + column * (frame_width + gap)
                sheet[y : y + frame_height, x : x + frame_width] = frame[::2, ::2]
    _write_png(path, sheet)


def run_capacity_audit(
    config: FramePairCapacityConfig,
    output_dir: Path,
    *,
    workers: int = 4,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(
            pool.map(
                collect_class_capacity,
                [config] * len(PAIR_CLASSES),
                range(len(PAIR_CLASSES)),
            )
        )
    summaries = [result[0] for result in results]
    examples = [result[1] for result in results]
    contact_sheet_path = output_dir / "frame_pair_capacity_contact_sheet.png"
    _write_contact_sheet(contact_sheet_path, examples)
    output = {
        "config": asdict(config),
        "workers": workers,
        "sdl_video_driver": os.environ.get("SDL_VIDEODRIVER"),
        "classes": summaries,
        "contact_sheet": str(contact_sheet_path.resolve()),
        "contact_sheet_columns": [
            "pair_0_before",
            "pair_0_after",
            "pair_1_before",
            "pair_1_after",
            "pair_2_before",
            "pair_2_after",
        ],
        "numeric_gate_passed": all(row["passed"] for row in summaries),
        "manual_contact_sheet_gate": "pending",
        "passed": False,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = FramePairCapacityConfig()
    run_id = f"frame_pair_capacity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/mountaincar_continuous"
    ) / run_id
    output = run_capacity_audit(config, output_dir, workers=args.workers)
    print(
        json.dumps(
            {
                "output": str((output_dir / "audit.json").resolve()),
                "classes": output["classes"],
                "numeric_gate_passed": output["numeric_gate_passed"],
                "manual_contact_sheet_gate": output["manual_contact_sheet_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
