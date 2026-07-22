"""Audit capacity for frozen MountainCar stratified none proposals."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np

from skill_discovery.audit_mountaincar_continuous_environment import (
    expected_transition,
)
from skill_discovery.audit_mountaincar_frame_pair_capacity import (
    PAIR_CLASSES,
    pair_hash,
    transition_class,
)
from skill_discovery.generate_point_cup_dataset import _write_png


@dataclass(frozen=True)
class NoneStratum:
    name: str
    next_position_range: tuple[float, float]
    next_velocity_range: tuple[float, float]
    proposal_position_range: tuple[float, float]
    proposal_velocity_range: tuple[float, float]
    position_low_closed: bool = True
    position_high_closed: bool = False
    velocity_low_closed: bool = True
    velocity_high_closed: bool = False
    minimum_absolute_velocity: float = 0.0


STRATA = (
    NoneStratum(
        "left_edge",
        (-1.20, -1.15),
        (-0.07, 0.07),
        (-1.20, -1.14),
        (-0.02, 0.02),
        velocity_high_closed=True,
    ),
    NoneStratum(
        "left_threshold",
        (-1.15, -0.75),
        (-0.005, 0.005),
        (-1.16, -0.74),
        (-0.012, 0.012),
        position_high_closed=True,
        velocity_low_closed=False,
    ),
    NoneStratum(
        "left_reverse",
        (-1.15, -0.75),
        (0.005, 0.07),
        (-1.16, -0.74),
        (0.002, 0.07),
        position_high_closed=True,
        velocity_high_closed=True,
    ),
    NoneStratum(
        "valley_threshold",
        (-0.75, 0.0),
        (-0.005, 0.005),
        (-0.76, 0.01),
        (-0.012, 0.012),
        position_low_closed=False,
    ),
    NoneStratum(
        "valley_reverse",
        (-0.75, 0.0),
        (-0.07, -0.005),
        (-0.76, 0.01),
        (-0.07, -0.002),
        position_low_closed=False,
    ),
    NoneStratum(
        "right_threshold",
        (0.0, 0.45),
        (-0.005, 0.005),
        (-0.01, 0.46),
        (-0.012, 0.012),
    ),
    NoneStratum(
        "right_reverse_body",
        (0.0, 0.38),
        (-0.07, -0.005),
        (-0.01, 0.39),
        (-0.07, -0.002),
    ),
    NoneStratum(
        "right_reverse_goal_edge",
        (0.38, 0.45),
        (-0.07, -0.005),
        (0.37, 0.46),
        (-0.07, -0.002),
    ),
)

VISIBLE_STRATA = (
    *STRATA[:3],
    replace(STRATA[3], minimum_absolute_velocity=0.001),
    *STRATA[4:],
)


@dataclass(frozen=True)
class StratifiedNoneCapacityConfig:
    accepted_per_stratum: int = 256
    minimum_unique_pairs: int = 128
    minimum_motion_visible_rate: float = 0.95
    random_seed: int = 8_150_007
    max_attempt_multiplier: int = 100

    def __post_init__(self) -> None:
        if self.accepted_per_stratum <= 0 or self.minimum_unique_pairs <= 0:
            raise ValueError("capacity counts must be positive")
        if self.minimum_unique_pairs > self.accepted_per_stratum:
            raise ValueError("unique-pair gate cannot exceed accepted count")
        if not 0.0 <= self.minimum_motion_visible_rate <= 1.0:
            raise ValueError("motion gate must be a rate")
        if self.max_attempt_multiplier <= 0:
            raise ValueError("attempt multiplier must be positive")


def _in_interval(
    value: float,
    bounds: tuple[float, float],
    low_closed: bool,
    high_closed: bool,
) -> bool:
    low, high = bounds
    low_ok = value >= low if low_closed else value > low
    high_ok = value <= high if high_closed else value < high
    return bool(low_ok and high_ok)


def stratum_matches(
    stratum_index: int,
    next_state: np.ndarray,
    terminated: bool,
    strata: tuple[NoneStratum, ...] = STRATA,
) -> bool:
    spec = strata[stratum_index]
    position, velocity = (float(value) for value in next_state)
    no_relation = not any(
        transition_class(class_index, next_state, terminated)
        for class_index in range(len(PAIR_CLASSES))
    )
    return bool(
        no_relation
        and _in_interval(
            position,
            spec.next_position_range,
            spec.position_low_closed,
            spec.position_high_closed,
        )
        and _in_interval(
            velocity,
            spec.next_velocity_range,
            spec.velocity_low_closed,
            spec.velocity_high_closed,
        )
        and abs(velocity) >= spec.minimum_absolute_velocity
    )


def sample_candidate(
    rng: np.random.Generator,
    stratum_index: int,
    strata: tuple[NoneStratum, ...] = STRATA,
) -> tuple[np.ndarray, np.ndarray]:
    spec = strata[stratum_index]
    state = np.asarray(
        [
            rng.uniform(*spec.proposal_position_range),
            rng.uniform(*spec.proposal_velocity_range),
        ],
        dtype=np.float32,
    )
    action = np.asarray([rng.uniform(-1.0, 1.0)], dtype=np.float32)
    return state, action


def collect_stratum_capacity(
    config: StratifiedNoneCapacityConfig,
    stratum_index: int,
    strata: tuple[NoneStratum, ...] = STRATA,
) -> tuple[dict[str, object], list[np.ndarray]]:
    seed = config.random_seed + 100_000 * stratum_index
    rng = np.random.default_rng(seed)
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    env.reset(seed=seed)
    pair_hashes: set[str] = set()
    examples = []
    accepted = 0
    attempts = 0
    motion_visible = 0
    finite_count = 0
    max_attempts = config.accepted_per_stratum * config.max_attempt_multiplier
    try:
        while accepted < config.accepted_per_stratum:
            if attempts >= max_attempts:
                raise RuntimeError(
                    f"{strata[stratum_index].name} collected {accepted}/"
                    f"{config.accepted_per_stratum} after {attempts} attempts"
                )
            attempts += 1
            state, action = sample_candidate(rng, stratum_index, strata)
            next_state, _, terminated = expected_transition(state, action)
            if not stratum_matches(stratum_index, next_state, terminated, strata):
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
            digest = pair_hash(before, after)
            pair_hashes.add(digest)
            motion_visible += int(not np.array_equal(before, after))
            if len(examples) < 2:
                examples.append(np.stack((before, after)))
            accepted += 1
    finally:
        env.close()
    unique_pairs = len(pair_hashes)
    motion_visible_rate = motion_visible / accepted
    output = {
        "stratum_index": stratum_index,
        "stratum": asdict(strata[stratum_index]),
        "seed": seed,
        "attempts": attempts,
        "accepted_transitions": accepted,
        "finite_transition_count": finite_count,
        "unique_pair_hashes": unique_pairs,
        "motion_visible_rate": motion_visible_rate,
        "capacity_gate_passed": unique_pairs >= config.minimum_unique_pairs,
        "motion_gate_passed": (
            motion_visible_rate >= config.minimum_motion_visible_rate
        ),
        "predicate_gate_passed": finite_count == accepted,
    }
    output["passed"] = bool(
        output["capacity_gate_passed"]
        and output["motion_gate_passed"]
        and output["predicate_gate_passed"]
    )
    return output, examples


def _write_contact_sheet(path: Path, rows: list[list[np.ndarray]]) -> None:
    frame_height, frame_width = 200, 300
    marker, gap, row_gap = 10, 5, 6
    columns = 4
    sheet = np.full(
        (
            len(rows) * frame_height + (len(rows) - 1) * row_gap,
            marker + columns * frame_width + (columns - 1) * gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    for row_index, examples in enumerate(rows):
        y = row_index * (frame_height + row_gap)
        color = np.asarray(
            [60 + 20 * (row_index % 4), 95 + 16 * row_index, 155 - 10 * row_index],
            dtype=np.uint8,
        )
        sheet[y : y + frame_height, :marker] = color
        for example_index, frames in enumerate(examples):
            for frame_index, frame in enumerate(frames):
                column = 2 * example_index + frame_index
                x = marker + column * (frame_width + gap)
                sheet[y : y + frame_height, x : x + frame_width] = frame[::2, ::2]
    _write_png(path, sheet)


def run_capacity_audit(
    config: StratifiedNoneCapacityConfig,
    output_dir: Path,
    *,
    workers: int = 4,
    strata: tuple[NoneStratum, ...] = STRATA,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(
            pool.map(
                collect_stratum_capacity,
                [config] * len(STRATA),
                range(len(STRATA)),
                [strata] * len(STRATA),
            )
        )
    summaries = [result[0] for result in results]
    contact_sheet_path = output_dir / "stratified_none_capacity_contact_sheet.png"
    _write_contact_sheet(contact_sheet_path, [result[1] for result in results])
    output = {
        "config": asdict(config),
        "workers": workers,
        "sdl_video_driver": os.environ.get("SDL_VIDEODRIVER"),
        "strata": summaries,
        "contact_sheet": str(contact_sheet_path.resolve()),
        "numeric_gate_passed": all(row["passed"] for row in summaries),
        "manual_contact_sheet_gate": "pending",
        "passed": False,
    }
    metrics_path = output_dir / "audit.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["metrics"] = str(metrics_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = StratifiedNoneCapacityConfig()
    run_id = f"stratified_none_capacity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/mountaincar_continuous"
    ) / run_id
    output = run_capacity_audit(config, output_dir, workers=args.workers)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
