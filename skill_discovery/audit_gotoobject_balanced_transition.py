"""Audit balanced assignments and transition-aware rewards from saved runs."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from skill_discovery.minigrid_gotoobject import GOTOOBJECT_STAGES


def maximum_weight_assignment(
    matrix: np.ndarray,
) -> tuple[tuple[int, ...], list[dict[str, object]]]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] == 0
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("assignment matrix must be finite and square")
    size = matrix.shape[0]
    scores = []
    for assignment in itertools.permutations(range(size)):
        score = float(
            sum(matrix[skill, stage] for skill, stage in enumerate(assignment))
        )
        scores.append({"assignment": assignment, "score": score})
    best = max(
        scores,
        key=lambda row: (
            row["score"],
            tuple(-value for value in row["assignment"]),
        ),
    )
    return tuple(best["assignment"]), scores


def transition_counts_from_stage_episodes(
    episodes: list[list[int]],
    *,
    num_stages: int = len(GOTOOBJECT_STAGES),
) -> np.ndarray:
    if num_stages <= 0:
        raise ValueError("number of stages must be positive")
    counts = np.zeros((num_stages, num_stages), dtype=np.int64)
    for episode in episodes:
        for source, target in zip(episode[:-1], episode[1:]):
            if source < 0 or source >= num_stages:
                raise ValueError("episode contains an invalid source stage")
            if target < 0 or target >= num_stages:
                raise ValueError("episode contains an invalid target stage")
            if source != target:
                counts[source, target] += 1
    return counts


def transition_counts_from_buffer(
    path: Path,
    *,
    num_stages: int = len(GOTOOBJECT_STAGES),
) -> np.ndarray:
    with np.load(path) as archive:
        offsets = np.asarray(archive["episode_offsets"], dtype=np.int64)
        stages = np.asarray(archive["stages"], dtype=np.int64)
    if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0:
        raise ValueError("episode offsets must start at zero")
    if offsets[-1] != len(stages) or np.any(np.diff(offsets) <= 0):
        raise ValueError("episode offsets do not match non-empty transitions")
    if np.any(stages < 0) or np.any(stages >= num_stages):
        raise ValueError("buffer contains an invalid stage")
    episodes = [
        stages[start:stop].tolist()
        for start, stop in zip(offsets[:-1], offsets[1:])
    ]
    return transition_counts_from_stage_episodes(
        episodes,
        num_stages=num_stages,
    )


def supported_predecessors(
    transition_counts: np.ndarray,
    *,
    minimum_count: int = 25,
    minimum_share: float = 0.01,
) -> tuple[tuple[int, ...], ...]:
    counts = np.asarray(transition_counts, dtype=np.int64)
    if (
        counts.ndim != 2
        or counts.shape[0] != counts.shape[1]
        or counts.shape[0] == 0
        or np.any(counts < 0)
    ):
        raise ValueError("transition counts must be nonnegative and square")
    output = []
    for target in range(counts.shape[0]):
        incoming = counts[:, target].copy()
        incoming[target] = 0
        total = int(incoming.sum())
        predecessors = tuple(
            source
            for source, count in enumerate(incoming)
            if count > 0
            and count >= minimum_count
            and total > 0
            and count / total >= minimum_share
        )
        output.append(predecessors)
    return tuple(output)


def transitive_ancestors(
    predecessors: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    size = len(predecessors)
    if any(
        source < 0 or source >= size
        for sources in predecessors
        for source in sources
    ):
        raise ValueError("predecessor graph contains an invalid stage")
    output = []
    for target in range(size):
        ancestors = set()

        def visit(source: int, path: set[int]) -> None:
            if source in path:
                raise ValueError("predecessor graph contains a cycle")
            if source in ancestors:
                return
            next_path = path | {source}
            for predecessor in predecessors[source]:
                visit(predecessor, next_path)
            ancestors.add(source)

        for source in predecessors[target]:
            visit(source, {target})
        output.append(tuple(sorted(ancestors)))
    return tuple(output)


def transition_aware_matrix(
    assignment: tuple[int, ...],
    predecessors: tuple[tuple[int, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    size = len(assignment)
    if sorted(assignment) != list(range(size)):
        raise ValueError("assignment must be a stage permutation")
    if len(predecessors) != size:
        raise ValueError("one predecessor set is required per stage")
    rows = []
    for target in assignment:
        row = np.full(size, -1.0, dtype=np.float64)
        row[list(predecessors[target])] = 0.0
        row[target] = 1.0
        rows.append(tuple(float(value) for value in row))
    return tuple(rows)


def audit_run(run_dir: Path) -> dict[str, object]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    raw = np.asarray(metrics["bootstrap_raw_matrix"], dtype=np.float64)
    assignment, scores = maximum_weight_assignment(raw)
    counts = transition_counts_from_buffer(run_dir / "bootstrap_replay_buffer.npz")
    predecessors = supported_predecessors(counts)
    transformed = transition_aware_matrix(assignment, predecessors)
    independent = tuple(int(value) for value in np.argmax(raw, axis=1))
    target_rows_valid = all(
        row[target] == 1.0
        and sum(value == 1.0 for value in row) == 1
        and all(row[source] == 0.0 for source in predecessors[target])
        for row, target in zip(transformed, assignment)
    )
    gate = bool(
        sorted(assignment) == list(range(3))
        and all(predecessors)
        and target_rows_valid
    )
    return {
        "seed": int(metrics["config"]["seed"]),
        "run_dir": str(run_dir.resolve()),
        "independent_top_stages": independent,
        "independent_assignment_collision": sorted(independent) != list(range(3)),
        "balanced_assignment": assignment,
        "balanced_assignment_names": [GOTOOBJECT_STAGES[index] for index in assignment],
        "permutation_scores": scores,
        "transition_counts": counts.tolist(),
        "supported_predecessors": predecessors,
        "supported_predecessor_names": [
            [GOTOOBJECT_STAGES[index] for index in sources]
            for sources in predecessors
        ],
        "transition_aware_matrix": transformed,
        "structural_gate_passed": gate,
    }


def audit_runs(run_dirs: list[Path], output_path: Path) -> dict[str, object]:
    runs = [audit_run(run_dir) for run_dir in run_dirs]
    seeds = [row["seed"] for row in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("training seeds must be unique")
    output = {
        "method": "maximum_weight_assignment_plus_transition_predecessor_floor",
        "minimum_transition_count": 25,
        "minimum_transition_share": 0.01,
        "runs": sorted(runs, key=lambda row: row["seed"]),
        "seeds_passed": sum(bool(row["structural_gate_passed"]) for row in runs),
        "seeds_total": len(runs),
        "structural_gate_passed": all(
            bool(row["structural_gate_passed"]) for row in runs
        ),
        "runs_training": False,
    }
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_runs(args.run_dirs, args.output), indent=2))


if __name__ == "__main__":
    main()
