"""Build a finite DINOv2 frame lookup for online-free visual cluster rewards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from skill_discovery.encode_frozenlake_dinov2 import (
    compose_trajectory_embeddings,
)


ACTIVE = 0
TERMINAL_HOLE = 1
TERMINAL_GOAL = 2
HOLE_STATES = {5, 7, 11, 12}


def trajectory_frame_keys(
    states: np.ndarray,
    length: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    middle_index = (length + 1) // 2
    final_state = int(states[length])
    final_type = (
        TERMINAL_HOLE
        if final_state in HOLE_STATES
        else TERMINAL_GOAL
        if final_state == 15
        else ACTIVE
    )
    return (
        (ACTIVE, int(states[0])),
        (ACTIVE, int(states[middle_index])),
        (final_type, final_state),
    )


def _nearest_centers(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = (
        np.sum(features * features, axis=1)[:, None]
        + np.sum(centers * centers, axis=1)[None, :]
        - 2.0 * features @ centers.T
    )
    return np.argmin(distances, axis=1)


def audit_finite_keys(
    lookup: np.ndarray,
    centers: np.ndarray,
) -> dict[str, object]:
    active_states = (0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14)
    terminal_groups = {
        "safe_timeout": (ACTIVE, active_states),
        "hole_terminal": (TERMINAL_HOLE, (5, 7, 11, 12)),
        "goal_terminal": (TERMINAL_GOAL, (15,)),
    }
    clusters = {}
    combination_counts = {}
    for name, (final_type, final_states) in terminal_groups.items():
        trajectories = []
        for middle_state in active_states:
            for final_state in final_states:
                trajectories.extend(
                    (
                        lookup[ACTIVE, 0],
                        lookup[ACTIVE, middle_state],
                        lookup[final_type, final_state],
                    )
                )
        features = compose_trajectory_embeddings(
            np.stack(trajectories),
            len(active_states) * len(final_states),
        )
        predicted = _nearest_centers(features, centers)
        clusters[name] = sorted(set(predicted.tolist()))
        combination_counts[name] = int(len(predicted))
    singleton_clusters = [values[0] for values in clusters.values() if len(values) == 1]
    return {
        "combination_counts": combination_counts,
        "clusters_by_outcome": clusters,
        "one_to_one_partition_passed": bool(
            len(singleton_clusters) == len(terminal_groups)
            and len(set(singleton_clusters)) == len(terminal_groups)
        ),
    }


def build_lookup(
    dataset_path: Path,
    embedding_path: Path,
    cluster_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    data = np.load(dataset_path)
    embeddings = np.load(embedding_path)
    clusters = np.load(cluster_path)
    outcomes = data["outcomes"].astype(np.int64)
    lengths = data["lengths"].astype(np.int64)
    states = data["states"].astype(np.int64)
    train_indices = np.flatnonzero(data["split"] == 0)
    frame_embeddings = embeddings["frame_embeddings"].reshape(
        len(outcomes),
        3,
        -1,
    )
    direct_trajectories = embeddings["trajectory_embeddings"]
    centers = clusters["dinov2_small_centers"].astype(np.float32)

    embedding_size = frame_embeddings.shape[-1]
    sums = np.zeros((3, 16, embedding_size), dtype=np.float64)
    counts = np.zeros((3, 16), dtype=np.int64)
    for index in train_indices:
        keys = trajectory_frame_keys(states[index], int(lengths[index]))
        for frame_index, (frame_type, state) in enumerate(keys):
            sums[frame_type, state] += frame_embeddings[index, frame_index]
            counts[frame_type, state] += 1
    lookup = np.zeros_like(sums, dtype=np.float32)
    populated = counts > 0
    lookup[populated] = (sums[populated] / counts[populated][:, None]).astype(
        np.float32
    )

    reconstructed_frames = []
    for index in train_indices:
        keys = trajectory_frame_keys(states[index], int(lengths[index]))
        reconstructed_frames.extend(lookup[frame_type, state] for frame_type, state in keys)
    reconstructed_trajectories = compose_trajectory_embeddings(
        np.stack(reconstructed_frames),
        len(train_indices),
    ).astype(np.float32)
    direct_train = direct_trajectories[train_indices]
    direct_clusters = _nearest_centers(direct_train, centers)
    reconstructed_clusters = _nearest_centers(reconstructed_trajectories, centers)
    cosine = np.sum(direct_train * reconstructed_trajectories, axis=1)

    active_states = set(np.flatnonzero(counts[ACTIVE] > 0).tolist())
    hole_states = set(np.flatnonzero(counts[TERMINAL_HOLE] > 0).tolist())
    goal_states = set(np.flatnonzero(counts[TERMINAL_GOAL] > 0).tolist())
    expected_active = {0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14}
    expected_holes = {5, 7, 11, 12}
    expected_goals = {15}
    agreement = float(np.mean(direct_clusters == reconstructed_clusters))
    finite_key_audit = audit_finite_keys(lookup, centers)
    coverage_passed = bool(
        active_states == expected_active
        and hole_states == expected_holes
        and goal_states == expected_goals
    )
    cache_path = output_dir / "visual_cluster_lookup.npz"
    np.savez_compressed(
        cache_path,
        frame_lookup=lookup,
        frame_counts=counts,
        cluster_centers=centers,
        frame_type_names=np.asarray(
            ["active", "terminal_hole", "terminal_goal"]
        ),
    )
    output = {
        "dataset": str(dataset_path.resolve()),
        "embeddings": str(embedding_path.resolve()),
        "clusters": str(cluster_path.resolve()),
        "cache": str(cache_path.resolve()),
        "frame_lookup_shape": list(lookup.shape),
        "cluster_centers_shape": list(centers.shape),
        "coverage": {
            "active_states": sorted(active_states),
            "terminal_hole_states": sorted(hole_states),
            "terminal_goal_states": sorted(goal_states),
            "passed": coverage_passed,
        },
        "reconstruction": {
            "trajectory_count": len(train_indices),
            "cluster_agreement": agreement,
            "cosine_mean": float(np.mean(cosine)),
            "cosine_min": float(np.min(cosine)),
        },
        "finite_key_audit": finite_key_audit,
        "lookup_gate_passed": bool(
            coverage_passed
            and agreement >= 0.99
            and finite_key_audit["one_to_one_partition_passed"]
        ),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["metrics"] = str(metrics_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("embeddings", type=Path)
    parser.add_argument("clusters", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = build_lookup(
        args.dataset,
        args.embeddings,
        args.clusters,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "metrics": output["metrics"],
                "cache": output["cache"],
                "coverage": output["coverage"],
                "reconstruction": output["reconstruction"],
                "finite_key_audit": output["finite_key_audit"],
                "lookup_gate_passed": output["lookup_gate_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
