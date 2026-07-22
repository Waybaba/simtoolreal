"""Evaluate cached FrozenLake trajectory representations across seed splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from skill_discovery.evaluate_metrics import _farthest_point_sample
from skill_discovery.frozenlake import FROZENLAKE_OUTCOMES
from skill_discovery.generate_point_cup_dataset import _write_png


def _resize_frames(frames: np.ndarray, size: int) -> np.ndarray:
    y = np.linspace(0, frames.shape[2] - 1, size).astype(np.int64)
    x = np.linspace(0, frames.shape[3] - 1, size).astype(np.int64)
    return frames[:, :, y][:, :, :, x]


def apply_audit_nuisance(
    frames: np.ndarray,
    generation_seeds: np.ndarray,
    split: np.ndarray,
) -> np.ndarray:
    transformed = frames.copy()
    for generation_seed in np.unique(generation_seeds[split == 1]):
        rng = np.random.default_rng(int(generation_seed) + 500_000)
        scales = rng.uniform(0.72, 1.18, size=3)
        offsets = rng.uniform(-10.0, 14.0, size=3)
        dx = int(rng.choice((-3, -2, 2, 3)))
        dy = int(rng.choice((-3, -2, 2, 3)))
        indices = np.flatnonzero((split == 1) & (generation_seeds == generation_seed))
        styled = np.clip(
            transformed[indices].astype(np.float32) * scales + offsets,
            0,
            255,
        ).astype(np.uint8)
        shifted = np.full_like(styled, 245)
        source_y = slice(max(0, -dy), styled.shape[2] - max(0, dy))
        target_y = slice(max(0, dy), styled.shape[2] - max(0, -dy))
        source_x = slice(max(0, -dx), styled.shape[3] - max(0, dx))
        target_x = slice(max(0, dx), styled.shape[3] - max(0, -dx))
        shifted[:, :, target_y, target_x] = styled[:, :, source_y, source_x]
        transformed[indices] = shifted
    return transformed


def _write_nuisance_sample(
    path: Path,
    frames: np.ndarray,
    outcomes: np.ndarray,
    audit_indices: np.ndarray,
) -> None:
    frame_size = 80
    trajectories_per_outcome = 4
    triptych_width = 3 * frame_size
    gap = 5
    marker_width = 9
    colors = ((48, 116, 173), (198, 72, 58), (42, 137, 94))
    sheet = np.full(
        (
            3 * frame_size + 2 * gap,
            marker_width
            + trajectories_per_outcome * triptych_width
            + (trajectories_per_outcome - 1) * gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    rng = np.random.default_rng(707)
    for outcome in range(3):
        candidates = audit_indices[outcomes[audit_indices] == outcome]
        chosen = rng.choice(candidates, size=trajectories_per_outcome, replace=False)
        y = outcome * (frame_size + gap)
        sheet[y : y + frame_size, :marker_width] = colors[outcome]
        for column, trajectory_index in enumerate(chosen):
            x = marker_width + column * (triptych_width + gap)
            for frame_index in range(3):
                frame = frames[int(trajectory_index), frame_index]
                rows = np.linspace(0, frame.shape[0] - 1, frame_size).astype(np.int64)
                cols = np.linspace(0, frame.shape[1] - 1, frame_size).astype(np.int64)
                frame_x = x + frame_index * frame_size
                sheet[
                    y : y + frame_size,
                    frame_x : frame_x + frame_size,
                ] = frame[rows][:, cols]
    _write_png(path, sheet)


def build_representations(
    data: np.lib.npyio.NpzFile,
    *,
    projection_seed: int,
    audit_nuisance: bool = False,
) -> dict[str, np.ndarray]:
    outcomes = data["outcomes"].astype(np.int64)
    lengths = data["lengths"].astype(np.int64)
    states = data["states"].astype(np.int64)
    terminal_states = states[np.arange(len(states)), lengths]
    terminal_onehot = np.eye(16, dtype=np.float32)[terminal_states]

    frames = data["frames"]
    if audit_nuisance:
        frames = apply_audit_nuisance(
            frames,
            data["generation_seeds"],
            data["split"],
        )
    small_frames = _resize_frames(frames, 8).astype(np.float32) / 255.0
    raw_pixels = small_frames.reshape(len(small_frames), -1)
    rng = np.random.default_rng(projection_seed)
    projection = rng.normal(
        0.0,
        1.0 / np.sqrt(raw_pixels.shape[1]),
        size=(raw_pixels.shape[1], 64),
    ).astype(np.float32)
    random_projection = raw_pixels @ projection
    semantic_oracle = np.eye(len(FROZENLAKE_OUTCOMES), dtype=np.float32)[outcomes]
    return {
        "raw_terminal_state": terminal_onehot,
        "raw_pixels": raw_pixels,
        "random_pixel_projection": random_projection,
        "semantic_oracle": semantic_oracle,
    }


def _cross_seed_metrics(
    train_features: np.ndarray,
    train_outcomes: np.ndarray,
    audit_features: np.ndarray,
    audit_outcomes: np.ndarray,
) -> dict[str, object]:
    predictions = np.empty(len(audit_features), dtype=np.int64)
    nearest_same = np.empty(len(audit_features), dtype=np.float64)
    nearest_different = np.empty(len(audit_features), dtype=np.float64)
    chunk_size = 128
    train_norm = np.sum(train_features * train_features, axis=1)
    for start in range(0, len(audit_features), chunk_size):
        stop = min(start + chunk_size, len(audit_features))
        query = audit_features[start:stop]
        distances = (
            np.sum(query * query, axis=1)[:, None]
            + train_norm[None, :]
            - 2.0 * query @ train_features.T
        )
        distances = np.maximum(distances, 0.0)
        predictions[start:stop] = train_outcomes[np.argmin(distances, axis=1)]
        query_outcomes = audit_outcomes[start:stop]
        same_mask = query_outcomes[:, None] == train_outcomes[None, :]
        same_distances = np.where(same_mask, distances, np.inf)
        different_distances = np.where(~same_mask, distances, np.inf)
        nearest_same[start:stop] = np.sqrt(np.min(same_distances, axis=1))
        nearest_different[start:stop] = np.sqrt(
            np.min(different_distances, axis=1)
        )
    recalls = {
        name: float(np.mean(predictions[audit_outcomes == index] == index))
        for index, name in enumerate(FROZENLAKE_OUTCOMES)
    }
    margins = nearest_different - nearest_same
    return {
        "knn_accuracy": float(np.mean(predictions == audit_outcomes)),
        "knn_recall_by_outcome": recalls,
        "nuisance_triplet_accuracy": float(np.mean(margins > 0)),
        "nuisance_margin_mean": float(np.mean(margins)),
        "nearest_same_distance_mean": float(np.mean(nearest_same)),
        "nearest_different_distance_mean": float(np.mean(nearest_different)),
    }


def _representative_metrics(
    features: np.ndarray,
    outcomes: np.ndarray,
    train_indices: np.ndarray,
    *,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    desired_counts = (12, 384, 12)
    candidate_parts = []
    for outcome, desired in enumerate(desired_counts):
        available = train_indices[outcomes[train_indices] == outcome]
        count = min(desired, len(available))
        candidate_parts.append(rng.choice(available, size=count, replace=False))
    candidate_indices = np.concatenate(candidate_parts)
    selected_local = _farthest_point_sample(features[candidate_indices], 12)
    selected_indices = candidate_indices[selected_local]
    selected_outcomes = outcomes[selected_indices]
    counts = np.bincount(
        selected_outcomes,
        minlength=len(FROZENLAKE_OUTCOMES),
    )
    rare_recall = float(
        np.mean([np.any(selected_outcomes == outcome) for outcome in (0, 2)])
    )
    return {
        "candidate_counts": {
            name: int(np.count_nonzero(outcomes[candidate_indices] == index))
            for index, name in enumerate(FROZENLAKE_OUTCOMES)
        },
        "selected_counts": {
            name: int(counts[index])
            for index, name in enumerate(FROZENLAKE_OUTCOMES)
        },
        "class_coverage": float(np.count_nonzero(counts) / len(counts)),
        "rare_safe_goal_recall": rare_recall,
        "selected_indices": selected_indices.tolist(),
    }


def _write_chart(path: Path, metrics: dict[str, dict[str, object]]) -> None:
    width, height = 900, 430
    left, top, chart_width, chart_height = 70, 60, 760, 290
    names = list(metrics)
    colors = ("#68757d", "#2875a4", "#d17031", "#2b895f")
    group_width = chart_width / len(names)
    bars = []
    labels = []
    for index, name in enumerate(names):
        values = (
            metrics[name]["cross_seed"]["knn_accuracy"],
            metrics[name]["cross_seed"]["nuisance_triplet_accuracy"],
        )
        for offset, value in enumerate(values):
            x = left + index * group_width + 20 + offset * 38
            bar_height = chart_height * float(value)
            y = top + chart_height - bar_height
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="30" '
                f'height="{bar_height:.1f}" fill="{colors[index]}" '
                f'opacity="{1.0 if offset == 0 else 0.45}"/>'
            )
        labels.append(
            f'<text x="{left + index * group_width + 12:.1f}" '
            f'y="{top + chart_height + 24}" font-family="sans-serif" '
            f'font-size="11">{name.replace("_", " ")}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fbfcfd"/>
<text x="30" y="31" font-family="sans-serif" font-size="19" fill="#172b3a">FrozenLake cached visual metric probe</text>
<text x="590" y="31" font-family="sans-serif" font-size="11">solid: cross-seed 1-NN | light: nuisance triplet</text>
<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#83919a"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#83919a"/>
<text x="28" y="{top + 5}" font-family="sans-serif" font-size="12">1.0</text>
<text x="28" y="{top + chart_height + 5}" font-family="sans-serif" font-size="12">0.0</text>
{''.join(bars)}
{''.join(labels)}
</svg>'''
    path.write_text(svg, encoding="utf-8")


def evaluate_dataset(
    dataset_path: Path,
    output_dir: Path,
    *,
    projection_seed: int = 7,
    audit_nuisance: bool = False,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    data = np.load(dataset_path)
    outcomes = data["outcomes"].astype(np.int64)
    split = data["split"].astype(np.int64)
    train_indices = np.flatnonzero(split == 0)
    audit_indices = np.flatnonzero(split == 1)
    representations = build_representations(
        data,
        projection_seed=projection_seed,
        audit_nuisance=audit_nuisance,
    )
    nuisance_image = None
    if audit_nuisance:
        transformed_frames = apply_audit_nuisance(
            data["frames"],
            data["generation_seeds"],
            split,
        )
        nuisance_image = output_dir / "audit_nuisance_sample.png"
        _write_nuisance_sample(
            nuisance_image,
            transformed_frames,
            outcomes,
            audit_indices,
        )
    metrics = {}
    for name, features in representations.items():
        metrics[name] = {
            "feature_shape": list(features.shape),
            "cross_seed": _cross_seed_metrics(
                features[train_indices],
                outcomes[train_indices],
                features[audit_indices],
                outcomes[audit_indices],
            ),
            "representatives": _representative_metrics(
                features,
                outcomes,
                train_indices,
                seed=projection_seed + 10_000,
            ),
        }
    output = {
        "dataset": str(dataset_path.resolve()),
        "projection_seed": projection_seed,
        "audit_nuisance": audit_nuisance,
        "train_count": len(train_indices),
        "audit_count": len(audit_indices),
        "metrics": metrics,
        "oracle_gate_passed": bool(
            metrics["semantic_oracle"]["cross_seed"]["knn_accuracy"] == 1.0
            and metrics["semantic_oracle"]["cross_seed"][
                "nuisance_triplet_accuracy"
            ]
            == 1.0
            and metrics["semantic_oracle"]["representatives"][
                "rare_safe_goal_recall"
            ]
            == 1.0
        ),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    chart_path = output_dir / "metric_comparison.svg"
    _write_chart(chart_path, metrics)
    output["metrics_path"] = str(metrics_path.resolve())
    output["chart"] = str(chart_path.resolve())
    output["nuisance_image"] = (
        str(nuisance_image.resolve()) if nuisance_image is not None else None
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--projection-seed", type=int, default=7)
    parser.add_argument("--audit-nuisance", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = evaluate_dataset(
        args.dataset,
        args.output_dir,
        projection_seed=args.projection_seed,
        audit_nuisance=args.audit_nuisance,
    )
    print(
        json.dumps(
            {
                "metrics": output["metrics_path"],
                "chart": output["chart"],
                "oracle_gate_passed": output["oracle_gate_passed"],
                "nuisance_image": output["nuisance_image"],
                "summary": {
                    name: {
                        "knn": row["cross_seed"]["knn_accuracy"],
                        "triplet": row["cross_seed"][
                            "nuisance_triplet_accuracy"
                        ],
                        "rare_recall": row["representatives"][
                            "rare_safe_goal_recall"
                        ],
                    }
                    for name, row in output["metrics"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
