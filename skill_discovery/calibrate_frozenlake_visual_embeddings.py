"""Audit trajectory self-reference against FrozenLake visual style shift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from skill_discovery.cluster_frozenlake_visual_embeddings import _method_metrics
from skill_discovery.frozenlake import FROZENLAKE_OUTCOMES


def _normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.maximum(norms, 1.0e-8)


def build_self_reference_representations(
    frame_embeddings: np.ndarray,
) -> dict[str, np.ndarray]:
    if frame_embeddings.ndim == 2:
        if frame_embeddings.shape[0] % 3:
            raise ValueError("frame count must be divisible by three")
        frame_embeddings = frame_embeddings.reshape(-1, 3, frame_embeddings.shape[1])
    if frame_embeddings.ndim != 3 or frame_embeddings.shape[1] != 3:
        raise ValueError("expected trajectory x three frames x embedding")

    frames = _normalize(frame_embeddings.astype(np.float32))
    absolute = _normalize(frames.reshape(len(frames), -1))
    middle_final = _normalize(frames[:, 1:].reshape(len(frames), -1))
    deltas = _normalize(frames[:, 1:] - frames[:, :1])
    temporal_delta = _normalize(deltas.reshape(len(frames), -1))
    return {
        "absolute_3frame": absolute.astype(np.float32),
        "middle_final": middle_final.astype(np.float32),
        "temporal_delta": temporal_delta.astype(np.float32),
    }


def _aligned_predictions(
    clusters: np.ndarray,
    mapping: dict[str, str],
) -> np.ndarray:
    outcome_indices = {name: index for index, name in enumerate(FROZENLAKE_OUTCOMES)}
    cluster_mapping = np.asarray(
        [outcome_indices[mapping[str(cluster)]] for cluster in range(len(mapping))]
    )
    return cluster_mapping[clusters]


def _seed_audit(
    audit_clusters: np.ndarray,
    audit_indices: np.ndarray,
    outcomes: np.ndarray,
    generation_seeds: np.ndarray,
    mapping: dict[str, str],
) -> dict[str, object]:
    predictions = _aligned_predictions(audit_clusters, mapping)
    output = {}
    for seed in sorted(set(generation_seeds[audit_indices].tolist())):
        local = np.flatnonzero(generation_seeds[audit_indices] == seed)
        confusion = np.zeros((3, 3), dtype=np.int64)
        for expected, predicted in zip(
            outcomes[audit_indices[local]],
            predictions[local],
        ):
            confusion[int(expected), int(predicted)] += 1
        output[str(seed)] = {
            "accuracy": float(np.trace(confusion) / confusion.sum()),
            "recall_by_outcome": {
                name: float(confusion[index, index] / confusion[index].sum())
                for index, name in enumerate(FROZENLAKE_OUTCOMES)
            },
            "confusion": confusion.tolist(),
        }
    return output


def _write_chart(path: Path, methods: dict[str, dict[str, object]]) -> None:
    width, height = 820, 420
    left, top, chart_width, chart_height = 70, 55, 680, 290
    group_width = chart_width / len(methods)
    colors = ("#68757d", "#2875a4", "#2b895f")
    elements = []
    for index, (name, metrics) in enumerate(methods.items()):
        values = (
            metrics["audit_aligned_accuracy"],
            metrics["audit_recall_by_outcome"]["safe_timeout"],
        )
        for offset, value in enumerate(values):
            x = left + index * group_width + 45 + offset * 42
            bar_height = chart_height * float(value)
            elements.append(
                f'<rect x="{x:.1f}" y="{top + chart_height - bar_height:.1f}" '
                f'width="32" height="{bar_height:.1f}" fill="{colors[index]}" '
                f'opacity="{1.0 if offset == 0 else 0.45}"/>'
            )
        elements.append(
            f'<text x="{left + index * group_width + 22:.1f}" '
            f'y="{top + chart_height + 24}" font-family="sans-serif" '
            f'font-size="11">{name.replace("_", " ")}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fbfcfd"/>
<text x="30" y="30" font-family="sans-serif" font-size="19" fill="#172b3a">FrozenLake trajectory self-reference audit</text>
<text x="555" y="30" font-family="sans-serif" font-size="11">solid: accuracy | light: safe recall</text>
<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#83919a"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#83919a"/>
<text x="28" y="{top + 5}" font-family="sans-serif" font-size="12">1.0</text>
<text x="28" y="{top + chart_height + 5}" font-family="sans-serif" font-size="12">0.0</text>
{''.join(elements)}
</svg>'''
    path.write_text(svg, encoding="utf-8")


def calibrate_representations(
    dataset_path: Path,
    embedding_path: Path,
    output_dir: Path,
    *,
    seed: int = 7,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    data = np.load(dataset_path)
    embeddings = np.load(embedding_path)
    outcomes = data["outcomes"].astype(np.int64)
    split = data["split"].astype(np.int64)
    generation_seeds = data["generation_seeds"].astype(np.int64)
    train_indices = np.flatnonzero(split == 0)
    audit_indices = np.flatnonzero(split == 1)
    representations = build_self_reference_representations(
        embeddings["frame_embeddings"]
    )
    cached_absolute = embeddings["trajectory_embeddings"].astype(np.float32)
    baseline_max_difference = float(
        np.max(np.abs(representations["absolute_3frame"] - cached_absolute))
    )

    methods = {}
    assignments = {}
    for name, features in representations.items():
        metrics, train_clusters, audit_clusters, centers = _method_metrics(
            features[train_indices],
            features[audit_indices],
            outcomes[train_indices],
            outcomes[audit_indices],
            seed=seed,
        )
        recalls = metrics["audit_recall_by_outcome"]
        metrics["seed_audit"] = _seed_audit(
            audit_clusters,
            audit_indices,
            outcomes,
            generation_seeds,
            metrics["cluster_to_outcome"],
        )
        metrics["calibration_gate_passed"] = bool(
            name != "absolute_3frame"
            and metrics["audit_aligned_accuracy"] >= 0.84
            and recalls["safe_timeout"] >= 0.50
            and recalls["hole_terminal"] >= 0.95
            and recalls["goal_terminal"] >= 0.95
            and metrics["all_clusters_nonempty"]
        )
        methods[name] = metrics
        assignments[f"{name}_train"] = train_clusters
        assignments[f"{name}_audit"] = audit_clusters
        assignments[f"{name}_centers"] = centers

    candidate_passes = {
        name: bool(metrics["calibration_gate_passed"])
        for name, metrics in methods.items()
        if name != "absolute_3frame"
    }
    output = {
        "dataset": str(dataset_path.resolve()),
        "embeddings": str(embedding_path.resolve()),
        "seed": seed,
        "label_usage": "outcome labels used only after KMeans for alignment/evaluation",
        "baseline_cache_max_abs_difference": baseline_max_difference,
        "gate": {
            "minimum_audit_accuracy": 0.84,
            "minimum_safe_recall": 0.50,
            "minimum_hole_recall": 0.95,
            "minimum_goal_recall": 0.95,
            "require_all_clusters_nonempty": True,
        },
        "methods": methods,
        "candidate_passes": candidate_passes,
        "calibration_gate_passed": bool(any(candidate_passes.values())),
    }
    np.savez_compressed(output_dir / "calibrated_clusters.npz", **assignments)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    chart_path = output_dir / "self_reference_comparison.svg"
    _write_chart(chart_path, methods)
    output["metrics"] = str(metrics_path.resolve())
    output["chart"] = str(chart_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("embeddings", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = calibrate_representations(
        args.dataset,
        args.embeddings,
        args.output_dir,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "metrics": output["metrics"],
                "chart": output["chart"],
                "methods": {
                    name: {
                        "accuracy": row["audit_aligned_accuracy"],
                        "recall": row["audit_recall_by_outcome"],
                        "passed": row["calibration_gate_passed"],
                    }
                    for name, row in output["methods"].items()
                },
                "calibration_gate_passed": output["calibration_gate_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
