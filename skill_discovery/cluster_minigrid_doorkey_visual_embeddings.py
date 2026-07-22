"""Audit unsupervised K=4 structure in DoorKey visual representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from skill_discovery.cluster_frozenlake_visual_embeddings import (
    best_cluster_mapping,
)
from skill_discovery.minigrid_doorkey import DOORKEY_STAGES


DINOV2_SELECTION_PRIORITY = (
    "dinov2_temporal_delta",
    "dinov2_start_current",
    "dinov2_current",
)


def _normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1.0e-8)


def build_raw_representations(frames: np.ndarray) -> dict[str, np.ndarray]:
    if frames.ndim != 5 or frames.shape[1] != 4 or frames.shape[-1] != 3:
        raise ValueError("expected trajectory x four RGB stage frames")
    y = np.linspace(0, frames.shape[2] - 1, 32).astype(np.int64)
    x = np.linspace(0, frames.shape[3] - 1, 32).astype(np.int64)
    resized = frames[:, :, y][:, :, :, x].astype(np.float32) / 255.0
    current = resized.reshape(-1, 32 * 32 * 3)
    start = np.repeat(resized[:, :1], 4, axis=1).reshape(-1, 32 * 32 * 3)
    return {
        "raw_current": _normalize(current).astype(np.float32),
        "raw_temporal_delta": _normalize(current - start).astype(np.float32),
    }


def _cluster_method(
    train_features: np.ndarray,
    audit_features: np.ndarray,
    train_stages: np.ndarray,
    audit_stages: np.ndarray,
    *,
    seed: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    import sklearn
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    model = KMeans(
        n_clusters=len(DOORKEY_STAGES),
        random_state=seed,
        n_init=32,
        max_iter=500,
        algorithm="lloyd",
    )
    train_clusters = model.fit_predict(train_features)
    audit_clusters = model.predict(audit_features)
    mapping, train_accuracy = best_cluster_mapping(
        train_clusters,
        train_stages,
        len(DOORKEY_STAGES),
    )
    predictions = np.asarray(
        [mapping[int(cluster)] for cluster in audit_clusters],
        dtype=np.int64,
    )
    confusion = np.zeros((4, 4), dtype=np.int64)
    for expected, predicted in zip(audit_stages, predictions):
        confusion[int(expected), int(predicted)] += 1
    recalls = {
        stage: float(confusion[index, index] / confusion[index].sum())
        for index, stage in enumerate(DOORKEY_STAGES)
    }
    train_sizes = np.bincount(train_clusters, minlength=4)
    audit_sizes = np.bincount(audit_clusters, minlength=4)
    output = {
        "sklearn_version": sklearn.__version__,
        "train_shape": list(train_features.shape),
        "audit_shape": list(audit_features.shape),
        "inertia": float(model.inertia_),
        "cluster_to_stage": {
            str(cluster): DOORKEY_STAGES[stage]
            for cluster, stage in enumerate(mapping)
        },
        "train_aligned_accuracy": train_accuracy,
        "audit_aligned_accuracy": float(np.mean(predictions == audit_stages)),
        "audit_recall_by_stage": recalls,
        "audit_nmi": float(
            normalized_mutual_info_score(audit_stages, audit_clusters)
        ),
        "audit_adjusted_rand": float(
            adjusted_rand_score(audit_stages, audit_clusters)
        ),
        "audit_confusion": confusion.tolist(),
        "train_cluster_sizes": train_sizes.tolist(),
        "audit_cluster_sizes": audit_sizes.tolist(),
        "all_clusters_nonempty": bool(
            np.all(train_sizes > 0) and np.all(audit_sizes > 0)
        ),
    }
    return output, train_clusters, audit_clusters, model.cluster_centers_


def _group_audit(
    clusters: np.ndarray,
    stages: np.ndarray,
    groups: np.ndarray,
    cluster_to_stage: dict[str, str],
) -> dict[str, object]:
    stage_indices = {name: index for index, name in enumerate(DOORKEY_STAGES)}
    predictions = np.asarray(
        [stage_indices[cluster_to_stage[str(int(cluster))]] for cluster in clusters],
        dtype=np.int64,
    )
    output = {}
    for group in sorted(set(groups.tolist())):
        local = groups == group
        local_stages = stages[local]
        local_predictions = predictions[local]
        output[str(group)] = {
            "accuracy": float(np.mean(local_predictions == local_stages)),
            "recall_by_stage": {
                name: float(
                    np.mean(
                        local_predictions[local_stages == index] == index
                    )
                )
                for index, name in enumerate(DOORKEY_STAGES)
            },
        }
    return output


def _write_chart(path: Path, methods: dict[str, dict[str, object]]) -> None:
    names = list(methods)
    colors = ("#68757d", "#d17031", "#2875a4", "#2b895f", "#955f9a")
    width, height = 980, 430
    left, top, chart_width, chart_height = 70, 55, 840, 290
    group_width = chart_width / len(names)
    elements = []
    for index, name in enumerate(names):
        values = (
            methods[name]["audit_aligned_accuracy"],
            methods[name]["audit_nmi"],
        )
        for offset, value in enumerate(values):
            x = left + index * group_width + 22 + offset * 34
            bar_height = chart_height * float(value)
            elements.append(
                f'<rect x="{x:.1f}" y="{top + chart_height - bar_height:.1f}" '
                f'width="27" height="{bar_height:.1f}" fill="{colors[index]}" '
                f'opacity="{1.0 if offset == 0 else 0.45}"/>'
            )
        label = name.replace("dinov2_", "dino ").replace("_", " ")
        elements.append(
            f'<text x="{left + index * group_width + 4:.1f}" '
            f'y="{top + chart_height + 24}" font-family="sans-serif" '
            f'font-size="10">{label}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#fbfcfd"/>'
        '<text x="30" y="30" font-family="sans-serif" font-size="19" '
        'fill="#172b3a">DoorKey unsupervised visual K=4 audit</text>'
        '<text x="705" y="30" font-family="sans-serif" font-size="11">'
        'solid: accuracy | light: NMI</text>'
        f'<line x1="{left}" y1="{top + chart_height}" '
        f'x2="{left + chart_width}" y2="{top + chart_height}" '
        'stroke="#83919a"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{top + chart_height}" stroke="#83919a"/>'
        f'{"".join(elements)}</svg>'
    )
    path.write_text(svg, encoding="utf-8")


def cluster_embeddings(
    dataset_path: Path,
    embedding_path: Path,
    output_dir: Path,
    *,
    seed: int = 7,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(dataset_path) as data:
        frames = data["frames"].astype(np.uint8)
        stage_labels = data["stage_labels"].astype(np.int64).reshape(-1)
        trajectory_split = data["split"].astype(np.int8)
        generation_groups = data["generation_groups"].astype(np.int32)
    split = np.repeat(trajectory_split, 4)
    frame_generation_groups = np.repeat(generation_groups, 4)
    train_indices = np.flatnonzero(split == 0)
    audit_indices = np.flatnonzero(split == 1)
    with np.load(embedding_path) as embeddings:
        features = {
            **build_raw_representations(frames),
            **{
                name: embeddings[name].astype(np.float32)
                for name in DINOV2_SELECTION_PRIORITY
            },
        }
    methods = {}
    assignments = {}
    for name, representation in features.items():
        metrics, train_clusters, audit_clusters, centers = _cluster_method(
            representation[train_indices],
            representation[audit_indices],
            stage_labels[train_indices],
            stage_labels[audit_indices],
            seed=seed,
        )
        metrics["foundation_visual_gate_passed"] = bool(
            name.startswith("dinov2_")
            and metrics["audit_aligned_accuracy"] >= 0.85
            and min(metrics["audit_recall_by_stage"].values()) >= 0.75
            and metrics["audit_nmi"] >= 0.65
            and metrics["all_clusters_nonempty"]
        )
        metrics["audit_by_generation_group"] = _group_audit(
            audit_clusters,
            stage_labels[audit_indices],
            frame_generation_groups[audit_indices],
            metrics["cluster_to_stage"],
        )
        methods[name] = metrics
        assignments[f"{name}_train"] = train_clusters
        assignments[f"{name}_audit"] = audit_clusters
        assignments[f"{name}_centers"] = centers
    selected = next(
        (
            name
            for name in DINOV2_SELECTION_PRIORITY
            if methods[name]["foundation_visual_gate_passed"]
        ),
        None,
    )
    train_groups = sorted(set(generation_groups[trajectory_split == 0].tolist()))
    audit_groups = sorted(set(generation_groups[trajectory_split == 1].tolist()))
    output = {
        "dataset": str(dataset_path.resolve()),
        "embeddings": str(embedding_path.resolve()),
        "seed": seed,
        "cluster_count": len(DOORKEY_STAGES),
        "stage_names": DOORKEY_STAGES,
        "train_generation_groups": train_groups,
        "audit_generation_groups": audit_groups,
        "split_groups_disjoint": not bool(set(train_groups) & set(audit_groups)),
        "label_usage": (
            "train labels used only after KMeans for permutation; "
            "audit labels used only for final metrics"
        ),
        "selection_priority": DINOV2_SELECTION_PRIORITY,
        "gate": {
            "minimum_audit_accuracy": 0.85,
            "minimum_recall_per_stage": 0.75,
            "minimum_nmi": 0.65,
            "require_all_clusters_nonempty": True,
        },
        "methods": methods,
        "selected_representation": selected,
        "foundation_visual_gate_passed": selected is not None,
    }
    np.savez_compressed(
        output_dir / "cluster_assignments.npz",
        **assignments,
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    chart_path = output_dir / "cluster_comparison.svg"
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
    output = cluster_embeddings(
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
                        "recall": row["audit_recall_by_stage"],
                        "nmi": row["audit_nmi"],
                        "passed": row["foundation_visual_gate_passed"],
                    }
                    for name, row in output["methods"].items()
                },
                "selected_representation": output["selected_representation"],
                "foundation_visual_gate_passed": output[
                    "foundation_visual_gate_passed"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
