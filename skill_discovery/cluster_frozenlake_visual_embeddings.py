"""Audit unsupervised K=3 structure in cached FrozenLake representations."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from skill_discovery.frozenlake import FROZENLAKE_OUTCOMES


def best_cluster_mapping(
    clusters: np.ndarray,
    outcomes: np.ndarray,
    cluster_count: int,
) -> tuple[list[int], float]:
    best_mapping = list(range(cluster_count))
    best_accuracy = -1.0
    for mapping in itertools.permutations(range(cluster_count)):
        predictions = np.asarray([mapping[int(cluster)] for cluster in clusters])
        accuracy = float(np.mean(predictions == outcomes))
        if accuracy > best_accuracy:
            best_mapping = list(mapping)
            best_accuracy = accuracy
    return best_mapping, best_accuracy


def _method_metrics(
    train_features: np.ndarray,
    audit_features: np.ndarray,
    train_outcomes: np.ndarray,
    audit_outcomes: np.ndarray,
    *,
    seed: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    import sklearn
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    model = KMeans(
        n_clusters=len(FROZENLAKE_OUTCOMES),
        random_state=seed,
        n_init=32,
        max_iter=500,
        algorithm="lloyd",
    )
    train_clusters = model.fit_predict(train_features)
    audit_clusters = model.predict(audit_features)
    mapping, train_aligned_accuracy = best_cluster_mapping(
        train_clusters,
        train_outcomes,
        len(FROZENLAKE_OUTCOMES),
    )
    audit_predictions = np.asarray(
        [mapping[int(cluster)] for cluster in audit_clusters],
        dtype=np.int64,
    )
    recalls = {
        name: float(
            np.mean(audit_predictions[audit_outcomes == index] == index)
        )
        for index, name in enumerate(FROZENLAKE_OUTCOMES)
    }
    train_sizes = np.bincount(
        train_clusters,
        minlength=len(FROZENLAKE_OUTCOMES),
    )
    audit_sizes = np.bincount(
        audit_clusters,
        minlength=len(FROZENLAKE_OUTCOMES),
    )
    output = {
        "sklearn_version": sklearn.__version__,
        "train_shape": list(train_features.shape),
        "audit_shape": list(audit_features.shape),
        "inertia": float(model.inertia_),
        "cluster_to_outcome": {
            str(cluster): FROZENLAKE_OUTCOMES[outcome]
            for cluster, outcome in enumerate(mapping)
        },
        "train_aligned_accuracy": train_aligned_accuracy,
        "audit_aligned_accuracy": float(
            np.mean(audit_predictions == audit_outcomes)
        ),
        "audit_recall_by_outcome": recalls,
        "audit_nmi": float(
            normalized_mutual_info_score(audit_outcomes, audit_clusters)
        ),
        "audit_adjusted_rand": float(
            adjusted_rand_score(audit_outcomes, audit_clusters)
        ),
        "train_cluster_sizes": train_sizes.tolist(),
        "audit_cluster_sizes": audit_sizes.tolist(),
        "all_clusters_nonempty": bool(
            np.all(train_sizes > 0) and np.all(audit_sizes > 0)
        ),
    }
    return output, train_clusters, audit_clusters, model.cluster_centers_


def _write_chart(path: Path, methods: dict[str, dict[str, object]]) -> None:
    names = list(methods)
    colors = ("#68757d", "#d17031", "#2875a4", "#2b895f")
    width, height = 900, 430
    left, top, chart_width, chart_height = 70, 55, 760, 290
    group_width = chart_width / len(names)
    bars = []
    labels = []
    for index, name in enumerate(names):
        values = (
            methods[name]["audit_aligned_accuracy"],
            methods[name]["audit_nmi"],
        )
        for offset, value in enumerate(values):
            x = left + index * group_width + 30 + offset * 38
            bar_height = chart_height * float(value)
            y = top + chart_height - bar_height
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="30" '
                f'height="{bar_height:.1f}" fill="{colors[index]}" '
                f'opacity="{1.0 if offset == 0 else 0.45}"/>'
            )
        labels.append(
            f'<text x="{left + index * group_width + 10:.1f}" '
            f'y="{top + chart_height + 24}" font-family="sans-serif" '
            f'font-size="11">{name.replace("_", " ")}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fbfcfd"/>
<text x="30" y="30" font-family="sans-serif" font-size="19" fill="#172b3a">FrozenLake unsupervised K=3 audit</text>
<text x="610" y="30" font-family="sans-serif" font-size="11">solid: aligned accuracy | light: NMI</text>
<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#83919a"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#83919a"/>
<text x="28" y="{top + 5}" font-family="sans-serif" font-size="12">1.0</text>
<text x="28" y="{top + chart_height + 5}" font-family="sans-serif" font-size="12">0.0</text>
{''.join(bars)}
{''.join(labels)}
</svg>'''
    path.write_text(svg, encoding="utf-8")


def cluster_embeddings(
    dataset_path: Path,
    dinov2_path: Path,
    output_dir: Path,
    *,
    seed: int = 7,
) -> dict[str, object]:
    from skill_discovery.evaluate_frozenlake_visual_metrics import (
        build_representations,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    data = np.load(dataset_path)
    outcomes = data["outcomes"].astype(np.int64)
    split = data["split"].astype(np.int64)
    train_indices = np.flatnonzero(split == 0)
    audit_indices = np.flatnonzero(split == 1)
    base = build_representations(
        data,
        projection_seed=seed,
        audit_nuisance=True,
    )
    dinov2 = np.load(dinov2_path)["trajectory_embeddings"].astype(np.float32)
    features = {
        "raw_pixels": base["raw_pixels"],
        "random_projection": base["random_pixel_projection"],
        "dinov2_small": dinov2,
        "semantic_oracle": base["semantic_oracle"],
    }
    methods = {}
    assignments = {}
    for name, representation in features.items():
        metrics, train_clusters, audit_clusters, centers = _method_metrics(
            representation[train_indices],
            representation[audit_indices],
            outcomes[train_indices],
            outcomes[audit_indices],
            seed=seed,
        )
        methods[name] = metrics
        assignments[f"{name}_train"] = train_clusters
        assignments[f"{name}_audit"] = audit_clusters
        assignments[f"{name}_centers"] = centers
    raw_accuracy = methods["raw_pixels"]["audit_aligned_accuracy"]
    dinov2_accuracy = methods["dinov2_small"]["audit_aligned_accuracy"]
    output = {
        "dataset": str(dataset_path.resolve()),
        "dinov2_embeddings": str(dinov2_path.resolve()),
        "seed": seed,
        "cluster_count": len(FROZENLAKE_OUTCOMES),
        "label_usage": "outcome labels used only after KMeans for alignment/evaluation",
        "methods": methods,
        "gate": {
            "minimum_dinov2_accuracy": 0.70,
            "minimum_improvement_over_raw": 0.15,
        },
        "cluster_gate_passed": bool(
            dinov2_accuracy >= 0.70
            and dinov2_accuracy >= raw_accuracy + 0.15
            and methods["dinov2_small"]["all_clusters_nonempty"]
        ),
    }
    np.savez_compressed(output_dir / "cluster_assignments.npz", **assignments)
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
    parser.add_argument("dinov2_embeddings", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = cluster_embeddings(
        args.dataset,
        args.dinov2_embeddings,
        args.output_dir,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "metrics": output["metrics"],
                "chart": output["chart"],
                "summary": {
                    name: {
                        "audit_accuracy": row["audit_aligned_accuracy"],
                        "audit_nmi": row["audit_nmi"],
                        "audit_adjusted_rand": row["audit_adjusted_rand"],
                        "cluster_sizes": row["audit_cluster_sizes"],
                    }
                    for name, row in output["methods"].items()
                },
                "cluster_gate_passed": output["cluster_gate_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
