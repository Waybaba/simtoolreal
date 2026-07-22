"""Evaluate frozen 4x4 tile-delta clusters across scale and style shifts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from skill_discovery.cluster_frozenlake_visual_embeddings import (
    best_cluster_mapping,
)
from skill_discovery.frozenlake import FROZENLAKE_OUTCOMES
from skill_discovery.transfer_frozenlake_visual_embeddings import (
    classification_metrics,
)


def _nearest_centers(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = (
        np.sum(features * features, axis=1)[:, None]
        + np.sum(centers * centers, axis=1)[None, :]
        - 2.0 * features @ centers.T
    )
    return np.argmin(distances, axis=1)


def _align(clusters: np.ndarray, mapping: list[int]) -> np.ndarray:
    return np.asarray(mapping, dtype=np.int64)[clusters]


def style_transfer_metrics(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    generation_seeds: np.ndarray,
) -> dict[str, object]:
    base = classification_metrics(predictions, outcomes, generation_seeds)
    style_accuracies = {
        seed: row["accuracy"] for seed, row in base["per_seed"].items()
    }
    passing = sum(value >= 0.80 for value in style_accuracies.values())
    worst_seed = min(style_accuracies, key=style_accuracies.get)
    recalls = base["recall_by_outcome"]
    base.update(
        {
            "styles_passing_0_80": passing,
            "style_count": len(style_accuracies),
            "worst_style_seed": int(worst_seed),
            "worst_style_accuracy": style_accuracies[worst_seed],
            "gate_passed": bool(
                base["accuracy"] >= 0.85
                and recalls["safe_timeout"] >= 0.85
                and recalls["hole_terminal"] >= 0.85
                and recalls["goal_terminal"] >= 0.70
                and passing >= 12
            ),
        }
    )
    return base


def _four_by_four_metrics(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    seeds: np.ndarray,
    clusters: np.ndarray,
) -> dict[str, object]:
    output = classification_metrics(predictions, outcomes, seeds)
    recalls = output["recall_by_outcome"]
    cluster_sizes = np.bincount(clusters, minlength=3)
    output.update(
        {
            "cluster_sizes": cluster_sizes.tolist(),
            "all_clusters_nonempty": bool(np.all(cluster_sizes > 0)),
            "gate_passed": bool(
                output["accuracy"] >= 0.90
                and recalls["safe_timeout"] >= 0.75
                and recalls["hole_terminal"] >= 0.95
                and recalls["goal_terminal"] >= 0.95
                and np.all(cluster_sizes > 0)
            ),
        }
    )
    return output


def _write_chart(
    path: Path,
    candidate: dict[str, dict[str, object]],
) -> None:
    audit_names = ("4x4 style", "8x8 scale", "8x8 style")
    baseline_accuracy = (0.9830729167, 0.5895833333, 0.7819010417)
    baseline_goal = (1.0, 0.0, 0.369140625)
    candidate_accuracy = tuple(row["accuracy"] for row in candidate.values())
    candidate_goal = tuple(
        row["recall_by_outcome"]["goal_terminal"] for row in candidate.values()
    )
    width, height = 900, 430
    left, top, chart_width, chart_height = 70, 55, 760, 290
    group_width = chart_width / 3
    elements = []
    colors = ("#68757d", "#2b895f")
    for index in range(3):
        values = (
            baseline_accuracy[index],
            baseline_goal[index],
            candidate_accuracy[index],
            candidate_goal[index],
        )
        for bar, value in enumerate(values):
            x = left + index * group_width + 32 + bar * 38
            bar_height = chart_height * float(value)
            elements.append(
                f'<rect x="{x:.1f}" y="{top + chart_height - bar_height:.1f}" '
                f'width="30" height="{bar_height:.1f}" '
                f'fill="{colors[bar // 2]}" opacity="{1.0 if bar % 2 == 0 else 0.45}"/>'
            )
        elements.append(
            f'<text x="{left + index * group_width + 42:.1f}" '
            f'y="{top + chart_height + 24}" font-family="sans-serif" '
            f'font-size="11">{audit_names[index]}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fbfcfd"/>
<text x="30" y="30" font-family="sans-serif" font-size="19" fill="#172b3a">Global CLS vs scale-aware tile delta</text>
<text x="520" y="30" font-family="sans-serif" font-size="11">gray: global | green: tile; solid: accuracy | light: goal recall</text>
<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#83919a"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#83919a"/>
<text x="28" y="{top + 5}" font-family="sans-serif" font-size="12">1.0</text>
<text x="28" y="{top + chart_height + 5}" font-family="sans-serif" font-size="12">0.0</text>
{''.join(elements)}
</svg>'''
    path.write_text(svg, encoding="utf-8")


def evaluate_tile_transfer(
    four_by_four_path: Path,
    eight_by_eight_path: Path,
    styled_eight_by_eight_path: Path,
    output_dir: Path,
    *,
    seed: int = 7,
) -> dict[str, object]:
    import sklearn
    from sklearn.cluster import KMeans

    output_dir.mkdir(parents=True, exist_ok=False)
    four = np.load(four_by_four_path)
    eight = np.load(eight_by_eight_path)
    styled = np.load(styled_eight_by_eight_path)
    four_features = four["trajectory_embeddings"].astype(np.float32)
    train_indices = np.flatnonzero(four["split"] == 0)
    audit_indices = np.flatnonzero(four["split"] == 1)
    model = KMeans(
        n_clusters=3,
        random_state=seed,
        n_init=32,
        max_iter=500,
        algorithm="lloyd",
    )
    train_clusters = model.fit_predict(four_features[train_indices])
    mapping, train_accuracy = best_cluster_mapping(
        train_clusters,
        four["outcomes"][train_indices],
        3,
    )
    audit_clusters = model.predict(four_features[audit_indices])
    eight_clusters = _nearest_centers(
        eight["trajectory_embeddings"].astype(np.float32),
        model.cluster_centers_,
    )
    styled_clusters = _nearest_centers(
        styled["trajectory_embeddings"].astype(np.float32),
        model.cluster_centers_,
    )
    methods = {
        "four_by_four_style": _four_by_four_metrics(
            _align(audit_clusters, mapping),
            four["outcomes"][audit_indices],
            four["generation_seeds"][audit_indices],
            audit_clusters,
        ),
        "eight_by_eight_scale": classification_metrics(
            _align(eight_clusters, mapping),
            eight["outcomes"],
            eight["generation_seeds"],
        ),
        "eight_by_eight_scale_style": style_transfer_metrics(
            _align(styled_clusters, mapping),
            styled["outcomes"],
            styled["generation_seeds"],
        ),
    }
    passed = bool(all(row["gate_passed"] for row in methods.values()))
    assignments_path = output_dir / "tile_transfer_assignments.npz"
    np.savez_compressed(
        assignments_path,
        centers=model.cluster_centers_,
        mapping=np.asarray(mapping),
        four_train=train_clusters,
        four_audit=audit_clusters,
        eight=eight_clusters,
        eight_style=styled_clusters,
    )
    output = {
        "inputs": {
            "four_by_four": str(four_by_four_path.resolve()),
            "eight_by_eight": str(eight_by_eight_path.resolve()),
            "styled_eight_by_eight": str(styled_eight_by_eight_path.resolve()),
        },
        "sklearn_version": sklearn.__version__,
        "seed": seed,
        "training": {
            "fit_split": "4x4 seeds 7/17/27 only",
            "cluster_sizes": np.bincount(train_clusters, minlength=3).tolist(),
            "aligned_accuracy": train_accuracy,
            "cluster_to_outcome": {
                str(cluster): FROZENLAKE_OUTCOMES[outcome]
                for cluster, outcome in enumerate(mapping)
            },
        },
        "label_usage": "labels used only after 4x4 KMeans for alignment/evaluation",
        "methods": methods,
        "all_representation_gates_passed": passed,
        "assignments": str(assignments_path.resolve()),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    chart_path = output_dir / "tile_transfer_comparison.svg"
    _write_chart(chart_path, methods)
    output["metrics"] = str(metrics_path.resolve())
    output["chart"] = str(chart_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("four_by_four", type=Path)
    parser.add_argument("eight_by_eight", type=Path)
    parser.add_argument("styled_eight_by_eight", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = evaluate_tile_transfer(
        args.four_by_four,
        args.eight_by_eight,
        args.styled_eight_by_eight,
        args.output_dir,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "metrics": output["metrics"],
                "chart": output["chart"],
                "training": output["training"],
                "methods": output["methods"],
                "all_representation_gates_passed": output[
                    "all_representation_gates_passed"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
