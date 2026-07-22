"""Evaluate frozen visual cluster centers on a new FrozenLake layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from skill_discovery.calibrate_frozenlake_visual_embeddings import (
    _aligned_predictions,
    build_self_reference_representations,
)
from skill_discovery.frozenlake import FROZENLAKE_OUTCOMES


def classification_metrics(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    generation_seeds: np.ndarray,
) -> dict[str, object]:
    confusion = np.zeros((3, 3), dtype=np.int64)
    for expected, predicted in zip(outcomes, predictions):
        confusion[int(expected), int(predicted)] += 1
    recalls = {
        name: float(confusion[index, index] / confusion[index].sum())
        for index, name in enumerate(FROZENLAKE_OUTCOMES)
    }
    per_seed = {}
    for seed in sorted(set(generation_seeds.tolist())):
        indices = np.flatnonzero(generation_seeds == seed)
        local = np.zeros((3, 3), dtype=np.int64)
        for expected, predicted in zip(outcomes[indices], predictions[indices]):
            local[int(expected), int(predicted)] += 1
        per_seed[str(seed)] = {
            "accuracy": float(np.trace(local) / local.sum()),
            "confusion": local.tolist(),
        }
    accuracy = float(np.trace(confusion) / confusion.sum())
    return {
        "accuracy": accuracy,
        "recall_by_outcome": recalls,
        "confusion": confusion.tolist(),
        "per_seed": per_seed,
        "gate_passed": bool(
            accuracy >= 0.80 and all(value >= 0.70 for value in recalls.values())
        ),
    }


def _nearest_centers(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = (
        np.sum(features * features, axis=1)[:, None]
        + np.sum(centers * centers, axis=1)[None, :]
        - 2.0 * features @ centers.T
    )
    return np.argmin(distances, axis=1)


def _write_chart(path: Path, methods: dict[str, dict[str, object]]) -> None:
    labels = ("accuracy", *FROZENLAKE_OUTCOMES)
    colors = ("#68757d", "#2b895f")
    width, height = 880, 430
    left, top, chart_width, chart_height = 70, 55, 740, 290
    group_width = chart_width / len(labels)
    elements = []
    for method_index, (name, metrics) in enumerate(methods.items()):
        values = (
            metrics["accuracy"],
            *(metrics["recall_by_outcome"][label] for label in FROZENLAKE_OUTCOMES),
        )
        for index, value in enumerate(values):
            x = left + index * group_width + 35 + method_index * 42
            bar_height = chart_height * float(value)
            elements.append(
                f'<rect x="{x:.1f}" y="{top + chart_height - bar_height:.1f}" '
                f'width="32" height="{bar_height:.1f}" fill="{colors[method_index]}"/>'
            )
        elements.append(
            f'<text x="{560 + method_index * 125}" y="30" '
            f'font-family="sans-serif" font-size="11" fill="{colors[method_index]}">{name}</text>'
        )
    label_elements = [
        f'<text x="{left + index * group_width + 18:.1f}" '
        f'y="{top + chart_height + 24}" font-family="sans-serif" '
        f'font-size="11">{label.replace("_terminal", "")}</text>'
        for index, label in enumerate(labels)
    ]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fbfcfd"/>
<text x="30" y="30" font-family="sans-serif" font-size="19" fill="#172b3a">Frozen 4x4 centers to 8x8 zero-shot</text>
<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#83919a"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#83919a"/>
<text x="28" y="{top + 5}" font-family="sans-serif" font-size="12">1.0</text>
<text x="28" y="{top + chart_height + 5}" font-family="sans-serif" font-size="12">0.0</text>
{''.join(elements)}
{''.join(label_elements)}
</svg>'''
    path.write_text(svg, encoding="utf-8")


def evaluate_transfer(
    dataset_path: Path,
    embedding_path: Path,
    source_metrics_path: Path,
    source_clusters_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    data = np.load(dataset_path)
    embeddings = np.load(embedding_path)
    source_metrics = json.loads(source_metrics_path.read_text())
    source_clusters = np.load(source_clusters_path)
    outcomes = data["outcomes"].astype(np.int64)
    generation_seeds = data["generation_seeds"].astype(np.int64)
    representations = build_self_reference_representations(
        embeddings["frame_embeddings"]
    )
    methods = {}
    assignments = {}
    for name in ("absolute_3frame", "temporal_delta"):
        centers = source_clusters[f"{name}_centers"]
        clusters = _nearest_centers(representations[name], centers)
        predictions = _aligned_predictions(
            clusters,
            source_metrics["methods"][name]["cluster_to_outcome"],
        )
        methods[name] = classification_metrics(
            predictions,
            outcomes,
            generation_seeds,
        )
        assignments[f"{name}_clusters"] = clusters
        assignments[f"{name}_predictions"] = predictions
    np.savez_compressed(output_dir / "transfer_assignments.npz", **assignments)
    output = {
        "target_dataset": str(dataset_path.resolve()),
        "target_embeddings": str(embedding_path.resolve()),
        "source_metrics": str(source_metrics_path.resolve()),
        "source_clusters": str(source_clusters_path.resolve()),
        "label_usage": "target labels used only for balanced acceptance and evaluation",
        "gate": {
            "minimum_accuracy": 0.80,
            "minimum_recall_per_outcome": 0.70,
        },
        "methods": methods,
        "temporal_delta_transfer_gate_passed": bool(
            methods["temporal_delta"]["gate_passed"]
        ),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    chart_path = output_dir / "zero_shot_transfer.svg"
    _write_chart(chart_path, methods)
    output["metrics"] = str(metrics_path.resolve())
    output["chart"] = str(chart_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("embeddings", type=Path)
    parser.add_argument("source_metrics", type=Path)
    parser.add_argument("source_clusters", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = evaluate_transfer(
        args.dataset,
        args.embeddings,
        args.source_metrics,
        args.source_clusters,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "metrics": output["metrics"],
                "chart": output["chart"],
                "methods": output["methods"],
                "temporal_delta_transfer_gate_passed": output[
                    "temporal_delta_transfer_gate_passed"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
