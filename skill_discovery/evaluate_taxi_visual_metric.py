"""Evaluate frozen raw/DINO reference metrics over the complete Taxi domain."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from skill_discovery.audit_gotoobject_balanced_transition import (
    maximum_weight_assignment,
)
from skill_discovery.audit_taxi_environment import TAXI_STAGES
from skill_discovery.encode_minigrid_doorkey_dinov2 import MODEL_ID
from skill_discovery.evaluate_minigrid_doorkey_visual_sequence_graph import (
    _encode_frames,
)
from skill_discovery.evaluate_minigrid_doorkey_visual_state_transfer import (
    _normalize,
    nearest_centers,
)
from skill_discovery.generate_taxi_visual_dataset import TAXI_SPLITS


def raw_frame_features(frames: np.ndarray) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("raw Taxi frames must be an RGB batch")
    y = np.linspace(0, frames.shape[1] - 1, 32).astype(np.int64)
    x = np.linspace(0, frames.shape[2] - 1, 32).astype(np.int64)
    resized = frames[:, y][:, :, x].astype(np.float32) / 255.0
    return _normalize(resized.reshape(len(frames), -1)).astype(np.float32)


def fit_reference_centers(features: np.ndarray, stages: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    stages = np.asarray(stages, dtype=np.int64)
    if features.ndim != 2 or stages.shape != (len(features),):
        raise ValueError("reference features and stages do not align")
    centers = []
    for stage in range(len(TAXI_STAGES)):
        selected = features[stages == stage]
        if not len(selected):
            raise ValueError("reference split is missing a Taxi stage")
        centers.append(selected.mean(axis=0))
    return _normalize(np.asarray(centers, dtype=np.float32)).astype(np.float32)


def taxi_classification_metrics(
    predictions: np.ndarray,
    stages: np.ndarray,
    destinations: np.ndarray,
    orientations: np.ndarray,
) -> dict[str, object]:
    predictions = np.asarray(predictions, dtype=np.int64)
    stages = np.asarray(stages, dtype=np.int64)
    destinations = np.asarray(destinations, dtype=np.int64)
    orientations = np.asarray(orientations, dtype=np.int64)
    if not (
        predictions.shape
        == stages.shape
        == destinations.shape
        == orientations.shape
        and predictions.ndim == 1
    ):
        raise ValueError("Taxi classification vectors do not align")
    confusion = np.zeros((3, 3), dtype=np.int64)
    for expected, predicted in zip(stages, predictions):
        confusion[int(expected), int(predicted)] += 1
    if np.any(confusion.sum(axis=1) == 0):
        raise ValueError("Taxi metric split is missing an oracle stage")
    recalls = {
        name: float(confusion[index, index] / confusion[index].sum())
        for index, name in enumerate(TAXI_STAGES)
    }
    by_destination = {
        str(int(group)): {
            "accuracy": float(
                np.mean(predictions[destinations == group] == stages[destinations == group])
            )
        }
        for group in sorted(set(destinations.tolist()))
    }
    by_orientation = {
        str(int(group)): {
            "accuracy": float(
                np.mean(predictions[orientations == group] == stages[orientations == group])
            )
        }
        for group in sorted(set(orientations.tolist()))
    }
    sizes = np.bincount(predictions, minlength=3)
    return {
        "accuracy": float(np.mean(predictions == stages)),
        "recall_by_stage": recalls,
        "confusion": confusion.tolist(),
        "predicted_class_sizes": sizes.tolist(),
        "all_classes_nonempty": bool(np.all(sizes > 0)),
        "by_destination": by_destination,
        "by_orientation": by_orientation,
    }


def reference_method(
    features: np.ndarray,
    stages: np.ndarray,
    destinations: np.ndarray,
    orientations: np.ndarray,
    splits: np.ndarray,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    train = splits == 0
    centers = fit_reference_centers(features[train], stages[train])
    predictions = nearest_centers(features, centers)
    metrics = {}
    for split, name in enumerate(TAXI_SPLITS):
        selected = splits == split
        metrics[name] = taxi_classification_metrics(
            predictions[selected],
            stages[selected],
            destinations[selected],
            orientations[selected],
        )
    return metrics, centers, predictions


def natural_kmeans_method(
    features: np.ndarray,
    stages: np.ndarray,
    destinations: np.ndarray,
    orientations: np.ndarray,
    splits: np.ndarray,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, tuple[int, ...]]:
    from sklearn.cluster import KMeans

    train = splits == 0
    model = KMeans(n_clusters=3, random_state=7, n_init=20)
    model.fit(features[train])
    clusters = model.predict(features).astype(np.int64)
    train_confusion = np.zeros((3, 3), dtype=np.int64)
    for cluster, stage in zip(clusters[train], stages[train]):
        train_confusion[int(cluster), int(stage)] += 1
    assignment, _ = maximum_weight_assignment(train_confusion)
    predictions = np.asarray([assignment[value] for value in clusters], dtype=np.int64)
    metrics = {
        "train_cluster_to_stage": list(assignment),
        "train_cluster_stage_confusion": train_confusion.tolist(),
        "splits": {},
    }
    for split, name in enumerate(TAXI_SPLITS):
        selected = splits == split
        row = taxi_classification_metrics(
            predictions[selected],
            stages[selected],
            destinations[selected],
            orientations[selected],
        )
        row["raw_cluster_sizes"] = np.bincount(
            clusters[selected],
            minlength=3,
        ).tolist()
        metrics["splits"][name] = row
    return metrics, model.cluster_centers_.astype(np.float32), clusters, assignment


def _write_chart(path: Path, methods: dict[str, dict[str, object]]) -> None:
    colors = ("#68757d", "#2875a4", "#d17031")
    width, height = 980, 440
    left, top, chart_width, chart_height = 70, 55, 840, 290
    group_width = chart_width / len(TAXI_SPLITS)
    bar_width = 24
    elements = []
    names = list(methods)
    for split_index, split_name in enumerate(TAXI_SPLITS):
        for method_index, method_name in enumerate(names):
            row = methods[method_name][split_name]
            value = float(row["accuracy"])
            x = left + split_index * group_width + 24 + method_index * 34
            bar_height = value * chart_height
            elements.append(
                f'<rect x="{x:.1f}" y="{top + chart_height - bar_height:.1f}" '
                f'width="{bar_width}" height="{bar_height:.1f}" '
                f'fill="{colors[method_index]}"/>'
            )
        label = split_name.replace("_", " ")
        elements.append(
            f'<text x="{left + split_index * group_width + 8:.1f}" '
            f'y="{top + chart_height + 25}" font-family="sans-serif" '
            f'font-size="11">{label}</text>'
        )
    legend = " | ".join(names)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#fbfcfd"/>'
        '<text x="28" y="30" font-family="sans-serif" font-size="20" '
        'fill="#172b3a">Taxi full-domain visual metric accuracy</text>'
        f'<text x="570" y="30" font-family="sans-serif" font-size="11">'
        f'{legend}</text>'
        f'<line x1="{left}" y1="{top + chart_height}" '
        f'x2="{left + chart_width}" y2="{top + chart_height}" stroke="#83919a"/>'
        f'{"".join(elements)}</svg>'
    )
    path.write_text(svg, encoding="utf-8")


def evaluate_taxi_visual_metric(
    dataset_path: Path,
    output_dir: Path,
    *,
    model_id: str = MODEL_ID,
    batch_size: int = 64,
    device: str = "cuda:0",
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(dataset_path) as data:
        frames = np.asarray(data["frames"], dtype=np.uint8)
        stages = data["stages"].astype(np.int64)
        destinations = data["destinations"].astype(np.int64)
        orientations = data["orientations"].astype(np.int64)
        splits = data["splits"].astype(np.int64)
    data_metrics_path = dataset_path.parent / "metrics.json"
    data_metrics = json.loads(data_metrics_path.read_text(encoding="utf-8"))
    if not data_metrics["data_gate_passed"]:
        raise ValueError("Taxi visual dataset gate did not pass")
    raw = raw_frame_features(frames)
    dino, elapsed, gpu_name, versions = _encode_frames(
        frames,
        model_id=model_id,
        batch_size=batch_size,
        device=device,
    )
    representations = {"raw_reference": raw, "dinov2_reference": dino}
    reference_metrics = {}
    arrays = {"dinov2_embeddings": dino}
    for name, features in representations.items():
        metrics, centers, predictions = reference_method(
            features,
            stages,
            destinations,
            orientations,
            splits,
        )
        reference_metrics[name] = metrics
        arrays[f"{name}_centers"] = centers
        arrays[f"{name}_predictions"] = predictions
    kmeans_metrics = {}
    for name, features in representations.items():
        metrics, centers, clusters, assignment = natural_kmeans_method(
            features,
            stages,
            destinations,
            orientations,
            splits,
        )
        kmeans_metrics[name] = metrics
        arrays[f"{name}_kmeans_centers"] = centers
        arrays[f"{name}_kmeans_clusters"] = clusters
        arrays[f"{name}_kmeans_assignment"] = np.asarray(assignment)
    primary = reference_metrics["dinov2_reference"]["joint_audit"]
    gate = {
        "minimum_accuracy": 0.80,
        "minimum_recall_per_stage": 0.75,
        "minimum_accuracy_per_destination": 0.75,
        "minimum_accuracy_per_orientation": 0.75,
        "require_all_classes_nonempty": True,
    }
    passed = bool(
        primary["accuracy"] >= gate["minimum_accuracy"]
        and min(primary["recall_by_stage"].values())
        >= gate["minimum_recall_per_stage"]
        and min(row["accuracy"] for row in primary["by_destination"].values())
        >= gate["minimum_accuracy_per_destination"]
        and min(row["accuracy"] for row in primary["by_orientation"].values())
        >= gate["minimum_accuracy_per_orientation"]
        and primary["all_classes_nonempty"]
    )
    output = {
        "dataset": str(dataset_path.resolve()),
        "dataset_metrics": str(data_metrics_path.resolve()),
        "model_id": model_id,
        "device": device,
        "gpu_name": gpu_name,
        "versions": versions,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "frame_count": len(frames),
        "fit_or_refit_on_audit_performed": False,
        "oracle_labels_used_for_reference_centers": True,
        "oracle_labels_used_for_kmeans_fit": False,
        "gate": gate,
        "reference_methods": reference_metrics,
        "natural_kmeans_diagnostics": kmeans_metrics,
        "primary_method": "dinov2_reference",
        "primary_split": "joint_audit",
        "visual_metric_upper_bound_gate_passed": passed,
    }
    assignments_path = output_dir / "taxi_visual_metric_assignments.npz"
    np.savez_compressed(assignments_path, **arrays)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    chart_path = output_dir / "taxi_visual_metric_accuracy.svg"
    _write_chart(chart_path, reference_metrics)
    output["metrics"] = str(metrics_path.resolve())
    output["assignments"] = str(assignments_path.resolve())
    output["chart"] = str(chart_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/taxi"
    ) / f"taxi_visual_metric_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = evaluate_taxi_visual_metric(
        args.dataset,
        output_dir,
        model_id=args.model_id,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "metrics": output["metrics"],
                "chart": output["chart"],
                "elapsed_seconds": output["elapsed_seconds"],
                "reference_methods": output["reference_methods"],
                "natural_kmeans_diagnostics": output[
                    "natural_kmeans_diagnostics"
                ],
                "visual_metric_upper_bound_gate_passed": output[
                    "visual_metric_upper_bound_gate_passed"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
