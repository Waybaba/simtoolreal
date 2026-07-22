"""Evaluate raw, DINO, and RGB car-motion features on MountainCar pairs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from skill_discovery.audit_mountaincar_frame_pair_capacity import PAIR_CLASSES
from skill_discovery.generate_point_cup_dataset import _write_png


def normalize(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1.0e-8)


def raw_pair_features(frames: np.ndarray) -> np.ndarray:
    if frames.ndim != 5 or frames.shape[1:] != (2, 400, 600, 3):
        raise ValueError("raw pair features require Nx2x400x600x3 RGB")
    y = np.linspace(0, 399, 20).astype(np.int64)
    x = np.linspace(0, 599, 30).astype(np.int64)
    sampled = frames[:, :, y][:, :, :, x]
    grayscale = (
        0.299 * sampled[..., 0]
        + 0.587 * sampled[..., 1]
        + 0.114 * sampled[..., 2]
    )
    return normalize(grayscale.reshape(len(frames), -1))


def car_x_pair_features(
    frames: np.ndarray,
    background: np.ndarray,
    *,
    chunk_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    if frames.ndim != 5 or frames.shape[1:] != (2, 400, 600, 3):
        raise ValueError("car motion features require Nx2x400x600x3 RGB")
    if background.shape != (400, 600, 3):
        raise ValueError("background must be a full MountainCar RGB frame")
    x_coordinates = np.arange(600, dtype=np.float64)
    centroids = []
    background_int = background.astype(np.int16)
    for start in range(0, len(frames), chunk_size):
        batch = frames[start : start + chunk_size].astype(np.int16)
        difference = np.abs(batch - background_int[None, None])
        column_energy = difference.sum(axis=(2, 4), dtype=np.float64)
        denominator = column_energy.sum(axis=2)
        if np.any(denominator <= 0):
            raise RuntimeError("RGB frame has no foreground difference from background")
        centroids.append(
            (column_energy * x_coordinates[None, None]).sum(axis=2) / denominator
        )
    car_x = np.concatenate(centroids).astype(np.float32)
    delta = car_x[:, 1] - car_x[:, 0]
    features = np.stack((car_x[:, 0], car_x[:, 1], delta), axis=1) / 600.0
    return features.astype(np.float32), car_x


def nearest_neighbor_predictions(
    features: np.ndarray,
    classes: np.ndarray,
    splits: np.ndarray,
    *,
    chunk_size: int = 128,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    reference = splits == 0
    reference_features = features[reference]
    reference_classes = classes[reference]
    predictions = []
    reference_squared = np.square(reference_features).sum(axis=1)
    for start in range(0, len(features), chunk_size):
        rows = features[start : start + chunk_size]
        distances = (
            np.square(rows).sum(axis=1, keepdims=True)
            + reference_squared[None]
            - 2.0 * rows @ reference_features.T
        )
        predictions.append(reference_classes[np.argmin(distances, axis=1)])
    return np.concatenate(predictions).astype(np.int8)


def classification_metrics(
    predictions: np.ndarray,
    classes: np.ndarray,
) -> dict[str, object]:
    predictions = np.asarray(predictions, dtype=np.int64)
    classes = np.asarray(classes, dtype=np.int64)
    confusion = np.zeros((len(PAIR_CLASSES), len(PAIR_CLASSES)), dtype=np.int64)
    np.add.at(confusion, (classes, predictions), 1)
    recalls = {
        name: float(confusion[index, index] / max(confusion[index].sum(), 1))
        for index, name in enumerate(PAIR_CLASSES)
    }
    return {
        "accuracy": float(np.mean(predictions == classes)),
        "macro_recall": float(np.mean(list(recalls.values()))),
        "recall_by_class": recalls,
        "confusion": confusion.tolist(),
        "predicted_class_sizes": np.bincount(
            predictions, minlength=len(PAIR_CLASSES)
        ).tolist(),
    }


def _write_chart(path: Path, methods: dict[str, dict[str, object]]) -> None:
    labels = ("accuracy", *PAIR_CLASSES)
    colors = ("#66747c", "#2875a4", "#2b895f")
    width, height = 980, 450
    left, top, chart_width, chart_height = 65, 65, 850, 290
    elements = [f'<rect width="{width}" height="{height}" fill="#f8fafb"/>']
    group_width = chart_width / len(labels)
    for group, label in enumerate(labels):
        group_x = left + group * group_width
        for method_index, (_, metrics) in enumerate(methods.items()):
            value = (
                float(metrics["accuracy"])
                if label == "accuracy"
                else float(metrics["recall_by_class"][label])
            )
            x = group_x + 14 + method_index * 28
            bar_height = value * chart_height
            elements.append(
                f'<rect x="{x:.1f}" y="{top + chart_height - bar_height:.1f}" '
                f'width="21" height="{bar_height:.1f}" fill="{colors[method_index]}"/>'
            )
        elements.append(
            f'<text x="{group_x + group_width / 2:.1f}" y="{top + chart_height + 23}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="11">{label}</text>'
        )
    elements.append(
        '<text x="65" y="35" font-family="sans-serif" font-size="22">'
        "MountainCar frame-pair relation metric</text>"
    )
    for index, method in enumerate(methods):
        x = 100 + index * 290
        elements.append(
            f'<rect x="{x}" y="415" width="14" height="14" fill="{colors[index]}"/>'
            f'<text x="{x + 21}" y="427" font-family="sans-serif" font-size="12">'
            f"{method}</text>"
        )
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        + "".join(elements)
        + "</svg>\n",
        encoding="utf-8",
    )


def evaluate_dataset(
    dataset_dir: Path,
    embeddings_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    shard_paths = sorted(dataset_dir.glob("class_*.npz"))
    if len(shard_paths) != 4:
        raise ValueError("expected four MountainCar pair shards")
    output_dir.mkdir(parents=True, exist_ok=False)

    reference_frame_rows = []
    for shard_path in shard_paths:
        with np.load(shard_path) as shard:
            frames = shard["frames"].astype(np.uint8)
            splits = shard["splits"].astype(np.int8)
            reference_frame_rows.append(frames[splits == 0].reshape(-1, 400, 600, 3))
    reference_frames = np.concatenate(reference_frame_rows)
    background = np.median(reference_frames, axis=0, overwrite_input=True).astype(
        np.uint8
    )
    del reference_frames, reference_frame_rows
    background_path = output_dir / "median_reference_background.png"
    _write_png(background_path, background)

    raw_rows = []
    car_feature_rows = []
    car_x_rows = []
    class_rows = []
    split_rows = []
    state_rows = []
    next_state_rows = []
    pair_hash_rows = []
    for shard_path in shard_paths:
        with np.load(shard_path) as shard:
            frames = shard["frames"].astype(np.uint8)
            raw_rows.append(raw_pair_features(frames))
            car_features, car_x = car_x_pair_features(frames, background)
            car_feature_rows.append(car_features)
            car_x_rows.append(car_x)
            class_rows.append(shard["classes"].astype(np.int8))
            split_rows.append(shard["splits"].astype(np.int8))
            state_rows.append(shard["states"].astype(np.float32))
            next_state_rows.append(shard["next_states"].astype(np.float32))
            pair_hash_rows.append(shard["pair_hashes"])
    raw_features = np.concatenate(raw_rows)
    car_features = np.concatenate(car_feature_rows)
    car_x = np.concatenate(car_x_rows)
    classes = np.concatenate(class_rows)
    splits = np.concatenate(split_rows)
    states = np.concatenate(state_rows)
    next_states = np.concatenate(next_state_rows)
    pair_hashes = np.concatenate(pair_hash_rows)

    with np.load(embeddings_path) as encoded:
        dino_features = encoded["pair_features"].astype(np.float32)
        dino_hashes = encoded["pair_hashes"]
        dino_classes = encoded["classes"].astype(np.int8)
        dino_splits = encoded["splits"].astype(np.int8)
    alignment_gate = bool(
        np.array_equal(pair_hashes, dino_hashes)
        and np.array_equal(classes, dino_classes)
        and np.array_equal(splits, dino_splits)
    )
    if not alignment_gate:
        raise RuntimeError("DINO embeddings do not align with dataset shards")

    audit = splits == 1
    methods = {}
    for name, features in (
        ("raw_pair_1nn", raw_features),
        ("dinov2_pair_1nn", dino_features),
        ("median_background_car_motion_1nn", car_features),
    ):
        predictions = nearest_neighbor_predictions(features, classes, splits)
        methods[name] = classification_metrics(predictions[audit], classes[audit])

    x_before_correlation = float(spearmanr(car_x[audit, 0], states[audit, 0]).statistic)
    x_after_correlation = float(
        spearmanr(car_x[audit, 1], next_states[audit, 0]).statistic
    )
    visual_delta = car_x[:, 1] - car_x[:, 0]
    delta_sign_accuracy = float(
        np.mean(
            np.sign(visual_delta[audit])
            == np.sign(next_states[audit, 1])
        )
    )
    localization_gate = bool(
        abs(x_before_correlation) >= 0.995
        and abs(x_after_correlation) >= 0.995
        and delta_sign_accuracy >= 0.99
    )
    candidate = methods["median_background_car_motion_1nn"]
    semantic_gate = bool(
        candidate["accuracy"] >= 0.98
        and candidate["macro_recall"] >= 0.98
        and min(candidate["recall_by_class"].values()) >= 0.95
        and all(size > 0 for size in candidate["predicted_class_sizes"])
    )
    chart_path = output_dir / "frame_pair_metric.svg"
    _write_chart(chart_path, methods)
    feature_path = output_dir / "car_motion_features.npz"
    np.savez_compressed(
        feature_path,
        car_features=car_features,
        car_x=car_x,
        classes=classes,
        splits=splits,
        pair_hashes=pair_hashes,
    )
    output = {
        "dataset_dir": str(dataset_dir.resolve()),
        "embeddings": str(embeddings_path.resolve()),
        "pair_count": len(classes),
        "alignment_gate_passed": alignment_gate,
        "median_background": str(background_path.resolve()),
        "car_motion_features": str(feature_path.resolve()),
        "localization": {
            "x_before_spearman": x_before_correlation,
            "x_after_spearman": x_after_correlation,
            "delta_sign_accuracy": delta_sign_accuracy,
            "gate_passed": localization_gate,
        },
        "methods": methods,
        "semantic_gate_passed": semantic_gate,
        "chart": str(chart_path.resolve()),
        "final_gate_passed": bool(
            alignment_gate and localization_gate and semantic_gate
        ),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["metrics"] = str(metrics_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("embeddings", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_id = f"frame_pair_metric_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/mountaincar_continuous"
    ) / run_id
    output = evaluate_dataset(args.dataset_dir, args.embeddings, output_dir)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
