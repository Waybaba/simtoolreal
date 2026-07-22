"""Compare raw, transition, random, and oracle trajectory distances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _standardize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    scale = features.std(axis=0, keepdims=True)
    return (features - mean) / np.maximum(scale, 1.0e-6)


def _raw_features(states: np.ndarray, samples: int = 12) -> np.ndarray:
    indices = np.linspace(0, states.shape[1] - 1, samples, dtype=np.int64)
    return _standardize(states[:, indices].reshape(states.shape[0], -1))


def _transition_features(states: np.ndarray) -> np.ndarray:
    delta = np.diff(states, axis=1)
    path_length = np.linalg.norm(delta, axis=-1).sum(axis=1, keepdims=True)
    max_step = np.linalg.norm(delta, axis=-1).max(axis=1, keepdims=True)
    displacement = states[:, -1] - states[:, 0]
    features = np.concatenate(
        [
            states[:, 0],
            states[:, -1],
            displacement,
            states.mean(axis=1),
            states.std(axis=1),
            path_length,
            max_step,
        ],
        axis=1,
    )
    return _standardize(features)


def _oracle_features(classes: np.ndarray, transition: np.ndarray, class_count: int) -> np.ndarray:
    one_hot = np.eye(class_count, dtype=np.float32)[classes]
    return np.concatenate([one_hot, 0.02 * transition], axis=1)


def _feature_sets(states: np.ndarray, classes: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    raw = _raw_features(states)
    transition = _transition_features(states)
    rng = np.random.default_rng(seed)
    projection = rng.normal(0.0, 1.0 / np.sqrt(raw.shape[1]), size=(raw.shape[1], 16))
    random_features = _standardize(raw @ projection)
    oracle = _oracle_features(classes, transition, int(classes.max()) + 1)
    return {
        "raw_l2": raw,
        "random_feature": random_features,
        "object_transition": transition,
        "semantic_oracle": oracle,
    }


def _pairwise_squared(features: np.ndarray) -> np.ndarray:
    norms = np.sum(features * features, axis=1, keepdims=True)
    return np.maximum(norms + norms.T - 2.0 * features @ features.T, 0.0)


def _balanced_nearest_neighbor_accuracy(features: np.ndarray, classes: np.ndarray) -> float:
    distances = _pairwise_squared(features)
    np.fill_diagonal(distances, np.inf)
    predictions = classes[np.argmin(distances, axis=1)]
    recalls = [np.mean(predictions[classes == class_id] == class_id) for class_id in np.unique(classes)]
    return float(np.mean(recalls))


def _distance_ratio(features: np.ndarray, classes: np.ndarray) -> float:
    distances = np.sqrt(_pairwise_squared(features) + 1.0e-12)
    same = classes[:, None] == classes[None, :]
    diagonal = np.eye(len(classes), dtype=bool)
    intra = distances[same & ~diagonal].mean()
    inter = distances[~same].mean()
    return float(inter / max(intra, 1.0e-8))


def _farthest_point_sample(features: np.ndarray, count: int) -> np.ndarray:
    count = min(count, len(features))
    selected = np.empty(count, dtype=np.int64)
    center = features.mean(axis=0, keepdims=True)
    selected[0] = int(np.argmax(np.sum((features - center) ** 2, axis=1)))
    min_distance = np.sum((features - features[selected[0]]) ** 2, axis=1)
    for index in range(1, count):
        selected[index] = int(np.argmax(min_distance))
        next_distance = np.sum((features - features[selected[index]]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, next_distance)
    return selected


def _normalized_entropy(classes: np.ndarray, class_count: int) -> float:
    counts = np.bincount(classes, minlength=class_count).astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / np.log(class_count))


def _geometric_coverage(states: np.ndarray, bins: int = 16) -> float:
    coordinates = np.floor((np.clip(states, -1.0, 1.0) + 1.0) * 0.5 * bins).astype(np.int64)
    coordinates = np.clip(coordinates, 0, bins - 1)
    occupied = np.unique(coordinates.reshape(-1, 2), axis=0)
    return float(len(occupied) / (bins * bins))


def _write_svg(path: Path, results: dict[str, dict[str, float]]) -> None:
    width, height = 760, 360
    names = list(results)
    colors = ["#334e68", "#9a6b3f", "#487c6c", "#c8483c"]
    bars = []
    group_width = 150
    baseline = 290
    for index, name in enumerate(names):
        x = 80 + index * group_width
        for offset, key in enumerate(("balanced_knn_accuracy", "representative_semantic_entropy")):
            value = results[name][key]
            bar_height = 210 * value
            bar_x = x + offset * 42
            bars.append(
                f'<rect x="{bar_x}" y="{baseline - bar_height:.1f}" width="32" height="{bar_height:.1f}" fill="{colors[index]}" opacity="{1.0 - 0.25 * offset}"/>'
            )
            bars.append(f'<text x="{bar_x + 16}" y="{baseline - bar_height - 7:.1f}" text-anchor="middle" font-size="12">{value:.2f}</text>')
        bars.append(f'<text x="{x + 37}" y="315" text-anchor="middle" font-size="12">{name.replace("_", " ")}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fbfcfd"/>
<text x="32" y="34" font-family="sans-serif" font-size="20" fill="#172b3a">Point-Cup metric sanity check</text>
<line x1="55" y1="{baseline}" x2="730" y2="{baseline}" stroke="#9aa6ad"/>
<line x1="55" y1="80" x2="55" y2="{baseline}" stroke="#9aa6ad"/>
<text x="62" y="65" font-family="sans-serif" font-size="12">solid: balanced kNN, light: representative semantic entropy</text>
<g font-family="sans-serif" fill="#172b3a">{''.join(bars)}</g>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--representatives", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    payload = np.load(args.dataset)
    audit_states = payload["audit_states"]
    audit_classes = payload["audit_episode_class"].astype(np.int64)
    discovery_states = payload["discovery_states"]
    discovery_classes = payload["discovery_episode_class"].astype(np.int64)
    class_names = [str(value) for value in payload["class_names"]]
    class_count = len(class_names)

    audit_features = _feature_sets(audit_states, audit_classes, args.seed)
    discovery_features = _feature_sets(discovery_states, discovery_classes, args.seed)
    results: dict[str, dict[str, float | int | list[int]]] = {}

    for name, features in audit_features.items():
        selected = _farthest_point_sample(discovery_features[name], args.representatives)
        selected_classes = discovery_classes[selected]
        covered = set(int(value) for value in selected_classes)
        results[name] = {
            "balanced_knn_accuracy": _balanced_nearest_neighbor_accuracy(features, audit_classes),
            "inter_intra_distance_ratio": _distance_ratio(features, audit_classes),
            "representative_class_coverage": len(covered) / class_count,
            "representative_rare_recall": len(covered.intersection({1, 2, 3})) / 3.0,
            "representative_semantic_entropy": _normalized_entropy(selected_classes, class_count),
            "representative_geometric_coverage": _geometric_coverage(discovery_states[selected]),
            "representative_class_counts": np.bincount(selected_classes, minlength=class_count).tolist(),
        }

    output = {
        "dataset": str(args.dataset.resolve()),
        "representatives": args.representatives,
        "class_names": class_names,
        "discovery_class_counts": np.bincount(discovery_classes, minlength=class_count).tolist(),
        "audit_observed_matches_intent": float(np.mean(payload["audit_observed_class"] == audit_classes)),
        "metrics": results,
    }
    output_path = args.dataset.parent / "metrics.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    _write_svg(args.dataset.parent / "metric_comparison.svg", results)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
