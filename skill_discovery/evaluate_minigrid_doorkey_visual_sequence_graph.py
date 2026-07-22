"""Audit a frozen visual representation on complete DoorKey policy sequences."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from skill_discovery.audit_gotoobject_balanced_transition import (
    supported_predecessors,
    transition_counts_from_stage_episodes,
    transitive_ancestors,
)
from skill_discovery.encode_minigrid_doorkey_dinov2 import MODEL_ID
from skill_discovery.evaluate_minigrid_doorkey_visual_state_transfer import (
    _normalize,
    nearest_centers,
)
from skill_discovery.minigrid_doorkey import DOORKEY_STAGES


def classification_metrics(
    clusters: np.ndarray,
    stages: np.ndarray,
    cluster_to_stage: dict[str, str],
) -> dict[str, object]:
    clusters = np.asarray(clusters, dtype=np.int64)
    stages = np.asarray(stages, dtype=np.int64)
    if clusters.ndim != 1 or stages.shape != clusters.shape:
        raise ValueError("clusters and stages must be matching vectors")
    stage_indices = {name: index for index, name in enumerate(DOORKEY_STAGES)}
    try:
        predictions = np.asarray(
            [stage_indices[cluster_to_stage[str(int(value))]] for value in clusters],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError("cluster mapping is incomplete or invalid") from error
    confusion = np.zeros((4, 4), dtype=np.int64)
    for expected, predicted in zip(stages, predictions):
        if expected < 0 or expected >= len(DOORKEY_STAGES):
            raise ValueError("oracle stage is outside the DoorKey stage range")
        confusion[int(expected), int(predicted)] += 1
    if np.any(confusion.sum(axis=1) == 0):
        raise ValueError("classification audit must contain every oracle stage")
    recalls = {
        name: float(confusion[index, index] / confusion[index].sum())
        for index, name in enumerate(DOORKEY_STAGES)
    }
    cluster_sizes = np.bincount(clusters, minlength=4)
    return {
        "accuracy": float(np.mean(predictions == stages)),
        "recall_by_stage": recalls,
        "confusion": confusion.tolist(),
        "cluster_sizes": cluster_sizes.tolist(),
        "all_clusters_nonempty": bool(np.all(cluster_sizes[:4] > 0)),
        "predictions": predictions,
    }


def audit_cluster_graph(
    cluster_sequences: list[list[int]],
    reset_clusters: np.ndarray,
    native_goal_terminal_clusters: np.ndarray,
    *,
    num_clusters: int = 4,
    minimum_count: int = 25,
    minimum_share: float = 0.01,
) -> dict[str, object]:
    counts = transition_counts_from_stage_episodes(
        cluster_sequences,
        num_stages=num_clusters,
    )
    predecessors = supported_predecessors(
        counts,
        minimum_count=minimum_count,
        minimum_share=minimum_share,
    )
    roots = tuple(sorted(set(np.asarray(reset_clusters, dtype=np.int64).tolist())))
    goals = tuple(
        sorted(
            set(
                np.asarray(
                    native_goal_terminal_clusters,
                    dtype=np.int64,
                ).tolist()
            )
        )
    )
    root_unique = len(roots) == 1
    goal_unique = len(goals) == 1
    cycle_free = True
    try:
        ancestors = transitive_ancestors(predecessors)
    except ValueError:
        cycle_free = False
        ancestors = tuple(() for _ in range(num_clusters))
    ancestor_cardinalities = tuple(len(row) for row in ancestors)
    root = roots[0] if root_unique else None
    goal = goals[0] if goal_unique else None
    root_has_no_predecessors = bool(
        root is not None and not predecessors[root]
    )
    every_nonroot_has_predecessor = bool(
        root is not None
        and all(predecessors[index] for index in range(num_clusters) if index != root)
    )
    ordered_chain = bool(
        cycle_free
        and sorted(ancestor_cardinalities) == list(range(num_clusters))
        and root is not None
        and ancestor_cardinalities[root] == 0
        and goal is not None
        and ancestor_cardinalities[goal] == num_clusters - 1
    )
    graph_gate = bool(
        root_unique
        and goal_unique
        and root != goal
        and root_has_no_predecessors
        and every_nonroot_has_predecessor
        and cycle_free
        and ordered_chain
    )
    return {
        "transition_counts": counts.tolist(),
        "supported_predecessors": [list(row) for row in predecessors],
        "transitive_ancestors": [list(row) for row in ancestors],
        "ancestor_cardinalities": list(ancestor_cardinalities),
        "reset_clusters": list(roots),
        "native_goal_terminal_clusters": list(goals),
        "root_unique": root_unique,
        "goal_unique": goal_unique,
        "root_has_no_predecessors": root_has_no_predecessors,
        "every_nonroot_has_predecessor": every_nonroot_has_predecessor,
        "cycle_free": cycle_free,
        "ordered_chain": ordered_chain,
        "graph_gate_passed": graph_gate,
    }


def _write_transition_chart(path: Path, graph: dict[str, object]) -> None:
    counts = np.asarray(graph["transition_counts"], dtype=np.int64)
    predecessors = graph["supported_predecessors"]
    colors = ("#68757d", "#2875a4", "#d17031", "#2b895f")
    cell = 100
    left, top = 150, 100
    width, height = 620, 590
    maximum = max(int(counts.max()), 1)
    elements = []
    for index in range(4):
        elements.append(
            f'<text x="{left - 12}" y="{top + index * cell + 57}" '
            f'text-anchor="end" font-family="sans-serif" font-size="13">'
            f'cluster {index}</text>'
        )
        elements.append(
            f'<text x="{left + index * cell + 50}" y="{top - 18}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="13">'
            f'cluster {index}</text>'
        )
        for target in range(4):
            value = int(counts[index, target])
            opacity = 0.08 + 0.82 * value / maximum
            supported = index in predecessors[target]
            elements.append(
                f'<rect x="{left + target * cell}" y="{top + index * cell}" '
                f'width="{cell - 3}" height="{cell - 3}" '
                f'fill="{colors[target]}" opacity="{opacity:.3f}" '
                f'stroke="{"#172b3a" if supported else "#d8dee2"}" '
                f'stroke-width="{3 if supported else 1}"/>'
            )
            elements.append(
                f'<text x="{left + target * cell + 49}" '
                f'y="{top + index * cell + 57}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="17">{value}</text>'
            )
    status = "PASS" if graph["graph_gate_passed"] else "FAIL"
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#fbfcfd"/>'
        '<text x="28" y="32" font-family="sans-serif" font-size="20" '
        'fill="#172b3a">Frozen DINO sequence transition audit</text>'
        f'<text x="28" y="60" font-family="sans-serif" font-size="14" '
        f'fill="{colors[3] if status == "PASS" else colors[2]}">{status}</text>'
        f'<text x="{left + 200}" y="{top - 48}" text-anchor="middle" '
        'font-family="sans-serif" font-size="12">target cluster</text>'
        f'<text x="32" y="{top + 200}" font-family="sans-serif" '
        'font-size="12">source</text>'
        f'{"".join(elements)}</svg>'
    )
    path.write_text(svg, encoding="utf-8")


def _encode_frames(
    frames: np.ndarray,
    *,
    model_id: str,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, float, str, dict[str, str]]:
    import torch
    import transformers
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
    model = AutoModel.from_pretrained(model_id)
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    use_half = torch_device.type == "cuda"
    if use_half:
        model.half()
    batches = []
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            inputs = processor(
                images=[image for image in frames[start : start + batch_size]],
                return_tensors="pt",
            )
            pixel_values = inputs["pixel_values"].to(torch_device)
            if use_half:
                pixel_values = pixel_values.half()
            output = model(pixel_values=pixel_values)
            batches.append(output.last_hidden_state[:, 0].float().cpu().numpy())
    elapsed = time.monotonic() - started
    embeddings = _normalize(np.concatenate(batches).astype(np.float32))
    gpu_name = (
        torch.cuda.get_device_name(torch_device)
        if torch_device.type == "cuda"
        else "cpu"
    )
    versions = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
    }
    return embeddings, elapsed, gpu_name, versions


def evaluate_sequence_graph(
    dataset_path: Path,
    cluster_metrics_path: Path,
    cluster_assignments_path: Path,
    output_dir: Path,
    *,
    model_id: str = MODEL_ID,
    batch_size: int = 64,
    device: str = "cuda:0",
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(dataset_path) as data:
        frames = data["state_frames"].astype(np.uint8)
        stages = data["state_stages"].astype(np.int64)
        offsets = data["sequence_offsets"].astype(np.int64)
        state_indices = data["sequence_state_indices"].astype(np.int64)
        target_stages = data["sequence_target_stages"].astype(np.int64)
        native_success = data["sequence_native_success"].astype(np.bool_)
    if offsets[0] != 0 or offsets[-1] != len(state_indices):
        raise ValueError("sequence offsets do not cover state occurrences")
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("all policy sequences must be non-empty")
    if len(target_stages) != len(offsets) - 1:
        raise ValueError("sequence metadata length does not match offsets")
    source_metrics = json.loads(cluster_metrics_path.read_text(encoding="utf-8"))
    source_method = source_metrics["methods"]["dinov2_current"]
    with np.load(cluster_assignments_path) as source_assignments:
        centers = source_assignments["dinov2_current_centers"].astype(np.float32)
    embeddings, elapsed, gpu_name, versions = _encode_frames(
        frames,
        model_id=model_id,
        batch_size=batch_size,
        device=device,
    )
    clusters = nearest_centers(embeddings, centers)
    unique_metrics = classification_metrics(
        clusters,
        stages,
        source_method["cluster_to_stage"],
    )
    unique_predictions = unique_metrics.pop("predictions")
    occurrence_clusters = clusters[state_indices]
    occurrence_stages = stages[state_indices]
    occurrence_metrics = classification_metrics(
        occurrence_clusters,
        occurrence_stages,
        source_method["cluster_to_stage"],
    )
    occurrence_predictions = occurrence_metrics.pop("predictions")
    cluster_sequences = [
        occurrence_clusters[start:stop].tolist()
        for start, stop in zip(offsets[:-1], offsets[1:])
    ]
    reset_clusters = occurrence_clusters[offsets[:-1]]
    goal_sequences = (target_stages == 3) & native_success
    goal_terminal_indices = offsets[1:][goal_sequences] - 1
    graph = audit_cluster_graph(
        cluster_sequences,
        reset_clusters,
        occurrence_clusters[goal_terminal_indices],
    )
    gate = {
        "minimum_accuracy": 0.85,
        "minimum_recall_per_stage": 0.75,
        "minimum_transition_count": 25,
        "minimum_transition_incoming_share": 0.01,
        "require_all_clusters_nonempty": True,
        "require_unique_root_and_goal": True,
        "require_acyclic_total_order": True,
    }
    unique_gate = bool(
        unique_metrics["accuracy"] >= gate["minimum_accuracy"]
        and min(unique_metrics["recall_by_stage"].values())
        >= gate["minimum_recall_per_stage"]
        and unique_metrics["all_clusters_nonempty"]
    )
    occurrence_gate = bool(
        occurrence_metrics["accuracy"] >= gate["minimum_accuracy"]
        and min(occurrence_metrics["recall_by_stage"].values())
        >= gate["minimum_recall_per_stage"]
        and occurrence_metrics["all_clusters_nonempty"]
    )
    dataset_metrics_path = dataset_path.parent / "metrics.json"
    dataset_metrics = json.loads(dataset_metrics_path.read_text(encoding="utf-8"))
    data_gate = bool(dataset_metrics["data_gate_passed"])
    sequence_gate = bool(
        data_gate
        and source_method["foundation_visual_gate_passed"]
        and unique_gate
        and occurrence_gate
        and graph["graph_gate_passed"]
    )
    output = {
        "dataset": str(dataset_path.resolve()),
        "dataset_metrics": str(dataset_metrics_path.resolve()),
        "cluster_metrics": str(cluster_metrics_path.resolve()),
        "cluster_assignments": str(cluster_assignments_path.resolve()),
        "model_id": model_id,
        "device": device,
        "gpu_name": gpu_name,
        "versions": versions,
        "elapsed_seconds": elapsed,
        "unique_state_count": len(frames),
        "state_occurrence_count": len(state_indices),
        "sequence_count": len(offsets) - 1,
        "fit_or_refit_performed": False,
        "representation": "dinov2_current",
        "gate": gate,
        "source_foundation_visual_gate_passed": bool(
            source_method["foundation_visual_gate_passed"]
        ),
        "data_gate_passed": data_gate,
        "unique_state_metrics": unique_metrics,
        "unique_state_accuracy_gate_passed": unique_gate,
        "occurrence_weighted_metrics": occurrence_metrics,
        "occurrence_accuracy_gate_passed": occurrence_gate,
        "graph": graph,
        "visual_sequence_gate_passed": sequence_gate,
    }
    assignments_path = output_dir / "sequence_cluster_assignments.npz"
    np.savez_compressed(
        assignments_path,
        state_embeddings=embeddings,
        state_clusters=clusters,
        state_predictions=unique_predictions,
        occurrence_clusters=occurrence_clusters,
        occurrence_predictions=occurrence_predictions,
        sequence_offsets=offsets,
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    chart_path = output_dir / "sequence_transition_audit.svg"
    _write_transition_chart(chart_path, graph)
    output["metrics"] = str(metrics_path.resolve())
    output["assignments"] = str(assignments_path.resolve())
    output["chart"] = str(chart_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--cluster-metrics", type=Path, required=True)
    parser.add_argument("--cluster-assignments", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/minigrid_doorkey_visual"
    ) / f"doorkey5_visual_sequence_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = evaluate_sequence_graph(
        args.dataset,
        args.cluster_metrics,
        args.cluster_assignments,
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
                "unique_state_metrics": output["unique_state_metrics"],
                "occurrence_weighted_metrics": output[
                    "occurrence_weighted_metrics"
                ],
                "graph": output["graph"],
                "visual_sequence_gate_passed": output[
                    "visual_sequence_gate_passed"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
