"""Evaluate an unlabeled causal ordered-cluster decoder on fresh DoorKey sequences."""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from skill_discovery.audit_gotoobject_balanced_transition import (
    transition_counts_from_stage_episodes,
)
from skill_discovery.encode_minigrid_doorkey_dinov2 import MODEL_ID
from skill_discovery.evaluate_minigrid_doorkey_visual_sequence_graph import (
    _encode_frames,
    classification_metrics,
)
from skill_discovery.evaluate_minigrid_doorkey_visual_state_transfer import (
    nearest_centers,
)
from skill_discovery.minigrid_doorkey import DOORKEY_STAGES


def infer_cluster_order(
    cluster_sequences: list[list[int]],
    *,
    num_clusters: int = 4,
) -> dict[str, object]:
    if not cluster_sequences or any(not sequence for sequence in cluster_sequences):
        raise ValueError("calibration sequences must be non-empty")
    reset_clusters = {int(sequence[0]) for sequence in cluster_sequences}
    if len(reset_clusters) != 1:
        raise ValueError("calibration resets do not share one root cluster")
    root = next(iter(reset_clusters))
    counts = transition_counts_from_stage_episodes(
        cluster_sequences,
        num_stages=num_clusters,
    )
    rows = []
    remaining = [index for index in range(num_clusters) if index != root]
    for suffix in itertools.permutations(remaining):
        order = (root, *suffix)
        positions = {cluster: index for index, cluster in enumerate(order)}
        forward = 0
        backward = 0
        for source in range(num_clusters):
            for target in range(num_clusters):
                if positions[source] < positions[target]:
                    forward += int(counts[source, target])
                elif positions[source] > positions[target]:
                    backward += int(counts[source, target])
        rows.append(
            {
                "order": list(order),
                "forward_count": forward,
                "backward_count": backward,
                "score": forward - backward,
            }
        )
    selected = sorted(rows, key=lambda row: (-row["score"], row["order"]))[0]
    return {
        "root_cluster": root,
        "transition_counts": counts.tolist(),
        "candidate_scores": rows,
        "selected_cluster_order": selected["order"],
        "selected_forward_count": selected["forward_count"],
        "selected_backward_count": selected["backward_count"],
        "selected_score": selected["score"],
        "oracle_labels_used": False,
    }


def causal_decode(raw_clusters: list[int], cluster_order: tuple[int, ...]) -> list[int]:
    if not raw_clusters:
        raise ValueError("cannot decode an empty sequence")
    if sorted(cluster_order) != list(range(len(cluster_order))):
        raise ValueError("cluster order must be a permutation")
    if raw_clusters[0] != cluster_order[0]:
        raise ValueError("sequence does not start at the calibrated root")
    position = 0
    decoded = []
    for cluster in raw_clusters:
        if cluster < 0 or cluster >= len(cluster_order):
            raise ValueError("raw sequence contains an invalid cluster")
        if (
            position + 1 < len(cluster_order)
            and cluster == cluster_order[position + 1]
        ):
            position += 1
        decoded.append(position)
    return decoded


def decoded_stage_metrics(
    decoded_sequences: list[list[int]],
    oracle_sequences: list[list[int]],
    target_stages: np.ndarray,
    native_success: np.ndarray,
    generation_groups: np.ndarray,
) -> dict[str, object]:
    sequence_count = len(decoded_sequences)
    if not (
        len(oracle_sequences)
        == len(target_stages)
        == len(native_success)
        == len(generation_groups)
        == sequence_count
    ):
        raise ValueError("sequence metadata lengths do not match")
    if sorted(set(target_stages.tolist())) != list(range(len(DOORKEY_STAGES))):
        raise ValueError("fresh audit must contain every target stage")
    if any(
        len(decoded) != len(oracle) or not decoded
        for decoded, oracle in zip(decoded_sequences, oracle_sequences)
    ):
        raise ValueError("decoded and oracle sequences must be matching and non-empty")
    predictions = np.asarray(
        [value for sequence in decoded_sequences for value in sequence],
        dtype=np.int64,
    )
    expected = np.asarray(
        [value for sequence in oracle_sequences for value in sequence],
        dtype=np.int64,
    )
    occurrence_groups = np.asarray(
        [
            int(group)
            for group, sequence in zip(generation_groups, decoded_sequences)
            for _ in sequence
        ],
        dtype=np.int64,
    )
    confusion = np.zeros((4, 4), dtype=np.int64)
    for actual, predicted in zip(expected, predictions):
        confusion[int(actual), int(predicted)] += 1
    if np.any(confusion.sum(axis=1) == 0):
        raise ValueError("fresh audit must contain every oracle stage")
    recall = {
        name: float(confusion[index, index] / confusion[index].sum())
        for index, name in enumerate(DOORKEY_STAGES)
    }
    by_group = {}
    for group in sorted(set(generation_groups.tolist())):
        selected = occurrence_groups == group
        by_group[str(int(group))] = {
            "accuracy": float(np.mean(predictions[selected] == expected[selected]))
        }
    final_predictions = np.asarray(
        [sequence[-1] for sequence in decoded_sequences],
        dtype=np.int64,
    )
    final_rates = {
        DOORKEY_STAGES[stage]: float(
            np.mean(final_predictions[target_stages == stage] == stage)
        )
        for stage in range(4)
    }
    goal_native = (target_stages == 3) & native_success
    if not np.any(goal_native):
        raise ValueError("fresh audit has no native-success goal sequence")
    goal_terminal_recall = float(np.mean(final_predictions[goal_native] == 3))
    non_goal = target_stages != 3
    false_goal_sequences = np.asarray(
        [3 in sequence for sequence in decoded_sequences],
        dtype=np.bool_,
    )
    non_goal_false_goal_rate = float(np.mean(false_goal_sequences[non_goal]))
    transition_counts = transition_counts_from_stage_episodes(
        decoded_sequences,
        num_stages=4,
    )
    adjacent_counts = [int(transition_counts[index, index + 1]) for index in range(3)]
    decoded_counts = np.bincount(predictions, minlength=4)
    return {
        "accuracy": float(np.mean(predictions == expected)),
        "recall_by_stage": recall,
        "confusion": confusion.tolist(),
        "by_generation_group": by_group,
        "decoded_stage_counts": decoded_counts.tolist(),
        "all_stages_nonempty": bool(np.all(decoded_counts > 0)),
        "final_state_rate_by_target": final_rates,
        "goal_native_terminal_recall": goal_terminal_recall,
        "non_goal_false_goal_sequence_rate": non_goal_false_goal_rate,
        "all_resets_root": bool(all(sequence[0] == 0 for sequence in decoded_sequences)),
        "decoded_transition_counts": transition_counts.tolist(),
        "adjacent_transition_counts": adjacent_counts,
    }


def _write_chart(
    path: Path,
    order: list[int],
    metrics: dict[str, object],
    passed: bool,
) -> None:
    colors = ("#68757d", "#2875a4", "#d17031", "#2b895f")
    nodes = []
    arrows = []
    for index, cluster in enumerate(order):
        x = 90 + index * 160
        nodes.append(
            f'<rect x="{x}" y="105" width="110" height="70" rx="4" '
            f'fill="{colors[index]}" opacity="0.88"/>'
            f'<text x="{x + 55}" y="136" text-anchor="middle" '
            f'font-family="sans-serif" font-size="14" fill="white">stage {index}</text>'
            f'<text x="{x + 55}" y="158" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12" fill="white">cluster {cluster}</text>'
        )
        if index < 3:
            count = metrics["adjacent_transition_counts"][index]
            arrows.append(
                f'<line x1="{x + 112}" y1="140" x2="{x + 154}" y2="140" '
                'stroke="#172b3a" stroke-width="2"/>'
                f'<text x="{x + 133}" y="128" text-anchor="middle" '
                f'font-family="sans-serif" font-size="11">{count}</text>'
            )
    status = "PASS" if passed else "FAIL"
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="280" '
        'viewBox="0 0 760 280">'
        '<rect width="100%" height="100%" fill="#fbfcfd"/>'
        '<text x="28" y="34" font-family="sans-serif" font-size="20" '
        'fill="#172b3a">Causal ordered-cluster fresh audit</text>'
        f'<text x="28" y="63" font-family="sans-serif" font-size="14" '
        f'fill="{colors[3] if passed else colors[2]}">{status}</text>'
        f'{"".join(arrows)}{"".join(nodes)}'
        f'<text x="90" y="224" font-family="sans-serif" font-size="13">'
        f'accuracy {metrics["accuracy"]:.3f} | worst recall '
        f'{min(metrics["recall_by_stage"].values()):.3f} | false goal '
        f'{metrics["non_goal_false_goal_sequence_rate"]:.3f}</text>'
        '</svg>'
    )
    path.write_text(svg, encoding="utf-8")


def evaluate_causal_decoder(
    calibration_assignments_path: Path,
    fresh_dataset_path: Path,
    cluster_metrics_path: Path,
    cluster_assignments_path: Path,
    output_dir: Path,
    *,
    model_id: str = MODEL_ID,
    batch_size: int = 64,
    device: str = "cuda:0",
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(calibration_assignments_path) as calibration:
        calibration_clusters = calibration["occurrence_clusters"].astype(np.int64)
        calibration_offsets = calibration["sequence_offsets"].astype(np.int64)
    calibration_sequences = [
        calibration_clusters[start:stop].tolist()
        for start, stop in zip(calibration_offsets[:-1], calibration_offsets[1:])
    ]
    order_audit = infer_cluster_order(calibration_sequences)
    cluster_order = tuple(order_audit["selected_cluster_order"])
    with np.load(fresh_dataset_path) as data:
        frames = data["state_frames"].astype(np.uint8)
        state_stages = data["state_stages"].astype(np.int64)
        offsets = data["sequence_offsets"].astype(np.int64)
        state_indices = data["sequence_state_indices"].astype(np.int64)
        target_stages = data["sequence_target_stages"].astype(np.int64)
        native_success = data["sequence_native_success"].astype(np.bool_)
        generation_groups = data["sequence_generation_groups"].astype(np.int64)
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
    state_clusters = nearest_centers(embeddings, centers)
    occurrence_clusters = state_clusters[state_indices]
    occurrence_stages = state_stages[state_indices]
    raw_metrics = classification_metrics(
        occurrence_clusters,
        occurrence_stages,
        source_method["cluster_to_stage"],
    )
    raw_predictions = raw_metrics.pop("predictions")
    raw_sequences = [
        occurrence_clusters[start:stop].tolist()
        for start, stop in zip(offsets[:-1], offsets[1:])
    ]
    oracle_sequences = [
        occurrence_stages[start:stop].tolist()
        for start, stop in zip(offsets[:-1], offsets[1:])
    ]
    decoded_sequences = [causal_decode(sequence, cluster_order) for sequence in raw_sequences]
    decoded_metrics = decoded_stage_metrics(
        decoded_sequences,
        oracle_sequences,
        target_stages,
        native_success,
        generation_groups,
    )
    gate = {
        "minimum_accuracy": 0.85,
        "minimum_recall_per_stage": 0.75,
        "minimum_accuracy_per_group": 0.80,
        "minimum_final_state_rate_per_target": 0.80,
        "minimum_goal_native_terminal_recall": 0.80,
        "maximum_non_goal_false_goal_sequence_rate": 0.05,
        "minimum_adjacent_transition_count": 25,
        "require_all_resets_root": True,
        "require_all_stages_nonempty": True,
    }
    fresh_data_metrics = json.loads(
        (fresh_dataset_path.parent / "metrics.json").read_text(encoding="utf-8")
    )
    passed = bool(
        fresh_data_metrics["data_gate_passed"]
        and decoded_metrics["accuracy"] >= gate["minimum_accuracy"]
        and min(decoded_metrics["recall_by_stage"].values())
        >= gate["minimum_recall_per_stage"]
        and min(
            row["accuracy"]
            for row in decoded_metrics["by_generation_group"].values()
        )
        >= gate["minimum_accuracy_per_group"]
        and min(decoded_metrics["final_state_rate_by_target"].values())
        >= gate["minimum_final_state_rate_per_target"]
        and decoded_metrics["goal_native_terminal_recall"]
        >= gate["minimum_goal_native_terminal_recall"]
        and decoded_metrics["non_goal_false_goal_sequence_rate"]
        <= gate["maximum_non_goal_false_goal_sequence_rate"]
        and min(decoded_metrics["adjacent_transition_counts"])
        >= gate["minimum_adjacent_transition_count"]
        and decoded_metrics["all_resets_root"]
        and decoded_metrics["all_stages_nonempty"]
    )
    decoded_flat = np.asarray(
        [value for sequence in decoded_sequences for value in sequence],
        dtype=np.int8,
    )
    output = {
        "calibration_assignments": str(calibration_assignments_path.resolve()),
        "fresh_dataset": str(fresh_dataset_path.resolve()),
        "cluster_metrics": str(cluster_metrics_path.resolve()),
        "cluster_assignments": str(cluster_assignments_path.resolve()),
        "model_id": model_id,
        "device": device,
        "gpu_name": gpu_name,
        "versions": versions,
        "elapsed_seconds": elapsed,
        "fit_or_refit_visual_centers_performed": False,
        "order_calibration": order_audit,
        "gate": gate,
        "fresh_data_gate_passed": bool(fresh_data_metrics["data_gate_passed"]),
        "raw_hard_cluster_diagnostic": raw_metrics,
        "causal_decoded_metrics": decoded_metrics,
        "causal_decoder_gate_passed": passed,
    }
    assignment_path = output_dir / "causal_decoder_assignments.npz"
    np.savez_compressed(
        assignment_path,
        state_embeddings=embeddings,
        state_clusters=state_clusters,
        occurrence_clusters=occurrence_clusters,
        raw_stage_predictions=raw_predictions,
        causal_decoded_stages=decoded_flat,
        sequence_offsets=offsets,
        cluster_order=np.asarray(cluster_order, dtype=np.int8),
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    chart_path = output_dir / "causal_decoder_audit.svg"
    _write_chart(chart_path, list(cluster_order), decoded_metrics, passed)
    output["metrics"] = str(metrics_path.resolve())
    output["assignments"] = str(assignment_path.resolve())
    output["chart"] = str(chart_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-assignments", type=Path, required=True)
    parser.add_argument("--fresh-dataset", type=Path, required=True)
    parser.add_argument("--cluster-metrics", type=Path, required=True)
    parser.add_argument("--cluster-assignments", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/minigrid_doorkey_visual"
    ) / f"doorkey5_causal_decoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = evaluate_causal_decoder(
        args.calibration_assignments,
        args.fresh_dataset,
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
                "order_calibration": output["order_calibration"],
                "raw_hard_cluster_diagnostic": output[
                    "raw_hard_cluster_diagnostic"
                ],
                "causal_decoded_metrics": output["causal_decoded_metrics"],
                "causal_decoder_gate_passed": output[
                    "causal_decoder_gate_passed"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
