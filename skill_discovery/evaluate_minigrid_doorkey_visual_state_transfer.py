"""Evaluate frozen DoorKey visual clusters on real policy-state frames."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from skill_discovery.cluster_minigrid_doorkey_visual_embeddings import (
    DINOV2_SELECTION_PRIORITY,
)
from skill_discovery.encode_minigrid_doorkey_dinov2 import MODEL_ID
from skill_discovery.minigrid_doorkey import DOORKEY_STAGES


def _normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1.0e-8)


def build_pair_dinov2_representations(
    frame_embeddings: np.ndarray,
) -> dict[str, np.ndarray]:
    if frame_embeddings.ndim != 3 or frame_embeddings.shape[1] != 2:
        raise ValueError("expected sample x reset/current x embedding")
    frames = frame_embeddings / np.maximum(
        np.linalg.norm(frame_embeddings, axis=2, keepdims=True),
        1.0e-8,
    )
    start = frames[:, 0]
    current = frames[:, 1]
    return {
        "dinov2_current": current.astype(np.float32),
        "dinov2_start_current": _normalize(
            np.concatenate((start, current), axis=1)
        ).astype(np.float32),
        "dinov2_temporal_delta": _normalize(current - start).astype(np.float32),
    }


def build_pair_raw_representations(frames: np.ndarray) -> dict[str, np.ndarray]:
    if frames.ndim != 5 or frames.shape[1] != 2 or frames.shape[-1] != 3:
        raise ValueError("expected sample x reset/current RGB frames")
    y = np.linspace(0, frames.shape[2] - 1, 32).astype(np.int64)
    x = np.linspace(0, frames.shape[3] - 1, 32).astype(np.int64)
    resized = frames[:, :, y][:, :, :, x].astype(np.float32) / 255.0
    start = resized[:, 0].reshape(len(frames), -1)
    current = resized[:, 1].reshape(len(frames), -1)
    return {
        "raw_current": _normalize(current).astype(np.float32),
        "raw_temporal_delta": _normalize(current - start).astype(np.float32),
    }


def nearest_centers(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if features.ndim != 2 or centers.ndim != 2:
        raise ValueError("features and centers must be matrices")
    if features.shape[1] != centers.shape[1]:
        raise ValueError("feature and center dimensions do not match")
    distances = np.sum(
        (features[:, None, :] - centers[None, :, :]) ** 2,
        axis=2,
    )
    return np.argmin(distances, axis=1).astype(np.int64)


def transfer_metrics(
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
    confusion = np.zeros((4, 4), dtype=np.int64)
    for expected, predicted in zip(stages, predictions):
        confusion[int(expected), int(predicted)] += 1
    recalls = {
        name: float(confusion[index, index] / confusion[index].sum())
        for index, name in enumerate(DOORKEY_STAGES)
    }
    group_metrics = {}
    for group in sorted(set(groups.tolist())):
        local = groups == group
        group_metrics[str(group)] = {
            "accuracy": float(np.mean(predictions[local] == stages[local])),
            "recall_by_stage": {
                name: float(
                    np.mean(
                        predictions[local][stages[local] == index] == index
                    )
                )
                for index, name in enumerate(DOORKEY_STAGES)
            },
        }
    cluster_sizes = np.bincount(clusters, minlength=4)
    return {
        "accuracy": float(np.mean(predictions == stages)),
        "recall_by_stage": recalls,
        "confusion": confusion.tolist(),
        "predicted_cluster_sizes": cluster_sizes.tolist(),
        "all_clusters_nonempty": bool(np.all(cluster_sizes > 0)),
        "by_generation_group": group_metrics,
        "predictions": predictions,
    }


def _write_chart(path: Path, methods: dict[str, dict[str, object]]) -> None:
    names = list(methods)
    colors = ("#68757d", "#d17031", "#2875a4", "#2b895f", "#955f9a")
    width, height = 980, 430
    left, top, chart_width, chart_height = 70, 55, 840, 290
    group_width = chart_width / len(names)
    elements = []
    for index, name in enumerate(names):
        values = (
            methods[name]["accuracy"],
            min(methods[name]["recall_by_stage"].values()),
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
        'fill="#172b3a">Frozen visual centers on policy states</text>'
        '<text x="700" y="30" font-family="sans-serif" font-size="11">'
        'solid: accuracy | light: worst recall</text>'
        f'<line x1="{left}" y1="{top + chart_height}" '
        f'x2="{left + chart_width}" y2="{top + chart_height}" '
        'stroke="#83919a"/>'
        f'{"".join(elements)}</svg>'
    )
    path.write_text(svg, encoding="utf-8")


def evaluate_transfer(
    dataset_path: Path,
    cluster_metrics_path: Path,
    cluster_assignments_path: Path,
    output_dir: Path,
    *,
    model_id: str = MODEL_ID,
    batch_size: int = 64,
    device: str = "cuda:0",
    selected_representation: str = "dinov2_temporal_delta",
) -> dict[str, object]:
    import torch
    import transformers
    from transformers import AutoImageProcessor, AutoModel

    if selected_representation not in DINOV2_SELECTION_PRIORITY:
        raise ValueError("selected representation must be a frozen DINO method")
    output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(dataset_path) as data:
        frames = data["frames"].astype(np.uint8)
        stages = data["stages"].astype(np.int64)
        groups = data["generation_groups"].astype(np.int64)
        env_seeds = data["env_seeds"].astype(np.int64)
    source_metrics = json.loads(cluster_metrics_path.read_text(encoding="utf-8"))
    with np.load(cluster_assignments_path) as source_assignments:
        frozen_centers = {
            name: source_assignments[f"{name}_centers"].astype(np.float32)
            for name in (
                "raw_current",
                "raw_temporal_delta",
                *DINOV2_SELECTION_PRIORITY,
            )
        }
    flat_frames = frames.reshape(-1, *frames.shape[2:])
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
        for start in range(0, len(flat_frames), batch_size):
            stop = min(start + batch_size, len(flat_frames))
            inputs = processor(
                images=[image for image in flat_frames[start:stop]],
                return_tensors="pt",
            )
            pixel_values = inputs["pixel_values"].to(torch_device)
            if use_half:
                pixel_values = pixel_values.half()
            outputs = model(pixel_values=pixel_values)
            batches.append(
                outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
            )
    elapsed = time.monotonic() - started
    frame_embeddings = np.concatenate(batches).astype(np.float32).reshape(
        len(frames),
        2,
        -1,
    )
    features = {
        **build_pair_raw_representations(frames),
        **build_pair_dinov2_representations(frame_embeddings),
    }
    methods = {}
    assignments = {}
    for name, representation in features.items():
        clusters = nearest_centers(representation, frozen_centers[name])
        row = transfer_metrics(
            clusters,
            stages,
            groups,
            source_metrics["methods"][name]["cluster_to_stage"],
        )
        predictions = row.pop("predictions")
        row["frozen_transfer_gate_passed"] = bool(
            name == selected_representation
            and row["accuracy"] >= 0.85
            and min(row["recall_by_stage"].values()) >= 0.75
            and min(
                group["accuracy"]
                for group in row["by_generation_group"].values()
            )
            >= 0.80
            and row["all_clusters_nonempty"]
        )
        methods[name] = row
        assignments[f"{name}_clusters"] = clusters
        assignments[f"{name}_predictions"] = predictions
    selected = selected_representation
    output = {
        "dataset": str(dataset_path.resolve()),
        "cluster_metrics": str(cluster_metrics_path.resolve()),
        "cluster_assignments": str(cluster_assignments_path.resolve()),
        "model_id": model_id,
        "device": str(torch_device),
        "gpu_name": torch.cuda.get_device_name(torch_device)
        if torch_device.type == "cuda"
        else None,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "sample_count": len(frames),
        "frame_count": len(flat_frames),
        "generation_groups": sorted(set(groups.tolist())),
        "unique_env_seeds": int(len(set(env_seeds.tolist()))),
        "fit_or_refit_performed": False,
        "selected_representation": selected,
        "gate": {
            "minimum_accuracy": 0.85,
            "minimum_recall_per_stage": 0.75,
            "minimum_accuracy_per_group": 0.80,
            "require_all_clusters_nonempty": True,
        },
        "methods": methods,
        "state_distribution_gate_passed": bool(
            methods[selected]["frozen_transfer_gate_passed"]
        ),
    }
    np.savez_compressed(
        output_dir / "transfer_assignments.npz",
        frame_embeddings=frame_embeddings,
        **assignments,
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    chart_path = output_dir / "state_transfer_comparison.svg"
    _write_chart(chart_path, methods)
    output["metrics"] = str(metrics_path.resolve())
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
    parser.add_argument(
        "--selected-representation",
        choices=DINOV2_SELECTION_PRIORITY,
        default="dinov2_temporal_delta",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = evaluate_transfer(
        args.dataset,
        args.cluster_metrics,
        args.cluster_assignments,
        args.output_dir,
        model_id=args.model_id,
        batch_size=args.batch_size,
        device=args.device,
        selected_representation=args.selected_representation,
    )
    print(
        json.dumps(
            {
                "metrics": output["metrics"],
                "chart": output["chart"],
                "elapsed_seconds": output["elapsed_seconds"],
                "methods": {
                    name: {
                        "accuracy": row["accuracy"],
                        "recall": row["recall_by_stage"],
                        "by_group": row["by_generation_group"],
                        "passed": row["frozen_transfer_gate_passed"],
                    }
                    for name, row in output["methods"].items()
                },
                "state_distribution_gate_passed": output[
                    "state_distribution_gate_passed"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
