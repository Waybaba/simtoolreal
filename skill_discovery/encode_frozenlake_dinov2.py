"""Cache DINOv2-small embeddings for the FrozenLake visual probe."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


MODEL_ID = "facebook/dinov2-small"


def compose_trajectory_embeddings(
    frame_embeddings: np.ndarray,
    trajectory_count: int,
) -> np.ndarray:
    if frame_embeddings.shape[0] != trajectory_count * 3:
        raise ValueError("exactly three frame embeddings are required per trajectory")
    frame_norms = np.linalg.norm(frame_embeddings, axis=1, keepdims=True)
    normalized_frames = frame_embeddings / np.maximum(frame_norms, 1.0e-8)
    trajectories = normalized_frames.reshape(trajectory_count, -1)
    trajectory_norms = np.linalg.norm(trajectories, axis=1, keepdims=True)
    return trajectories / np.maximum(trajectory_norms, 1.0e-8)


def _write_comparison_chart(
    path: Path,
    dinov2_knn: float,
    dinov2_triplet: float,
) -> None:
    names = ("raw pixels", "random projection", "DINOv2-small", "oracle")
    knn = (0.37109375, 0.44140625, dinov2_knn, 1.0)
    triplet = (0.37109375, 0.44140625, dinov2_triplet, 1.0)
    colors = ("#68757d", "#d17031", "#2875a4", "#2b895f")
    width, height = 880, 420
    left, top, chart_width, chart_height = 70, 55, 740, 290
    group_width = chart_width / len(names)
    bars = []
    labels = []
    for index, name in enumerate(names):
        for offset, value in enumerate((knn[index], triplet[index])):
            x = left + index * group_width + 35 + offset * 38
            bar_height = chart_height * value
            y = top + chart_height - bar_height
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="30" '
                f'height="{bar_height:.1f}" fill="{colors[index]}" '
                f'opacity="{1.0 if offset == 0 else 0.45}"/>'
            )
        labels.append(
            f'<text x="{left + index * group_width + 17:.1f}" '
            f'y="{top + chart_height + 24}" font-family="sans-serif" '
            f'font-size="11">{name}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fbfcfd"/>
<text x="30" y="30" font-family="sans-serif" font-size="19" fill="#172b3a">FrozenLake audit-style nuisance invariance</text>
<text x="580" y="30" font-family="sans-serif" font-size="11">solid: 1-NN | light: triplet</text>
<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#83919a"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#83919a"/>
<text x="28" y="{top + 5}" font-family="sans-serif" font-size="12">1.0</text>
<text x="28" y="{top + chart_height + 5}" font-family="sans-serif" font-size="12">0.0</text>
{''.join(bars)}
{''.join(labels)}
</svg>'''
    path.write_text(svg, encoding="utf-8")


def encode_dataset(
    dataset_path: Path,
    output_dir: Path,
    *,
    model_id: str = MODEL_ID,
    batch_size: int = 64,
    device: str = "cuda:0",
    audit_nuisance: bool = True,
) -> dict[str, object]:
    import torch
    import transformers
    from transformers import AutoImageProcessor, AutoModel

    from skill_discovery.evaluate_frozenlake_visual_metrics import (
        _cross_seed_metrics,
        _representative_metrics,
        apply_audit_nuisance,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    data = np.load(dataset_path)
    outcomes = data["outcomes"].astype(np.int64)
    split = data["split"].astype(np.int64)
    transformed_frames = (
        apply_audit_nuisance(
            data["frames"],
            data["generation_seeds"],
            split,
        )
        if audit_nuisance
        else data["frames"]
    )
    flat_frames = transformed_frames.reshape(-1, *transformed_frames.shape[2:])

    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
    model = AutoModel.from_pretrained(model_id)
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    use_half = torch_device.type == "cuda"
    if use_half:
        model.half()

    encoded_batches = []
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
            embeddings = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
            encoded_batches.append(embeddings)
    elapsed = time.monotonic() - started
    frame_embeddings = np.concatenate(encoded_batches).astype(np.float32)
    trajectory_embeddings = compose_trajectory_embeddings(
        frame_embeddings,
        len(outcomes),
    ).astype(np.float32)

    train_indices = np.flatnonzero(split == 0)
    audit_indices = np.flatnonzero(split == 1)
    cross_seed = _cross_seed_metrics(
        trajectory_embeddings[train_indices],
        outcomes[train_indices],
        trajectory_embeddings[audit_indices],
        outcomes[audit_indices],
    )
    representatives = _representative_metrics(
        trajectory_embeddings,
        outcomes,
        train_indices,
        seed=17_007,
    )
    embedding_path = output_dir / "dinov2_embeddings.npz"
    np.savez_compressed(
        embedding_path,
        frame_embeddings=frame_embeddings,
        trajectory_embeddings=trajectory_embeddings,
        outcomes=outcomes,
        split=split,
        generation_seeds=data["generation_seeds"],
    )
    raw_baseline = 0.37109375
    gate_threshold = raw_baseline + 0.15
    output = {
        "dataset": str(dataset_path.resolve()),
        "model_id": model_id,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
        "device": str(torch_device),
        "gpu_name": torch.cuda.get_device_name(torch_device)
        if torch_device.type == "cuda"
        else None,
        "batch_size": batch_size,
        "audit_nuisance": audit_nuisance,
        "elapsed_seconds": elapsed,
        "frame_embedding_shape": list(frame_embeddings.shape),
        "trajectory_embedding_shape": list(trajectory_embeddings.shape),
        "cross_seed": cross_seed,
        "representatives": representatives,
        "raw_nuisance_baseline": raw_baseline,
        "gate_threshold": gate_threshold,
        "visual_gate_passed": bool(
            cross_seed["knn_accuracy"] >= gate_threshold
            and cross_seed["nuisance_triplet_accuracy"] >= gate_threshold
        ),
        "embeddings": str(embedding_path.resolve()),
    }
    chart_path = output_dir / "dinov2_comparison.svg"
    _write_comparison_chart(
        chart_path,
        cross_seed["knn_accuracy"],
        cross_seed["nuisance_triplet_accuracy"],
    )
    output["chart"] = str(chart_path.resolve())
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["metrics"] = str(metrics_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-audit-nuisance", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = encode_dataset(
        args.dataset,
        args.output_dir,
        model_id=args.model_id,
        batch_size=args.batch_size,
        device=args.device,
        audit_nuisance=not args.no_audit_nuisance,
    )
    print(
        json.dumps(
            {
                "metrics": output["metrics"],
                "embeddings": output["embeddings"],
                "chart": output["chart"],
                "model_id": output["model_id"],
                "device": output["device"],
                "gpu_name": output["gpu_name"],
                "elapsed_seconds": output["elapsed_seconds"],
                "knn_accuracy": output["cross_seed"]["knn_accuracy"],
                "nuisance_triplet_accuracy": output["cross_seed"][
                    "nuisance_triplet_accuracy"
                ],
                "rare_recall": output["representatives"][
                    "rare_safe_goal_recall"
                ],
                "gate_threshold": output["gate_threshold"],
                "visual_gate_passed": output["visual_gate_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
