"""Stress frozen FrozenLake visual clusters across unseen renderer styles."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from skill_discovery.calibrate_frozenlake_visual_embeddings import (
    _aligned_predictions,
    build_self_reference_representations,
)
from skill_discovery.evaluate_frozenlake_visual_metrics import apply_audit_nuisance
from skill_discovery.frozenlake import FROZENLAKE_OUTCOMES
from skill_discovery.generate_point_cup_dataset import _write_png


STYLE_SEEDS = tuple(range(107, 258, 10))


def select_balanced_sources(
    outcomes: np.ndarray,
    split: np.ndarray,
    *,
    count_per_outcome: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for outcome in range(len(FROZENLAKE_OUTCOMES)):
        candidates = np.flatnonzero((split == 1) & (outcomes == outcome))
        if len(candidates) < count_per_outcome:
            raise ValueError("audit split does not contain enough balanced sources")
        selected.extend(
            rng.choice(candidates, size=count_per_outcome, replace=False).tolist()
        )
    return np.asarray(selected, dtype=np.int64)


def build_styled_trajectories(
    source_frames: np.ndarray,
    style_seeds: tuple[int, ...],
) -> np.ndarray:
    batches = []
    for seed in style_seeds:
        count = len(source_frames)
        batches.append(
            apply_audit_nuisance(
                source_frames,
                np.full(count, seed, dtype=np.int64),
                np.ones(count, dtype=np.int8),
            )
        )
    return np.stack(batches)


def evaluate_predictions(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    style_seeds: tuple[int, ...],
) -> dict[str, object]:
    if predictions.shape != (len(style_seeds), len(outcomes)):
        raise ValueError("prediction matrix does not match styles and outcomes")
    aggregate = np.zeros((3, 3), dtype=np.int64)
    per_style = {}
    passing_styles = 0
    for row, seed in enumerate(style_seeds):
        confusion = np.zeros((3, 3), dtype=np.int64)
        for expected, predicted in zip(outcomes, predictions[row]):
            confusion[int(expected), int(predicted)] += 1
        aggregate += confusion
        accuracy = float(np.trace(confusion) / confusion.sum())
        passing_styles += int(accuracy >= 0.84)
        per_style[str(seed)] = {
            "accuracy": accuracy,
            "recall_by_outcome": {
                name: float(confusion[index, index] / confusion[index].sum())
                for index, name in enumerate(FROZENLAKE_OUTCOMES)
            },
            "confusion": confusion.tolist(),
        }
    recalls = {
        name: float(aggregate[index, index] / aggregate[index].sum())
        for index, name in enumerate(FROZENLAKE_OUTCOMES)
    }
    accuracy = float(np.trace(aggregate) / aggregate.sum())
    worst_seed = min(per_style, key=lambda key: per_style[key]["accuracy"])
    return {
        "accuracy": accuracy,
        "recall_by_outcome": recalls,
        "confusion": aggregate.tolist(),
        "styles_passing_0_84": passing_styles,
        "style_count": len(style_seeds),
        "worst_style_seed": int(worst_seed),
        "worst_style_accuracy": per_style[worst_seed]["accuracy"],
        "per_style": per_style,
        "gate_passed": bool(
            accuracy >= 0.90
            and recalls["safe_timeout"] >= 0.75
            and recalls["hole_terminal"] >= 0.95
            and recalls["goal_terminal"] >= 0.95
            and passing_styles >= 14
        ),
    }


def _encode_frames(
    frames: np.ndarray,
    *,
    model_id: str,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, dict[str, object]]:
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
            pixels = inputs["pixel_values"].to(torch_device)
            if use_half:
                pixels = pixels.half()
            output = model(pixel_values=pixels)
            batches.append(output.last_hidden_state[:, 0].float().cpu().numpy())
    embeddings = np.concatenate(batches).astype(np.float32)
    metadata = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": str(torch_device),
        "gpu_name": torch.cuda.get_device_name(torch_device)
        if torch_device.type == "cuda"
        else None,
        "elapsed_seconds": time.monotonic() - started,
    }
    return embeddings, metadata


def _write_style_sample(
    path: Path,
    styled: np.ndarray,
    source_outcomes: np.ndarray,
) -> None:
    tile = 64
    columns = 4
    gap = 5
    cell_height = tile * 3
    sheet = np.full(
        (
            4 * cell_height + 3 * gap,
            columns * tile + (columns - 1) * gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    for style_index in range(16):
        row, column = divmod(style_index, columns)
        y = row * (cell_height + gap)
        x = column * (tile + gap)
        for outcome in range(3):
            index = int(np.flatnonzero(source_outcomes == outcome)[0])
            sheet[y + outcome * tile : y + (outcome + 1) * tile, x : x + tile] = (
                styled[style_index, index, -1]
            )
    _write_png(path, sheet)


def _write_chart(path: Path, methods: dict[str, dict[str, object]]) -> None:
    seeds = [int(seed) for seed in next(iter(methods.values()))["per_style"]]
    width, height = 920, 430
    left, top, chart_width, chart_height = 65, 55, 790, 290
    colors = {"absolute_3frame": "#68757d", "temporal_delta": "#2b895f"}
    lines = []
    for name, metrics in methods.items():
        points = []
        for index, seed in enumerate(seeds):
            value = metrics["per_style"][str(seed)]["accuracy"]
            x = left + chart_width * index / max(len(seeds) - 1, 1)
            y = top + chart_height * (1.0 - value)
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{colors[name]}" stroke-width="2"/>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fbfcfd"/>
<text x="30" y="30" font-family="sans-serif" font-size="19" fill="#172b3a">FrozenLake 16-style stress accuracy</text>
<text x="600" y="30" font-family="sans-serif" font-size="11">gray: absolute | green: temporal delta</text>
<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#83919a"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#83919a"/>
<text x="23" y="{top + 5}" font-family="sans-serif" font-size="12">1.0</text>
<text x="23" y="{top + chart_height + 5}" font-family="sans-serif" font-size="12">0.0</text>
{''.join(lines)}
</svg>'''
    path.write_text(svg, encoding="utf-8")


def run_style_stress(
    dataset_path: Path,
    calibration_metrics_path: Path,
    calibration_clusters_path: Path,
    output_dir: Path,
    *,
    model_id: str = "facebook/dinov2-small",
    batch_size: int = 64,
    device: str = "cuda:0",
    style_seeds: tuple[int, ...] = STYLE_SEEDS,
) -> dict[str, object]:
    if len(style_seeds) != 16 or len(set(style_seeds)) != 16:
        raise ValueError("style stress requires exactly 16 distinct seeds")
    output_dir.mkdir(parents=True, exist_ok=False)
    data = np.load(dataset_path)
    outcomes = data["outcomes"].astype(np.int64)
    split = data["split"].astype(np.int64)
    source_indices = select_balanced_sources(
        outcomes,
        split,
        count_per_outcome=32,
        seed=50_507,
    )
    source_outcomes = outcomes[source_indices]
    styled = build_styled_trajectories(data["frames"][source_indices], style_seeds)
    flat_frames = styled.reshape(-1, *styled.shape[3:])
    frame_embeddings, encoder = _encode_frames(
        flat_frames,
        model_id=model_id,
        batch_size=batch_size,
        device=device,
    )
    trajectory_embeddings = frame_embeddings.reshape(len(style_seeds), 96, 3, -1)
    calibration_metrics = json.loads(calibration_metrics_path.read_text())
    calibration_clusters = np.load(calibration_clusters_path)
    methods = {}
    predictions = {}
    for name in ("absolute_3frame", "temporal_delta"):
        features = build_self_reference_representations(
            trajectory_embeddings.reshape(-1, 3, trajectory_embeddings.shape[-1])
        )[name]
        centers = calibration_clusters[f"{name}_centers"]
        distances = (
            np.sum(features * features, axis=1)[:, None]
            + np.sum(centers * centers, axis=1)[None, :]
            - 2.0 * features @ centers.T
        )
        clusters = np.argmin(distances, axis=1)
        aligned = _aligned_predictions(
            clusters,
            calibration_metrics["methods"][name]["cluster_to_outcome"],
        ).reshape(len(style_seeds), -1)
        methods[name] = evaluate_predictions(aligned, source_outcomes, style_seeds)
        predictions[name] = aligned

    np.savez_compressed(
        output_dir / "stress_embeddings.npz",
        frame_embeddings=frame_embeddings,
        source_indices=source_indices,
        style_seeds=np.asarray(style_seeds),
        **{f"{name}_predictions": value for name, value in predictions.items()},
    )
    style_image = output_dir / "style_sample.png"
    _write_style_sample(style_image, styled, source_outcomes)
    chart = output_dir / "style_accuracy.svg"
    _write_chart(chart, methods)
    output = {
        "dataset": str(dataset_path.resolve()),
        "calibration_metrics": str(calibration_metrics_path.resolve()),
        "calibration_clusters": str(calibration_clusters_path.resolve()),
        "model_id": model_id,
        "style_seeds": list(style_seeds),
        "source_selection": {
            "seed": 50_507,
            "count_per_outcome": 32,
            "source_indices": source_indices.tolist(),
        },
        "trajectory_count": int(len(style_seeds) * len(source_indices)),
        "frame_count": int(len(flat_frames)),
        "encoder": encoder,
        "gate": {
            "minimum_accuracy": 0.90,
            "minimum_safe_recall": 0.75,
            "minimum_hole_recall": 0.95,
            "minimum_goal_recall": 0.95,
            "minimum_styles_passing_0_84": 14,
        },
        "methods": methods,
        "stress_gate_passed": bool(methods["temporal_delta"]["gate_passed"]),
        "style_image": str(style_image.resolve()),
        "chart": str(chart.resolve()),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["metrics"] = str(metrics_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("calibration_metrics", type=Path)
    parser.add_argument("calibration_clusters", type=Path)
    parser.add_argument("--model-id", default="facebook/dinov2-small")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--style-seeds", type=int, nargs=16, default=STYLE_SEEDS)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = run_style_stress(
        args.dataset,
        args.calibration_metrics,
        args.calibration_clusters,
        args.output_dir,
        model_id=args.model_id,
        batch_size=args.batch_size,
        device=args.device,
        style_seeds=tuple(args.style_seeds),
    )
    print(
        json.dumps(
            {
                "metrics": output["metrics"],
                "encoder": output["encoder"],
                "methods": {
                    name: {
                        "accuracy": row["accuracy"],
                        "recall": row["recall_by_outcome"],
                        "styles_passing": row["styles_passing_0_84"],
                        "worst_style": row["worst_style_seed"],
                        "worst_accuracy": row["worst_style_accuracy"],
                        "passed": row["gate_passed"],
                    }
                    for name, row in output["methods"].items()
                },
                "stress_gate_passed": output["stress_gate_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
