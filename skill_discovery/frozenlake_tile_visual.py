"""RGB-only tile selection and representation helpers for FrozenLake."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from skill_discovery.evaluate_frozenlake_visual_metrics import apply_audit_nuisance
from skill_discovery.frozenlake import FROZENLAKE_OUTCOMES
from skill_discovery.generate_point_cup_dataset import _write_png


def _normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.maximum(norms, 1.0e-8)


def select_top_change_tiles(
    frames: np.ndarray,
    grid_size: int,
    *,
    top_k: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if frames.ndim != 5 or frames.shape[1] != 3 or frames.shape[-1] != 3:
        raise ValueError("expected trajectory x three RGB frames")
    if frames.shape[2] != frames.shape[3] or frames.shape[2] % grid_size:
        raise ValueError("square frames must divide evenly into the grid")
    if not 0 < top_k <= grid_size * grid_size:
        raise ValueError("top_k must fit within the grid")
    cell = frames.shape[2] // grid_size
    tiled = frames.reshape(
        len(frames),
        3,
        grid_size,
        cell,
        grid_size,
        cell,
        3,
    ).transpose(0, 1, 2, 4, 3, 5, 6)
    changes = np.abs(
        tiled[:, 1:].astype(np.float32) - tiled[:, :1].astype(np.float32)
    ).mean(axis=(4, 5, 6))
    scores = changes.reshape(len(frames), 2, -1)
    selected = np.argsort(-scores, axis=2, kind="stable")[:, :, :top_k]
    selected_scores = np.take_along_axis(scores, selected, axis=2)
    total = scores.sum(axis=2)
    capture = selected_scores.sum(axis=2) / np.maximum(total, 1.0e-8)
    return selected.astype(np.int64), scores, capture


def extract_selected_tile_pairs(
    frames: np.ndarray,
    selected: np.ndarray,
    grid_size: int,
    *,
    output_size: int = 64,
) -> np.ndarray:
    if selected.shape[:2] != (len(frames), 2):
        raise ValueError("selected indices must cover two transitions")
    cell = frames.shape[2] // grid_size
    y_indices = np.linspace(0, cell - 1, output_size).astype(np.int64)
    x_indices = np.linspace(0, cell - 1, output_size).astype(np.int64)
    output = np.empty(
        (*selected.shape, 2, output_size, output_size, 3),
        dtype=np.uint8,
    )
    for trajectory in range(len(frames)):
        for transition in range(2):
            for rank, tile_index in enumerate(selected[trajectory, transition]):
                row, column = divmod(int(tile_index), grid_size)
                y = row * cell
                x = column * cell
                for pair_index, frame_index in enumerate((0, transition + 1)):
                    tile = frames[
                        trajectory,
                        frame_index,
                        y : y + cell,
                        x : x + cell,
                    ]
                    output[trajectory, transition, rank, pair_index] = tile[
                        y_indices
                    ][:, x_indices]
    return output


def compose_tile_delta_embeddings(pair_embeddings: np.ndarray) -> np.ndarray:
    if pair_embeddings.ndim != 5 or pair_embeddings.shape[1:4] != (2, 4, 2):
        raise ValueError("expected trajectory x two transitions x four pairs")
    pairs = _normalize(pair_embeddings.astype(np.float32))
    deltas = _normalize(pairs[:, :, :, 1] - pairs[:, :, :, 0])
    pooled = np.concatenate(
        (
            deltas.mean(axis=2),
            np.abs(deltas).mean(axis=2),
            np.abs(deltas).max(axis=2),
        ),
        axis=2,
    )
    return _normalize(pooled.reshape(len(pooled), -1)).astype(np.float32)


def selection_metrics(
    scores: np.ndarray,
    capture: np.ndarray,
) -> dict[str, object]:
    positive = scores.max(axis=2) > 0
    medians = np.median(capture, axis=0)
    positive_fraction = np.mean(positive, axis=0)
    return {
        "median_capture_by_transition": {
            "start_to_middle": float(medians[0]),
            "start_to_final": float(medians[1]),
        },
        "positive_top_score_fraction": {
            "start_to_middle": float(positive_fraction[0]),
            "start_to_final": float(positive_fraction[1]),
        },
        "gate_passed": bool(
            np.all(medians >= 0.50) and np.all(positive_fraction >= 0.95)
        ),
    }


def _resize(image: np.ndarray, size: int) -> np.ndarray:
    y = np.linspace(0, image.shape[0] - 1, size).astype(np.int64)
    x = np.linspace(0, image.shape[1] - 1, size).astype(np.int64)
    return image[y][:, x]


def _outlined_frame(
    frame: np.ndarray,
    tile_indices: np.ndarray,
    grid_size: int,
    color: tuple[int, int, int],
) -> np.ndarray:
    size = 96
    output = _resize(frame, size).copy()
    cell = size // grid_size
    for tile_index in tile_indices:
        row, column = divmod(int(tile_index), grid_size)
        y0, x0 = row * cell, column * cell
        y1, x1 = (row + 1) * cell - 1, (column + 1) * cell - 1
        output[y0 : y0 + 2, x0 : x1 + 1] = color
        output[y1 - 1 : y1 + 1, x0 : x1 + 1] = color
        output[y0 : y1 + 1, x0 : x0 + 2] = color
        output[y0 : y1 + 1, x1 - 1 : x1 + 1] = color
    return output


def _preview_indices(
    outcomes: np.ndarray,
    split: np.ndarray,
) -> np.ndarray:
    rng = np.random.default_rng(80_805)
    selected = []
    for outcome in range(3):
        candidates = np.flatnonzero((split == 1) & (outcomes == outcome))
        selected.extend(rng.choice(candidates, size=5, replace=False).tolist())
    return np.asarray(selected, dtype=np.int64)


def write_selection_preview(
    path: Path,
    frames: np.ndarray,
    outcomes: np.ndarray,
    selected_tiles: np.ndarray,
    preview_indices: np.ndarray,
    grid_size: int,
) -> None:
    full_size = 96
    tile_size = 64
    gap = 4
    marker = 8
    width = marker + 3 * full_size + 4 * tile_size + 6 * gap
    row_height = full_size
    sheet = np.full(
        (len(preview_indices) * row_height + (len(preview_indices) - 1) * gap, width, 3),
        255,
        dtype=np.uint8,
    )
    marker_colors = ((48, 116, 173), (198, 72, 58), (42, 137, 94))
    for row, trajectory_index in enumerate(preview_indices):
        trajectory_index = int(trajectory_index)
        y = row * (row_height + gap)
        sheet[y : y + row_height, :marker] = marker_colors[int(outcomes[trajectory_index])]
        x = marker
        sheet[y : y + full_size, x : x + full_size] = _resize(
            frames[trajectory_index, 0], full_size
        )
        x += full_size + gap
        sheet[y : y + full_size, x : x + full_size] = _outlined_frame(
            frames[trajectory_index, 1],
            selected_tiles[trajectory_index, 0],
            grid_size,
            (230, 145, 32),
        )
        x += full_size + gap
        sheet[y : y + full_size, x : x + full_size] = _outlined_frame(
            frames[trajectory_index, 2],
            selected_tiles[trajectory_index, 1],
            grid_size,
            (198, 72, 58),
        )
        x += full_size + gap
        final_pairs = extract_selected_tile_pairs(
            frames[trajectory_index : trajectory_index + 1],
            selected_tiles[trajectory_index : trajectory_index + 1],
            grid_size,
            output_size=tile_size,
        )[0, 1, :, 1]
        for tile in final_pairs:
            sheet[y : y + tile_size, x : x + tile_size] = tile
            x += tile_size + gap
    _write_png(path, sheet)


def audit_selection(
    dataset_path: Path,
    output_dir: Path,
    *,
    grid_size: int,
    audit_nuisance: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    data = np.load(dataset_path)
    frames = data["frames"]
    if audit_nuisance:
        frames = apply_audit_nuisance(
            frames,
            data["generation_seeds"],
            data["split"],
        )
    selected, scores, capture = select_top_change_tiles(frames, grid_size)
    metrics = selection_metrics(scores, capture)
    preview_indices = _preview_indices(data["outcomes"], data["split"])
    image_path = output_dir / "tile_selection_preview.png"
    write_selection_preview(
        image_path,
        frames,
        data["outcomes"],
        selected,
        preview_indices,
        grid_size,
    )
    np.savez_compressed(
        output_dir / "tile_selection.npz",
        selected_tiles=selected,
        change_scores=scores,
        capture_ratio=capture,
        preview_indices=preview_indices,
    )
    output = {
        "dataset": str(dataset_path.resolve()),
        "grid_size": grid_size,
        "audit_nuisance": audit_nuisance,
        "trajectory_count": len(frames),
        **metrics,
        "image": str(image_path.resolve()),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["metrics"] = str(metrics_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--audit-nuisance", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = audit_selection(
        args.dataset,
        args.output_dir,
        grid_size=args.grid_size,
        audit_nuisance=args.audit_nuisance,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
