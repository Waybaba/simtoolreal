"""Encode RGB-selected FrozenLake tile changes with frozen DINOv2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from skill_discovery.evaluate_frozenlake_visual_metrics import apply_audit_nuisance
from skill_discovery.frozenlake_tile_visual import (
    compose_tile_delta_embeddings,
    extract_selected_tile_pairs,
    select_top_change_tiles,
    selection_metrics,
)
from skill_discovery.stress_frozenlake_visual_styles import (
    _encode_frames,
    build_styled_trajectories,
    select_balanced_sources,
)


def prepare_tile_trajectories(
    data: np.lib.npyio.NpzFile,
    *,
    audit_nuisance: bool,
    style_seeds: tuple[int, ...] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    outcomes = data["outcomes"].astype(np.int64)
    split = data["split"].astype(np.int64)
    generation_seeds = data["generation_seeds"].astype(np.int64)
    frames = data["frames"]
    metadata: dict[str, object] = {"source_indices": None, "style_seeds": None}
    if style_seeds is not None:
        if len(style_seeds) != 16 or len(set(style_seeds)) != 16:
            raise ValueError("style mode requires exactly 16 distinct seeds")
        source_indices = select_balanced_sources(
            outcomes,
            split,
            count_per_outcome=32,
            seed=50_507,
        )
        styled = build_styled_trajectories(frames[source_indices], style_seeds)
        frames = styled.reshape(-1, *styled.shape[2:])
        outcomes = np.tile(outcomes[source_indices], len(style_seeds))
        generation_seeds = np.repeat(style_seeds, len(source_indices)).astype(np.int64)
        split = np.ones(len(frames), dtype=np.int8)
        metadata = {
            "source_indices": source_indices.tolist(),
            "style_seeds": list(style_seeds),
        }
    elif audit_nuisance:
        frames = apply_audit_nuisance(frames, generation_seeds, split)
    return frames, outcomes, split, generation_seeds, metadata


def encode_tile_dataset(
    dataset_path: Path,
    output_dir: Path,
    *,
    grid_size: int,
    audit_nuisance: bool = False,
    style_seeds: tuple[int, ...] | None = None,
    model_id: str = "facebook/dinov2-small",
    batch_size: int = 64,
    device: str = "cuda:0",
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    data = np.load(dataset_path)
    frames, outcomes, split, generation_seeds, source_metadata = (
        prepare_tile_trajectories(
            data,
            audit_nuisance=audit_nuisance,
            style_seeds=style_seeds,
        )
    )
    selected, scores, capture = select_top_change_tiles(frames, grid_size)
    selection = selection_metrics(scores, capture)
    if not selection["gate_passed"]:
        raise ValueError("tile selection gate failed before DINO encoding")
    pairs = extract_selected_tile_pairs(
        frames,
        selected,
        grid_size,
        output_size=64,
    )
    flat_pairs = pairs.reshape(-1, *pairs.shape[-3:])
    frame_embeddings, encoder = _encode_frames(
        flat_pairs,
        model_id=model_id,
        batch_size=batch_size,
        device=device,
    )
    pair_embeddings = frame_embeddings.reshape(
        len(frames),
        2,
        4,
        2,
        -1,
    )
    trajectory_embeddings = compose_tile_delta_embeddings(pair_embeddings)
    cache_path = output_dir / "tile_dinov2_embeddings.npz"
    np.savez_compressed(
        cache_path,
        pair_embeddings=pair_embeddings.astype(np.float16),
        trajectory_embeddings=trajectory_embeddings,
        selected_tiles=selected,
        outcomes=outcomes,
        split=split,
        generation_seeds=generation_seeds,
    )
    output = {
        "dataset": str(dataset_path.resolve()),
        "grid_size": grid_size,
        "audit_nuisance": audit_nuisance,
        **source_metadata,
        "trajectory_count": len(frames),
        "tile_pair_image_count": len(flat_pairs),
        "model_id": model_id,
        "encoder": encoder,
        "selection": selection,
        "pair_embedding_shape": list(pair_embeddings.shape),
        "trajectory_embedding_shape": list(trajectory_embeddings.shape),
        "cache": str(cache_path.resolve()),
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
    parser.add_argument("--style-seeds", type=int, nargs=16)
    parser.add_argument("--model-id", default="facebook/dinov2-small")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = encode_tile_dataset(
        args.dataset,
        args.output_dir,
        grid_size=args.grid_size,
        audit_nuisance=args.audit_nuisance,
        style_seeds=tuple(args.style_seeds) if args.style_seeds else None,
        model_id=args.model_id,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
