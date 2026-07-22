"""Cache frozen DINOv2 current-frame embeddings for GoToObject RGB data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from skill_discovery.encode_minigrid_doorkey_dinov2 import MODEL_ID
from skill_discovery.evaluate_minigrid_doorkey_visual_sequence_graph import _encode_frames


def encode_gotoobject_dataset(
    dataset_path: Path,
    output_dir: Path,
    *,
    model_id: str = MODEL_ID,
    batch_size: int = 64,
    device: str = "cuda:0",
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(dataset_path) as data:
        frames = data["frames"].astype(np.uint8)
        frame_hashes = data["frame_hashes"]
        stages = data["stages"].astype(np.int8)
        splits = data["splits"].astype(np.int8)
    if frames.ndim != 4 or frames.shape[1:] != (192, 192, 3):
        raise ValueError("GoToObject DINO input must contain 192x192 RGB frames")
    embeddings, elapsed, gpu_name, versions = _encode_frames(
        frames,
        model_id=model_id,
        batch_size=batch_size,
        device=device,
    )
    embedding_path = output_dir / "gotoobject_dinov2_embeddings.npz"
    np.savez_compressed(
        embedding_path,
        embeddings=embeddings,
        frame_hashes=frame_hashes,
        stages=stages,
        splits=splits,
    )
    output = {
        "dataset": str(dataset_path.resolve()),
        "model_id": model_id,
        "device": device,
        "gpu_name": gpu_name,
        "versions": versions,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "frame_count": len(frames),
        "embedding_shape": list(embeddings.shape),
        "embeddings": str(embedding_path.resolve()),
    }
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
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/minigrid_gotoobject_visual"
    ) / f"gotoobject_dinov2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = encode_gotoobject_dataset(
        args.dataset,
        output_dir,
        model_id=args.model_id,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
