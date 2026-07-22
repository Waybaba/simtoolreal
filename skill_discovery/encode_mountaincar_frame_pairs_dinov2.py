"""Encode MountainCar frame-pair shards with frozen DINOv2 CLS features."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from skill_discovery.encode_minigrid_doorkey_dinov2 import MODEL_ID


def build_pair_features(frame_embeddings: np.ndarray) -> np.ndarray:
    if frame_embeddings.ndim != 3 or frame_embeddings.shape[1] != 2:
        raise ValueError("expected pair x two frames x embedding")
    before = frame_embeddings[:, 0]
    after = frame_embeddings[:, 1]
    features = np.concatenate((before, after, after - before), axis=1).astype(
        np.float32
    )
    return features / np.maximum(
        np.linalg.norm(features, axis=1, keepdims=True), 1.0e-8
    )


def encode_dataset(
    dataset_dir: Path,
    output_dir: Path,
    *,
    model_id: str = MODEL_ID,
    batch_size: int = 64,
    device: str = "cuda:0",
) -> dict[str, object]:
    import torch
    import transformers
    from transformers import AutoImageProcessor, AutoModel

    shard_paths = sorted(dataset_dir.glob("class_*.npz"))
    if len(shard_paths) != 4:
        raise ValueError("MountainCar pair dataset must contain four class shards")
    output_dir.mkdir(parents=True, exist_ok=False)
    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
    model = AutoModel.from_pretrained(model_id)
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    use_half = torch_device.type == "cuda"
    if use_half:
        model.half()

    embedding_rows = []
    pair_hash_rows = []
    class_rows = []
    split_rows = []
    started = time.monotonic()
    with torch.inference_mode():
        for shard_path in shard_paths:
            with np.load(shard_path) as shard:
                frames = shard["frames"].astype(np.uint8)
                pair_hash_rows.append(shard["pair_hashes"])
                class_rows.append(shard["classes"].astype(np.int8))
                split_rows.append(shard["splits"].astype(np.int8))
            if frames.ndim != 5 or frames.shape[1:] != (2, 400, 600, 3):
                raise ValueError("pair shard must contain Nx2x400x600x3 RGB")
            flat = frames.reshape(-1, 400, 600, 3)
            batches = []
            for start in range(0, len(flat), batch_size):
                inputs = processor(
                    images=[image for image in flat[start : start + batch_size]],
                    return_tensors="pt",
                )
                pixel_values = inputs["pixel_values"].to(torch_device)
                if use_half:
                    pixel_values = pixel_values.half()
                encoded = model(pixel_values=pixel_values)
                batches.append(
                    encoded.last_hidden_state[:, 0].float().cpu().numpy()
                )
            embedding_rows.append(
                np.concatenate(batches).astype(np.float32).reshape(len(frames), 2, -1)
            )
    elapsed = time.monotonic() - started
    frame_embeddings = np.concatenate(embedding_rows)
    pair_features = build_pair_features(frame_embeddings)
    pair_hashes = np.concatenate(pair_hash_rows)
    classes = np.concatenate(class_rows)
    splits = np.concatenate(split_rows)
    embedding_path = output_dir / "mountaincar_frame_pair_dinov2.npz"
    np.savez_compressed(
        embedding_path,
        frame_embeddings=frame_embeddings,
        pair_features=pair_features,
        pair_hashes=pair_hashes,
        classes=classes,
        splits=splits,
    )
    output = {
        "dataset_dir": str(dataset_dir.resolve()),
        "shards": [str(path.resolve()) for path in shard_paths],
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
        "pair_count": len(frame_embeddings),
        "frame_count": len(frame_embeddings) * 2,
        "frame_embedding_shape": list(frame_embeddings.shape),
        "pair_feature_shape": list(pair_features.shape),
        "embeddings": str(embedding_path.resolve()),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["metrics"] = str(metrics_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_id = f"frame_pair_dinov2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/mountaincar_continuous"
    ) / run_id
    output = encode_dataset(
        args.dataset_dir,
        output_dir,
        model_id=args.model_id,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
