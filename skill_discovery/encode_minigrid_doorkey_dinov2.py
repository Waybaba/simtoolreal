"""Cache frozen DINOv2 embeddings for balanced DoorKey stage frames."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


MODEL_ID = "facebook/dinov2-small"


def build_dinov2_representations(
    frame_embeddings: np.ndarray,
) -> dict[str, np.ndarray]:
    if frame_embeddings.ndim != 3 or frame_embeddings.shape[1] != 4:
        raise ValueError("expected trajectory x four stages x embedding")
    norms = np.linalg.norm(frame_embeddings, axis=-1, keepdims=True)
    frames = frame_embeddings / np.maximum(norms, 1.0e-8)
    current = frames.reshape(-1, frames.shape[-1])
    start = np.repeat(frames[:, :1], 4, axis=1).reshape(
        -1,
        frames.shape[-1],
    )
    start_current = np.concatenate((start, current), axis=1)
    start_current /= np.maximum(
        np.linalg.norm(start_current, axis=1, keepdims=True),
        1.0e-8,
    )
    temporal_delta = current - start
    temporal_delta /= np.maximum(
        np.linalg.norm(temporal_delta, axis=1, keepdims=True),
        1.0e-8,
    )
    return {
        "dinov2_current": current.astype(np.float32),
        "dinov2_start_current": start_current.astype(np.float32),
        "dinov2_temporal_delta": temporal_delta.astype(np.float32),
    }


def encode_dataset(
    dataset_path: Path,
    output_dir: Path,
    *,
    model_id: str = MODEL_ID,
    batch_size: int = 64,
    device: str = "cuda:0",
) -> dict[str, object]:
    import torch
    import transformers
    from transformers import AutoImageProcessor, AutoModel

    output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(dataset_path) as data:
        frames = data["frames"].astype(np.uint8)
        split = data["split"].astype(np.int8)
        generation_groups = data["generation_groups"].astype(np.int32)
        env_seeds = data["env_seeds"].astype(np.int64)
    if frames.ndim != 5 or frames.shape[1] != 4 or frames.shape[-1] != 3:
        raise ValueError("DoorKey dataset must contain four RGB stage frames")
    flat_frames = frames.reshape(-1, *frames.shape[2:])
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
            encoded_batches.append(
                outputs.last_hidden_state[:, 0, :].float().cpu().numpy()
            )
    elapsed = time.monotonic() - started
    frame_embeddings = np.concatenate(encoded_batches).astype(np.float32).reshape(
        frames.shape[0],
        frames.shape[1],
        -1,
    )
    representations = build_dinov2_representations(frame_embeddings)
    embedding_path = output_dir / "dinov2_embeddings.npz"
    np.savez_compressed(
        embedding_path,
        frame_embeddings=frame_embeddings,
        split=split,
        generation_groups=generation_groups,
        env_seeds=env_seeds,
        **representations,
    )
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
        "elapsed_seconds": elapsed,
        "frame_count": len(flat_frames),
        "frame_embedding_shape": list(frame_embeddings.shape),
        "representation_shapes": {
            name: list(values.shape) for name, values in representations.items()
        },
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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = encode_dataset(
        args.dataset,
        args.output_dir,
        model_id=args.model_id,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "metrics": output["metrics"],
                "embeddings": output["embeddings"],
                "device": output["device"],
                "gpu_name": output["gpu_name"],
                "elapsed_seconds": output["elapsed_seconds"],
                "frame_embedding_shape": output["frame_embedding_shape"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
