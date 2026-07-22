"""Train DoorKey discovery with a frozen causal DINO stage metric."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from skill_discovery.encode_minigrid_doorkey_dinov2 import MODEL_ID
from skill_discovery.evaluate_minigrid_doorkey_visual_state_transfer import (
    _normalize,
    nearest_centers,
)
from skill_discovery.generate_minigrid_doorkey_policy_sequences import (
    _write_contact_sheet,
)
from skill_discovery.train_minigrid_doorkey_discovery import (
    DoorKeyDiscoveryConfig,
    train_discovery_run,
)
from skill_discovery.train_minigrid_doorkey_tabular import (
    DoorKeyState,
    compact_doorkey_state,
)


CAUSAL_CLUSTER_ORDER = (2, 0, 3, 1)


class DinoCurrentFrameEncoder:
    def __init__(self, model_id: str, device: str):
        import torch
        import transformers
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(
            model_id,
            use_fast=False,
        )
        self.model = AutoModel.from_pretrained(model_id)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.use_half = self.device.type == "cuda"
        if self.use_half:
            self.model.half()
        self.model_id = model_id
        self.gpu_name = (
            torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else "cpu"
        )
        self.versions = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        }

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        inputs = self.processor(images=[frame], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        if self.use_half:
            pixel_values = pixel_values.half()
        with self.torch.inference_mode():
            output = self.model(pixel_values=pixel_values)
        embedding = output.last_hidden_state[:, 0].float().cpu().numpy()
        return _normalize(embedding)[0].astype(np.float32)

    def summary(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "device": str(self.device),
            "gpu_name": self.gpu_name,
            "versions": self.versions,
        }


class RenderedClusterLookup:
    def __init__(
        self,
        centers: np.ndarray,
        encode_frame: Callable[[np.ndarray], np.ndarray],
    ):
        centers = np.asarray(centers, dtype=np.float32)
        if centers.ndim != 2 or centers.shape[0] != 4:
            raise ValueError("frozen visual lookup requires four center vectors")
        self.centers = centers
        self.encode_frame = encode_frame
        self.index_by_key: dict[DoorKeyState, int] = {}
        self.keys: list[DoorKeyState] = []
        self.frames: list[np.ndarray] = []
        self.hashes: list[str] = []
        self.embeddings: list[np.ndarray] = []
        self.clusters: list[int] = []
        self.query_count = 0
        self.alias_count = 0
        self.query_counts_by_cluster = np.zeros(4, dtype=np.int64)

    def cluster(self, key: DoorKeyState, frame: np.ndarray) -> int:
        image = np.asarray(frame, dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("visual lookup requires an RGB image")
        digest = hashlib.blake2b(image.tobytes(), digest_size=16).hexdigest()
        existing = self.index_by_key.get(key)
        if existing is not None:
            if self.hashes[existing] != digest:
                self.alias_count += 1
                raise ValueError("compact state maps to multiple RGB frames")
            cluster = self.clusters[existing]
        else:
            embedding = np.asarray(self.encode_frame(image), dtype=np.float32)
            if embedding.shape != (self.centers.shape[1],):
                raise ValueError("frame encoder returned the wrong embedding shape")
            cluster = int(nearest_centers(embedding[None], self.centers)[0])
            existing = len(self.keys)
            self.index_by_key[key] = existing
            self.keys.append(key)
            self.frames.append(image.copy())
            self.hashes.append(digest)
            self.embeddings.append(embedding.copy())
            self.clusters.append(cluster)
        self.query_count += 1
        self.query_counts_by_cluster[cluster] += 1
        return cluster

    def query(self, env: object) -> int:
        return self.cluster(compact_doorkey_state(env), env.render())

    def summary(self) -> dict[str, object]:
        unique_counts = np.bincount(self.clusters, minlength=4)
        return {
            "unique_compact_state_count": len(self.keys),
            "encoded_frame_count": len(self.embeddings),
            "query_count": self.query_count,
            "compact_state_rgb_alias_count": self.alias_count,
            "unique_state_count_by_raw_cluster": unique_counts.tolist(),
            "query_count_by_raw_cluster": self.query_counts_by_cluster.tolist(),
            "all_raw_clusters_queried": bool(
                np.all(self.query_counts_by_cluster > 0)
            ),
        }

    def save(self, output_dir: Path, cluster_order: tuple[int, ...]) -> dict[str, str]:
        keys = np.asarray(self.keys, dtype=np.int16)
        frames = np.asarray(self.frames, dtype=np.uint8)
        embeddings = np.asarray(self.embeddings, dtype=np.float32)
        clusters = np.asarray(self.clusters, dtype=np.int8)
        cache_path = output_dir / "visual_cluster_cache.npz"
        np.savez_compressed(
            cache_path,
            state_keys=keys,
            state_frames=frames,
            state_hashes=np.asarray(self.hashes),
            state_embeddings=embeddings,
            state_clusters=clusters,
            cluster_order=np.asarray(cluster_order, dtype=np.int8),
        )
        positions = {cluster: index for index, cluster in enumerate(cluster_order)}
        decoded_stages = np.asarray(
            [positions[int(cluster)] for cluster in clusters],
            dtype=np.int8,
        )
        image_path = output_dir / "visual_cluster_cache_audit.png"
        manifest = _write_contact_sheet(
            image_path,
            frames,
            decoded_stages,
            keys,
        )
        manifest_path = output_dir / "visual_cluster_cache_audit.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {
            "cache": str(cache_path.resolve()),
            "manual_audit_image": str(image_path.resolve()),
            "manual_audit_manifest": str(manifest_path.resolve()),
        }


class CausalVisualStageTracker:
    def __init__(self, factory: "CausalVisualStageFactory"):
        self.factory = factory
        self.position = 0
        self.initialized = False

    def observe(
        self,
        env: object,
        *,
        terminated: bool = False,
        reward: float = 0.0,
    ) -> int:
        del terminated, reward
        cluster = int(self.factory.lookup.query(env))
        order = self.factory.cluster_order
        if not self.initialized:
            if cluster != order[0]:
                raise ValueError("visual sequence does not reset at the frozen root")
            self.initialized = True
        elif self.position + 1 < len(order) and cluster == order[self.position + 1]:
            self.position += 1
        self.factory.decoded_query_counts[self.position] += 1
        return self.position


class CausalVisualStageFactory:
    def __init__(
        self,
        lookup: RenderedClusterLookup,
        cluster_order: tuple[int, ...] = CAUSAL_CLUSTER_ORDER,
    ):
        if sorted(cluster_order) != list(range(4)):
            raise ValueError("causal cluster order must be a four-cluster permutation")
        self.lookup = lookup
        self.cluster_order = cluster_order
        self.decoded_query_counts = np.zeros(4, dtype=np.int64)
        self.episode_count = 0

    def __call__(self) -> CausalVisualStageTracker:
        self.episode_count += 1
        return CausalVisualStageTracker(self)

    def summary(self) -> dict[str, object]:
        return {
            "name": "frozen_dinov2_current_causal_ordered_cluster",
            "oracle_stage_used_for_training": False,
            "skill_id_used_by_stage_tracker": False,
            "native_reward_used_by_stage_tracker": False,
            "future_frame_used_by_stage_tracker": False,
            "fit_or_refit_performed": False,
            "cluster_order": list(self.cluster_order),
            "episode_tracker_count": self.episode_count,
            "decoded_query_count_by_stage": self.decoded_query_counts.tolist(),
            "all_decoded_stages_queried": bool(
                np.all(self.decoded_query_counts > 0)
            ),
            **self.lookup.summary(),
        }


def train_visual_discovery_run(
    config: DoorKeyDiscoveryConfig,
    cluster_metrics_path: Path,
    cluster_assignments_path: Path,
    output_dir: Path,
    *,
    model_id: str = MODEL_ID,
    device: str = "cuda:0",
) -> dict[str, object]:
    source_metrics = json.loads(cluster_metrics_path.read_text(encoding="utf-8"))
    source_method = source_metrics["methods"]["dinov2_current"]
    if not source_method["foundation_visual_gate_passed"]:
        raise ValueError("frozen DINO current source gate did not pass")
    with np.load(cluster_assignments_path) as source_assignments:
        centers = source_assignments["dinov2_current_centers"].astype(np.float32)
    encoder = DinoCurrentFrameEncoder(model_id, device)
    lookup = RenderedClusterLookup(centers, encoder)
    stage_factory = CausalVisualStageFactory(lookup)
    output = train_discovery_run(
        config,
        output_dir,
        stage_tracker_factory=stage_factory,
        render_mode="rgb_array",
        training_stage_metric_summary=stage_factory.summary,
    )
    artifacts = lookup.save(output_dir, stage_factory.cluster_order)
    stage_summary = {
        **stage_factory.summary(),
        **encoder.summary(),
        "cluster_metrics": str(cluster_metrics_path.resolve()),
        "cluster_assignments": str(cluster_assignments_path.resolve()),
        **artifacts,
    }
    cache_gate = bool(
        stage_summary["compact_state_rgb_alias_count"] == 0
        and stage_summary["all_raw_clusters_queried"]
        and stage_summary["all_decoded_stages_queried"]
    )
    output.update(
        {
            "training_stage_metric": stage_summary,
            "visual_cache_gate_passed": cache_gate,
            "visual_signal_gate_passed": bool(
                cache_gate and output["signal_gate_passed"]
            ),
        }
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cluster-metrics", type=Path, required=True)
    parser.add_argument("--cluster-assignments", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = DoorKeyDiscoveryConfig(seed=args.seed)
    run_id = (
        f"doorkey5_online_visual_causal_discovery_seed{config.seed}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/minigrid_doorkey_training"
    ) / run_id
    output = train_visual_discovery_run(
        config,
        args.cluster_metrics,
        args.cluster_assignments,
        output_dir,
        model_id=args.model_id,
        device=args.device,
    )
    final = output.get("final_evaluation")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "elapsed_seconds": output["elapsed_seconds"],
                "bootstrap_gate_passed": output["bootstrap_gate_passed"],
                "bootstrap_assigned_stages": output[
                    "bootstrap_assigned_stages"
                ],
                "policy_phase_ran": output["policy_phase_ran"],
                "final_target_stage_rates": final["target_stage_rates"]
                if final
                else None,
                "final_state_rates": final["final_state_rates"]
                if final
                else None,
                "goal_native_success_rate": final["goal_native_success_rate"]
                if final
                else None,
                "checkpoint_stability_passed": output[
                    "checkpoint_stability_passed"
                ],
                "training_stage_metric": output["training_stage_metric"],
                "visual_cache_gate_passed": output["visual_cache_gate_passed"],
                "visual_signal_gate_passed": output["visual_signal_gate_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
