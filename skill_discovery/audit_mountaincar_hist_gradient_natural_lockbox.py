"""Run the frozen HistGradient model on fresh MountainCar natural seeds."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from skill_discovery.audit_mountaincar_rollout_rejection import (
    RolloutRejectionConfig,
    _select_examples,
    _write_contact_sheet,
    rejection_metrics,
    rollout_worker,
)
from skill_discovery.audit_mountaincar_visual_decision_tree_rollout import (
    natural_gate_results,
)


def fresh_natural_config() -> RolloutRejectionConfig:
    return RolloutRejectionConfig(
        energy_episodes=128,
        random_episodes=256,
        random_horizon=200,
        energy_seed_start=15_100_000,
        random_seed_start=16_100_000,
        action_seed=17_100_007,
        batch_size=32,
        workers=4,
    )


def run_natural_lockbox(
    config: RolloutRejectionConfig,
    model_path: Path,
    balanced_metrics_path: Path,
    background_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    balanced = json.loads(balanced_metrics_path.read_text(encoding="utf-8"))
    if not balanced["lockbox_gate_passed"]:
        raise RuntimeError("fresh balanced lockbox did not pass")
    classifier = joblib.load(model_path)
    background = np.asarray(Image.open(background_path).convert("RGB"), dtype=np.uint8)
    output_dir.mkdir(parents=True, exist_ok=False)
    unused_features = np.empty((0, 3), dtype=np.float32)
    unused_classes = np.empty(0, dtype=np.int8)
    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        worker_rows = list(
            pool.map(
                rollout_worker,
                [config] * config.workers,
                range(config.workers),
                [background] * config.workers,
                [unused_features] * config.workers,
                [unused_classes] * config.workers,
                [None] * config.workers,
                [classifier] * config.workers,
            )
        )
    predictions = np.concatenate([row["predictions"] for row in worker_rows])
    confidence = np.concatenate([row["distances"] for row in worker_rows])
    oracle = np.concatenate([row["oracle"] for row in worker_rows])
    sources = np.concatenate([row["sources"] for row in worker_rows])
    combined = rejection_metrics(oracle, predictions)
    energy = rejection_metrics(oracle[sources == 0], predictions[sources == 0])
    random = rejection_metrics(oracle[sources == 1], predictions[sources == 1])
    gates = natural_gate_results(combined)
    examples = _select_examples(worker_rows)
    contact_sheet_path = output_dir / "fresh_natural_lockbox_contact_sheet.png"
    _write_contact_sheet(contact_sheet_path, examples)
    public_examples = [
        {key: value for key, value in example.items() if key != "frames"}
        for example in examples
    ]
    output = {
        "config": asdict(config),
        "model_path": str(model_path.resolve()),
        "balanced_metrics_path": str(balanced_metrics_path.resolve()),
        "background_path": str(background_path.resolve()),
        "combined": combined,
        "energy": energy,
        "random": random,
        "confidence_summary": {
            "minimum": float(confidence.min()),
            "median": float(np.median(confidence)),
            "maximum": float(confidence.max()),
        },
        **gates,
        "contact_sheet": str(contact_sheet_path.resolve()),
        "contact_examples": public_examples,
        "manual_contact_sheet_gate": "pending",
        "passed": False,
    }
    metrics_path = output_dir / "audit.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["metrics"] = str(metrics_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("balanced_metrics", type=Path)
    parser.add_argument("background", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = fresh_natural_config()
    run_id = f"fresh_natural_lockbox_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/mountaincar_continuous"
    ) / run_id
    output = run_natural_lockbox(
        config,
        args.model,
        args.balanced_metrics,
        args.background,
        output_dir,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
