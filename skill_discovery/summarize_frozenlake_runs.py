"""Summarize and visually replay FrozenLake skill-discovery runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from skill_discovery.frozenlake import FROZENLAKE_OUTCOMES, FrozenLakeConfig
from skill_discovery.generate_point_cup_dataset import _write_png
from skill_discovery.train_frozenlake_skills import (
    FrozenLakeTrainConfig,
    _rollout_policy,
)


def _load_config(data: dict[str, object]) -> FrozenLakeTrainConfig:
    values = dict(data)
    values["lake"] = FrozenLakeConfig(**values["lake"])
    return FrozenLakeTrainConfig(**values)


def summarize_runs(run_dirs: list[Path], output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    frame_size = 160
    gap = 6
    marker_width = 9
    sheet = np.full(
        (
            len(run_dirs) * frame_size + (len(run_dirs) - 1) * gap,
            marker_width + 3 * frame_size + 2 * gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    colors = ((48, 116, 173), (198, 72, 58), (42, 137, 94))
    runs = []
    for row, run_dir in enumerate(run_dirs):
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        config = _load_config(metrics["config"])
        q_table = np.load(run_dir / "q_table.npz")["q_table"]
        replay = []
        y = row * (frame_size + gap)
        sheet[y : y + frame_size, :marker_width] = (40, 48, 54)
        for skill in range(config.num_skills):
            rollout = _rollout_policy(
                q_table,
                config,
                skill,
                seed=config.seed + 1_990_000 + skill,
                render=True,
            )
            frame = rollout.pop("frames")[-1]
            rows = np.linspace(0, frame.shape[0] - 1, frame_size).astype(np.int64)
            cols = np.linspace(0, frame.shape[1] - 1, frame_size).astype(np.int64)
            resized = frame[rows][:, cols]
            x = marker_width + skill * (frame_size + gap)
            sheet[y : y + frame_size, x : x + frame_size] = resized
            outcome = int(rollout["outcome"])
            sheet[y : y + 8, x : x + frame_size] = colors[outcome]
            replay.append(
                {
                    "skill": skill,
                    "outcome": FROZENLAKE_OUTCOMES[outcome],
                    **rollout,
                }
            )
        final = metrics["final_evaluation"]
        matched_by_outcome = [0.0] * len(FROZENLAKE_OUTCOMES)
        for skill, outcome in enumerate(final["outcome_assignment"]):
            matched_by_outcome[outcome] = final["matched_outcome_rates"][skill]
        runs.append(
            {
                "run_dir": str(run_dir.resolve()),
                "seed": config.seed,
                "objective": config.objective,
                "assignment": final["outcome_assignment_names"],
                "matched_rates": final["matched_outcome_rates"],
                "matched_rate_by_outcome": {
                    name: matched_by_outcome[index]
                    for index, name in enumerate(FROZENLAKE_OUTCOMES)
                },
                "goal_native_success": final["goal_skill_native_success_rate"],
                "checkpoint_stability_passed": metrics[
                    "checkpoint_stability_passed"
                ],
                "signal_gate_passed": metrics["signal_gate_passed"],
                "replay": replay,
            }
        )
    image_path = output_dir / "multiseed_final_replay.png"
    _write_png(image_path, sheet)
    pass_count = sum(run["signal_gate_passed"] for run in runs)
    matched = np.asarray([run["matched_rates"] for run in runs], dtype=np.float64)
    matched_by_outcome = np.asarray(
        [
            [run["matched_rate_by_outcome"][name] for name in FROZENLAKE_OUTCOMES]
            for run in runs
        ],
        dtype=np.float64,
    )
    output = {
        "run_count": len(runs),
        "runs": runs,
        "signal_gate_pass_count": pass_count,
        "multiseed_gate_passed": pass_count >= max(len(runs) - 1, 1),
        "matched_rate_mean": matched.mean(axis=0).tolist(),
        "matched_rate_min": matched.min(axis=0).tolist(),
        "matched_rate_by_outcome_mean": {
            name: float(matched_by_outcome[:, index].mean())
            for index, name in enumerate(FROZENLAKE_OUTCOMES)
        },
        "matched_rate_by_outcome_min": {
            name: float(matched_by_outcome[:, index].min())
            for index, name in enumerate(FROZENLAKE_OUTCOMES)
        },
        "image": str(image_path.resolve()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = summarize_runs(args.run_dirs, args.output_dir)
    print(
        json.dumps(
            {
                "summary": str((args.output_dir / "summary.json").resolve()),
                "image": output["image"],
                "signal_gate_pass_count": output["signal_gate_pass_count"],
                "multiseed_gate_passed": output["multiseed_gate_passed"],
                "matched_rate_mean": output["matched_rate_mean"],
                "matched_rate_by_outcome_mean": output[
                    "matched_rate_by_outcome_mean"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
