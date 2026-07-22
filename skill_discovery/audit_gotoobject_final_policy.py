"""Audit a frozen GoToObject Q table on independent layout seed blocks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from skill_discovery.minigrid_gotoobject import GOTOOBJECT_STAGES
from skill_discovery.train_minigrid_gotoobject_skills import (
    POLICY_ACTIONS,
    GoToObjectTrainConfig,
    RelationKey,
    evaluate_q_table,
)


@dataclass(frozen=True)
class SeedBlockAuditConfig:
    blocks: int = 5
    eval_episodes_per_skill: int = 512
    seed_start_offset: int = 1_100_000
    seed_stride: int = 100_000
    stage_rate_gate: float = 0.90

    def __post_init__(self) -> None:
        if self.blocks <= 0 or self.eval_episodes_per_skill <= 0:
            raise ValueError("blocks and evaluation episodes must be positive")
        if self.seed_stride <= 0:
            raise ValueError("seed stride must be positive")
        if not 0 <= self.stage_rate_gate <= 1:
            raise ValueError("stage rate gate must be in [0, 1]")


def load_saved_q_table(path: Path) -> dict[RelationKey, np.ndarray]:
    with np.load(path) as archive:
        keys = np.asarray(archive["relation_keys"])
        q_values = np.asarray(archive["q_values"])
        visits = np.asarray(archive["visits"])
    expected_q_shape = (len(keys), len(GOTOOBJECT_STAGES), len(POLICY_ACTIONS))
    if keys.ndim != 2 or keys.shape[1:] != (8,):
        raise ValueError("saved relation keys must have shape (N, 8)")
    if q_values.shape != expected_q_shape or visits.shape != expected_q_shape:
        raise ValueError("saved Q values and visits have incompatible shapes")
    if not np.isfinite(q_values).all() or not np.isfinite(visits).all():
        raise ValueError("saved Q values and visits must be finite")
    relation_keys = [tuple(int(value) for value in row) for row in keys]
    if len(set(relation_keys)) != len(relation_keys):
        raise ValueError("saved relation keys contain duplicates")
    return {
        key: q_values[index].astype(np.float64, copy=True)
        for index, key in enumerate(relation_keys)
    }


def _policy_config(
    metrics: dict[str, object],
    audit: SeedBlockAuditConfig,
) -> GoToObjectTrainConfig:
    source = metrics["config"]
    matrix = metrics["bootstrap_calibrated_matrix"]
    if not isinstance(source, dict):
        raise ValueError("source metrics config must be a mapping")
    return GoToObjectTrainConfig(
        objective="frozen_matrix",
        frozen_reward_matrix=tuple(tuple(row) for row in matrix),
        frozen_reward_source="phase5t_final_policy_audit",
        frozen_reward_calibration="runner_up_unit",
        reward_timing="occupancy",
        seed=int(source["seed"]),
        episodes=int(source["policy_episodes"]),
        horizon=int(source["horizon"]),
        gamma=float(source["gamma"]),
        eval_interval=int(source["eval_interval"]),
        eval_episodes_per_skill=audit.eval_episodes_per_skill,
        stability_checkpoints=int(source["stability_checkpoints"]),
        stage_rate_gates=(audit.stage_rate_gate,) * len(GOTOOBJECT_STAGES),
    )


def audit_final_policy(
    run_dir: Path,
    audit: SeedBlockAuditConfig,
    *,
    output_path: Path | None = None,
) -> dict[str, object]:
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not metrics.get("policy_phase_ran"):
        raise ValueError("source run did not complete the policy phase")
    q_table = load_saved_q_table(run_dir / "q_table.npz")
    policy_config = _policy_config(metrics, audit)
    blocks = []
    for index in range(audit.blocks):
        block_seed = (
            policy_config.seed
            + audit.seed_start_offset
            + index * audit.seed_stride
        )
        evaluation = evaluate_q_table(q_table, policy_config, seed=block_seed)
        blocks.append({"block": index, "seed": block_seed, **evaluation})
    rates = {
        stage: np.asarray(
            [row["matched_rate_by_stage"][stage] for row in blocks],
            dtype=np.float64,
        )
        for stage in GOTOOBJECT_STAGES
    }
    summary = {
        stage: {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std()),
            "worst": float(values.min()),
        }
        for stage, values in rates.items()
    }
    output = {
        "source_run": str(run_dir.resolve()),
        "audit_config": asdict(audit),
        "q_state_count": len(q_table),
        "blocks": blocks,
        "stage_summary": summary,
        "blocks_passed": sum(
            bool(row["specialization_gate_passed"]) for row in blocks
        ),
        "all_blocks_passed": all(
            bool(row["specialization_gate_passed"]) for row in blocks
        ),
        "changes_training_gate_result": False,
    }
    destination = output_path or run_dir / "final_policy_seed_block_audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=512)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()
    audit = SeedBlockAuditConfig(
        blocks=args.blocks,
        eval_episodes_per_skill=args.eval_episodes,
    )
    output = audit_final_policy(
        args.run_dir,
        audit,
        output_path=args.output_path,
    )
    print(
        json.dumps(
            {
                "source_run": output["source_run"],
                "blocks_passed": output["blocks_passed"],
                "all_blocks_passed": output["all_blocks_passed"],
                "stage_summary": output["stage_summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
