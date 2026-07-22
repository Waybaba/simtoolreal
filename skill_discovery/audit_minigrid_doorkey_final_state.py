"""Audit frozen DoorKey policies for compositional final-state persistence."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from skill_discovery.minigrid_doorkey import DOORKEY_STAGES
from skill_discovery.train_minigrid_doorkey_tabular import (
    DOORKEY_POLICY_ACTIONS,
    DoorKeyState,
    DoorKeyTabularConfig,
    _rollout,
)


@dataclass(frozen=True)
class DoorKeyFinalStateAuditConfig:
    eval_episodes_per_skill: int = 512
    seed_offset: int = 1_100_000
    final_state_gate: float = 0.80

    def __post_init__(self) -> None:
        if self.eval_episodes_per_skill <= 0:
            raise ValueError("evaluation episodes must be positive")
        if not 0 <= self.final_state_gate <= 1:
            raise ValueError("final state gate must be in [0, 1]")


def load_doorkey_q_table(path: Path) -> dict[DoorKeyState, np.ndarray]:
    with np.load(path) as archive:
        keys = np.asarray(archive["state_keys"])
        q_values = np.asarray(archive["q_values"])
        visits = np.asarray(archive["visits"])
    expected = (len(keys), len(DOORKEY_STAGES), len(DOORKEY_POLICY_ACTIONS))
    if keys.ndim != 2 or keys.shape[1:] != (12,):
        raise ValueError("saved DoorKey states must have shape (N, 12)")
    if q_values.shape != expected or visits.shape != expected:
        raise ValueError("saved DoorKey Q values or visits have invalid shape")
    if not np.isfinite(q_values).all() or not np.isfinite(visits).all():
        raise ValueError("saved DoorKey Q values and visits must be finite")
    state_keys = [tuple(int(value) for value in row) for row in keys]
    if len(set(state_keys)) != len(state_keys):
        raise ValueError("saved DoorKey states contain duplicates")
    return {
        key: q_values[index].astype(np.float64, copy=True)
        for index, key in enumerate(state_keys)
    }


def _final_state_success(skill: int, rollout: dict[str, object]) -> bool:
    carrying = rollout["carrying"]
    door_open = bool(rollout["door_open"])
    native_success = bool(rollout["native_success"])
    if skill == 0:
        return int(rollout["stage"]) == 0
    if skill == 1:
        return carrying is not None and carrying[0] == "key" and not door_open
    if skill == 2:
        return door_open and not native_success
    return native_success


def audit_final_states(
    run_dir: Path,
    audit: DoorKeyFinalStateAuditConfig,
    *,
    output_path: Path | None = None,
) -> dict[str, object]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    source = metrics["config"]
    if not isinstance(source, dict):
        raise ValueError("source config must be a mapping")
    config = DoorKeyTabularConfig(
        env_id=str(source["env_id"]),
        seed=int(source["seed"]),
        episodes=int(source["episodes"]),
        horizon=int(source["horizon"]),
        evaluation_checkpoints=tuple(source["evaluation_checkpoints"]),
        eval_episodes_per_skill=audit.eval_episodes_per_skill,
        stage_rate_gate=float(source["stage_rate_gate"]),
        valid_action_mask=bool(source["valid_action_mask"]),
    )
    if not config.valid_action_mask:
        raise ValueError("final-state audit requires the action-mask run")
    q_table = load_doorkey_q_table(run_dir / "q_table.npz")
    base_seed = config.seed + audit.seed_offset
    furthest_counts = np.zeros((4, 4), dtype=np.int64)
    success_counts = np.zeros(4, dtype=np.int64)
    for skill in range(4):
        for episode in range(audit.eval_episodes_per_skill):
            rollout = _rollout(
                q_table,
                config,
                skill,
                seed=base_seed + episode,
            )
            furthest_counts[skill, int(rollout["stage"])] += 1
            success_counts[skill] += int(_final_state_success(skill, rollout))
    rates = success_counts / audit.eval_episodes_per_skill
    output = {
        "source_run": str(run_dir.resolve()),
        "audit_config": asdict(audit),
        "seed": base_seed,
        "furthest_stage_rates": (
            furthest_counts / audit.eval_episodes_per_skill
        ).tolist(),
        "final_state_rates": rates.tolist(),
        "final_state_rate_by_target": {
            stage: float(rates[index])
            for index, stage in enumerate(DOORKEY_STAGES)
        },
        "final_state_gate_passed": bool(
            np.all(rates >= audit.final_state_gate)
        ),
        "changes_training_gate_result": False,
    }
    destination = output_path or run_dir / "final_state_persistence_audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-episodes", type=int, default=512)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()
    audit = DoorKeyFinalStateAuditConfig(
        eval_episodes_per_skill=args.eval_episodes,
    )
    output = audit_final_states(
        args.run_dir,
        audit,
        output_path=args.output_path,
    )
    print(
        json.dumps(
            {
                "source_run": output["source_run"],
                "final_state_rate_by_target": output[
                    "final_state_rate_by_target"
                ],
                "final_state_gate_passed": output["final_state_gate_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
