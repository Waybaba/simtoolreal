"""Post-hoc oracle audit of failed causal-visual DoorKey discovery replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from skill_discovery.minigrid_doorkey import DOORKEY_STAGES


def stage_from_compact_state(key: np.ndarray) -> int:
    state = np.asarray(key, dtype=np.int64)
    if state.shape != (12,):
        raise ValueError("DoorKey compact state must contain 12 integers")
    if np.array_equal(state[0:2], state[10:12]):
        return 3
    if state[8] == 1:
        return 2
    if state[5] == 1:
        return 1
    return 0


def stage_confusion(expected: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    expected = np.asarray(expected, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    if expected.ndim != 1 or predicted.shape != expected.shape:
        raise ValueError("expected and predicted stages must be matching vectors")
    if np.any(expected < 0) or np.any(expected >= 4):
        raise ValueError("expected vector contains an invalid stage")
    if np.any(predicted < 0) or np.any(predicted >= 4):
        raise ValueError("predicted vector contains an invalid stage")
    confusion = np.zeros((4, 4), dtype=np.int64)
    for actual, prediction in zip(expected, predicted):
        confusion[int(actual), int(prediction)] += 1
    return confusion


def audit_run(run_dir: Path) -> dict[str, object]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    with np.load(run_dir / "visual_cluster_cache.npz") as cache:
        cache_keys = cache["state_keys"].astype(np.int64)
        cache_clusters = cache["state_clusters"].astype(np.int64)
        cluster_order = cache["cluster_order"].astype(np.int64)
    positions = np.empty(4, dtype=np.int64)
    for position, cluster in enumerate(cluster_order):
        positions[int(cluster)] = position
    cache_oracle = np.asarray(
        [stage_from_compact_state(key) for key in cache_keys],
        dtype=np.int64,
    )
    cache_decoded_class = positions[cache_clusters]
    cache_confusion = stage_confusion(cache_oracle, cache_decoded_class)
    raw_goal_cluster = int(cluster_order[3])
    raw_goal_states = cache_clusters == raw_goal_cluster

    with np.load(run_dir / "bootstrap_replay_buffer.npz") as replay:
        offsets = replay["episode_offsets"].astype(np.int64)
        skills = replay["skills"].astype(np.int64)
        decoded = replay["stages"].astype(np.int64)
        next_keys = replay["next_state_keys"].astype(np.int64)
        terminals = replay["terminals"].astype(np.bool_)
    nonterminal = ~terminals
    oracle_next = np.asarray(
        [stage_from_compact_state(key) for key in next_keys[nonterminal]],
        dtype=np.int64,
    )
    decoded_next = decoded[nonterminal]
    replay_confusion = stage_confusion(oracle_next, decoded_next)
    false_goal = decoded_next == 3
    false_goal_oracle_counts = np.bincount(
        oracle_next[false_goal],
        minlength=4,
    )
    premature_goal_episodes = 0
    premature_goal_by_skill = np.zeros(4, dtype=np.int64)
    episode_count_by_skill = np.zeros(4, dtype=np.int64)
    for start, stop in zip(offsets[:-1], offsets[1:]):
        if stop <= start:
            continue
        skill = int(skills[start])
        episode_count_by_skill[skill] += 1
        local_nonterminal = nonterminal[start:stop]
        local_decoded = decoded[start:stop][local_nonterminal]
        if np.any(local_decoded == 3):
            premature_goal_episodes += 1
            premature_goal_by_skill[skill] += 1
    per_skill_rates = np.divide(
        premature_goal_by_skill,
        episode_count_by_skill,
        out=np.zeros(4, dtype=np.float64),
        where=episode_count_by_skill > 0,
    )
    return {
        "run_dir": str(run_dir.resolve()),
        "seed": int(metrics["config"]["seed"]),
        "formal_visual_signal_gate_passed": bool(
            metrics["visual_signal_gate_passed"]
        ),
        "cluster_order": cluster_order.tolist(),
        "cache_unique_state_count": len(cache_keys),
        "cache_oracle_by_decoded_class_confusion": cache_confusion.tolist(),
        "raw_goal_cluster": raw_goal_cluster,
        "raw_goal_cluster_unique_state_count": int(np.count_nonzero(raw_goal_states)),
        "raw_goal_cluster_oracle_stage_counts": np.bincount(
            cache_oracle[raw_goal_states],
            minlength=4,
        ).tolist(),
        "replay_nonterminal_transition_count": int(np.count_nonzero(nonterminal)),
        "replay_oracle_by_decoded_stage_confusion": replay_confusion.tolist(),
        "replay_nonterminal_decoded_goal_count": int(np.count_nonzero(false_goal)),
        "replay_nonterminal_decoded_goal_rate": float(np.mean(false_goal)),
        "replay_nonterminal_decoded_goal_oracle_stage_counts": (
            false_goal_oracle_counts.tolist()
        ),
        "bootstrap_episode_count": len(offsets) - 1,
        "bootstrap_premature_goal_episode_count": premature_goal_episodes,
        "bootstrap_premature_goal_episode_rate": float(
            premature_goal_episodes / (len(offsets) - 1)
        ),
        "bootstrap_premature_goal_episode_count_by_skill": (
            premature_goal_by_skill.tolist()
        ),
        "bootstrap_premature_goal_episode_rate_by_skill": per_skill_rates.tolist(),
        "oracle_used_for_training": False,
        "audit_is_posthoc": True,
    }


def audit_runs(run_dirs: list[Path], output_path: Path) -> dict[str, object]:
    runs = sorted((audit_run(path) for path in run_dirs), key=lambda row: row["seed"])
    seeds = [int(row["seed"]) for row in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("visual discovery run seeds must be unique")
    output = {
        "stage_names": DOORKEY_STAGES,
        "method": "posthoc_compact_state_oracle_audit_excluding_terminal_next_states",
        "runs": runs,
        "seeds": seeds,
        "all_runs_have_premature_goal": all(
            row["bootstrap_premature_goal_episode_count"] > 0 for row in runs
        ),
        "runs_training": False,
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = audit_runs(args.run_dirs, args.output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
