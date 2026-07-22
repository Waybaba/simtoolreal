"""Run frozen multi-seed FrozenLake baseline suites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skill_discovery.summarize_frozenlake_runs import summarize_runs
from skill_discovery.train_frozenlake_skills import (
    FrozenLakeTrainConfig,
    train_run,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=("random", "raw", "semantic", "semantic_spread"),
        default=("random", "raw", "semantic"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 17, 27, 37, 47))
    parser.add_argument("--episodes", type=int, default=30_000)
    parser.add_argument("--epsilon-decay-fraction", type=float, default=0.80)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    methods = {}
    for objective in args.objectives:
        run_dirs = []
        for seed in args.seeds:
            config = FrozenLakeTrainConfig(
                objective=objective,
                seed=seed,
                episodes=args.episodes,
                epsilon_decay_fraction=args.epsilon_decay_fraction,
            )
            run_dir = args.output_dir / f"{objective}_seed{seed}"
            output = train_run(config, run_dir)
            run_dirs.append(run_dir)
            print(
                json.dumps(
                    {
                        "objective": objective,
                        "seed": seed,
                        "signal_gate_passed": output["signal_gate_passed"],
                    }
                ),
                flush=True,
            )
        summary_dir = args.output_dir / f"{objective}_summary"
        methods[objective] = summarize_runs(run_dirs, summary_dir)

    suite = {
        "objectives": list(args.objectives),
        "seeds": list(args.seeds),
        "episodes": args.episodes,
        "epsilon_decay_fraction": args.epsilon_decay_fraction,
        "methods": {
            objective: {
                "signal_gate_pass_count": summary["signal_gate_pass_count"],
                "multiseed_gate_passed": summary["multiseed_gate_passed"],
                "matched_rate_mean": summary["matched_rate_mean"],
                "matched_rate_min": summary["matched_rate_min"],
                "matched_rate_by_outcome_mean": summary[
                    "matched_rate_by_outcome_mean"
                ],
                "matched_rate_by_outcome_min": summary[
                    "matched_rate_by_outcome_min"
                ],
                "image": summary["image"],
            }
            for objective, summary in methods.items()
        },
    }
    summary_path = args.output_dir / "suite_summary.json"
    summary_path.write_text(json.dumps(suite, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path.resolve()), **suite}, indent=2))


if __name__ == "__main__":
    main()
