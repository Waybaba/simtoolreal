"""Run the frozen visible-threshold MountainCar none capacity audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from skill_discovery.audit_mountaincar_stratified_none_capacity import (
    VISIBLE_STRATA,
    StratifiedNoneCapacityConfig,
    run_capacity_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = StratifiedNoneCapacityConfig()
    run_id = (
        "visible_stratified_none_capacity_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/mountaincar_continuous"
    ) / run_id
    output = run_capacity_audit(
        config,
        output_dir,
        workers=args.workers,
        strata=VISIBLE_STRATA,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
