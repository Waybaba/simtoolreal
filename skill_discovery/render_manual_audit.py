"""Render a stratified 30-trajectory contact sheet for manual label auditing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from skill_discovery.generate_point_cup_dataset import _write_png
from skill_discovery.point_cup import PointCupEnv, ShapeWorldConfig


CLASS_COLORS = np.asarray(
    [
        (104, 117, 125),
        (36, 119, 166),
        (49, 142, 102),
        (205, 110, 47),
    ],
    dtype=np.uint8,
)


def _sample_rows(payload: np.lib.npyio.NpzFile, seed: int) -> list[dict[str, int | str]]:
    rng = np.random.default_rng(seed)
    class_counts = (8, 8, 7, 7)
    rows: list[dict[str, int | str]] = []
    for class_id, count in enumerate(class_counts):
        audit_ids = np.flatnonzero(payload["audit_episode_class"] == class_id)
        nuisance_ids = np.flatnonzero(payload["nuisance_episode_class"] == class_id)
        audit_count = count // 2
        nuisance_count = count - audit_count
        rows.extend(
            {"source": "audit", "index": int(index), "class_id": class_id}
            for index in rng.choice(audit_ids, size=audit_count, replace=False)
        )
        rows.extend(
            {"source": "nuisance", "index": int(index), "class_id": class_id}
            for index in rng.choice(nuisance_ids, size=nuisance_count, replace=False)
        )
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--frame-size", type=int, default=48)
    args = parser.parse_args()

    payload = np.load(args.dataset)
    config = json.loads((args.dataset.parent / "config.json").read_text(encoding="utf-8"))
    class_names = [str(value) for value in payload["class_names"]]
    rows = _sample_rows(payload, args.seed)
    row_count = len(rows)
    steps = payload["audit_states"].shape[1]
    frame_indices = (0, steps // 2, steps - 1)

    positions = np.empty((row_count, len(frame_indices), 2), dtype=np.float32)
    centers = np.empty((row_count, 2), dtype=np.float32)
    axes = np.empty((row_count, 2), dtype=np.float32)
    manifest_rows = []
    fixed_center = np.asarray(config["environment"]["cup_center"], dtype=np.float32)
    fixed_axes = np.asarray(config["environment"]["cup_axes"], dtype=np.float32)

    for row_index, row in enumerate(rows):
        source = str(row["source"])
        index = int(row["index"])
        class_id = int(row["class_id"])
        states = payload[f"{source}_states"][index]
        positions[row_index] = states[list(frame_indices)]
        if source == "nuisance":
            centers[row_index] = payload["nuisance_cup_centers"][index]
            axes[row_index] = payload["nuisance_cup_axes"][index]
        else:
            centers[row_index] = fixed_center
            axes[row_index] = fixed_axes
        observed_class = int(payload[f"{source}_observed_class"][index])
        manifest_rows.append(
            {
                "row": row_index,
                "source": source,
                "source_index": index,
                "intent_class": class_names[class_id],
                "observed_class": class_names[observed_class],
                "matches": class_id == observed_class,
                "cup_center": centers[row_index].tolist(),
                "cup_axes": axes[row_index].tolist(),
                "frame_indices": list(frame_indices),
            }
        )

    render_config = ShapeWorldConfig(
        num_envs=row_count,
        episode_length=1,
        cup_center=tuple(fixed_center),
        cup_axes=tuple(fixed_axes),
    )
    env = PointCupEnv(render_config)
    env.set_layout(centers=centers, axes=axes)
    frame_batches = []
    for frame_index in range(len(frame_indices)):
        env.reset(positions=positions[:, frame_index])
        frame_batches.append(env.render(size=args.frame_size))

    gap = 3
    marker_width = 7
    sheet_width = marker_width + len(frame_indices) * args.frame_size + (len(frame_indices) - 1) * gap
    sheet_height = row_count * args.frame_size + (row_count - 1) * gap
    sheet = np.full((sheet_height, sheet_width, 3), 255, dtype=np.uint8)
    for row_index, row in enumerate(rows):
        y = row_index * (args.frame_size + gap)
        sheet[y : y + args.frame_size, :marker_width] = CLASS_COLORS[int(row["class_id"])]
        for frame_index, batch in enumerate(frame_batches):
            x = marker_width + frame_index * (args.frame_size + gap)
            sheet[y : y + args.frame_size, x : x + args.frame_size] = batch[row_index]

    output_path = args.dataset.parent / "manual_audit_30.png"
    _write_png(output_path, sheet)
    audit_output = {
        "dataset": str(args.dataset.resolve()),
        "image": str(output_path.resolve()),
        "seed": args.seed,
        "row_count": row_count,
        "all_labels_match": all(row["matches"] for row in manifest_rows),
        "legend": {name: CLASS_COLORS[index].tolist() for index, name in enumerate(class_names)},
        "rows": manifest_rows,
    }
    manifest_path = args.dataset.parent / "manual_audit_30.json"
    manifest_path.write_text(json.dumps(audit_output, indent=2), encoding="utf-8")
    print(json.dumps({key: audit_output[key] for key in ("image", "row_count", "all_labels_match")}, indent=2))


if __name__ == "__main__":
    main()
