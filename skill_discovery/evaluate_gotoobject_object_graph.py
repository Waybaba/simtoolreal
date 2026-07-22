"""Evaluate an RGB template object graph against full-frame visual baselines."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.world_object import Ball, Box, Key, Wall

from skill_discovery.generate_gotoobject_visual_dataset import SPLIT_NAMES
from skill_discovery.generate_point_cup_dataset import _write_png
from skill_discovery.minigrid_gotoobject import GOTOOBJECT_STAGES


GRID_SIZE = 6
TILE_SIZE = 32


@dataclass(frozen=True)
class TileLabel:
    kind: str
    object_type: str = ""
    color: str = ""
    agent_direction: int = -1


def build_tile_templates() -> tuple[dict[bytes, TileLabel], np.ndarray, tuple[TileLabel, ...]]:
    rows: list[tuple[np.ndarray, TileLabel]] = []
    for highlight in (False, True):
        rows.append((Grid.render_tile(None, highlight=highlight, tile_size=TILE_SIZE), TileLabel("empty")))
        rows.append(
            (
                Grid.render_tile(Wall(), highlight=highlight, tile_size=TILE_SIZE),
                TileLabel("wall"),
            )
        )
        for direction in range(4):
            rows.append(
                (
                    Grid.render_tile(
                        None,
                        agent_dir=direction,
                        highlight=highlight,
                        tile_size=TILE_SIZE,
                    ),
                    TileLabel("agent", agent_direction=direction),
                )
            )
        for object_type, object_class in (("key", Key), ("ball", Ball), ("box", Box)):
            for color in COLOR_NAMES:
                rows.append(
                    (
                        Grid.render_tile(
                            object_class(color),
                            highlight=highlight,
                            tile_size=TILE_SIZE,
                        ),
                        TileLabel("object", object_type=object_type, color=color),
                    )
                )
    lookup: dict[bytes, TileLabel] = {}
    unique_images = []
    unique_labels = []
    for image, label in rows:
        image = np.asarray(image, dtype=np.uint8)
        digest = image.tobytes()
        existing = lookup.get(digest)
        if existing is not None and existing != label:
            raise RuntimeError("ambiguous MiniGrid visual templates")
        if existing is None:
            lookup[digest] = label
            unique_images.append(image)
            unique_labels.append(label)
    return lookup, np.stack(unique_images), tuple(unique_labels)


def _nearest_template(
    tile: np.ndarray,
    template_images: np.ndarray,
    template_labels: Sequence[TileLabel],
) -> tuple[TileLabel, float]:
    delta = template_images.astype(np.int16) - tile.astype(np.int16)
    mse = np.mean(delta.astype(np.float32) ** 2, axis=(1, 2, 3))
    index = int(np.argmin(mse))
    return template_labels[index], float(mse[index])


def parse_object_graph(
    frame: np.ndarray,
    *,
    templates: tuple[dict[bytes, TileLabel], np.ndarray, tuple[TileLabel, ...]] | None = None,
) -> dict[str, object]:
    if frame.shape != (GRID_SIZE * TILE_SIZE, GRID_SIZE * TILE_SIZE, 3):
        raise ValueError("GoToObject frame must be 192x192 RGB")
    lookup, template_images, template_labels = templates or build_tile_templates()
    agents = []
    objects = []
    exact_tiles = 0
    max_template_mse = 0.0
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            tile = frame[
                y * TILE_SIZE : (y + 1) * TILE_SIZE,
                x * TILE_SIZE : (x + 1) * TILE_SIZE,
            ]
            label = lookup.get(tile.tobytes())
            if label is None:
                label, mse = _nearest_template(tile, template_images, template_labels)
                max_template_mse = max(max_template_mse, mse)
            else:
                exact_tiles += 1
            if label.kind == "agent":
                agents.append((x, y, label.agent_direction))
            elif label.kind == "object":
                objects.append((x, y, label.object_type, label.color))
    floor_count = len(objects)
    if len(agents) != 1:
        stage = -1
    elif floor_count < 2:
        stage = 2
    else:
        agent_x, agent_y, _ = agents[0]
        nearest = min(abs(agent_x - row[0]) + abs(agent_y - row[1]) for row in objects)
        stage = 1 if nearest == 1 else 0
    return {
        "agents": agents,
        "objects": sorted(objects),
        "floor_count": floor_count,
        "stage": stage,
        "exact_tile_fraction": exact_tiles / (GRID_SIZE * GRID_SIZE),
        "max_fallback_template_mse": max_template_mse,
    }


def classification_metrics(predictions: np.ndarray, stages: np.ndarray) -> dict[str, object]:
    predictions = np.asarray(predictions, dtype=np.int64)
    stages = np.asarray(stages, dtype=np.int64)
    if predictions.shape != stages.shape or predictions.ndim != 1:
        raise ValueError("prediction and stage vectors must align")
    confusion = np.zeros((3, 3), dtype=np.int64)
    for expected, predicted in zip(stages, predictions):
        if 0 <= predicted < 3:
            confusion[int(expected), int(predicted)] += 1
    recalls = {
        name: float(confusion[index, index] / max(confusion[index].sum(), 1))
        for index, name in enumerate(GOTOOBJECT_STAGES)
    }
    return {
        "accuracy": float(np.mean(predictions == stages)),
        "macro_recall": float(np.mean(list(recalls.values()))),
        "recall_by_stage": recalls,
        "confusion": confusion.tolist(),
        "predicted_class_sizes": np.bincount(
            predictions[predictions >= 0], minlength=3
        ).tolist(),
        "invalid_predictions": int(np.sum(predictions < 0)),
    }


def _normalize(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1.0e-8)


def raw_frame_features(frames: np.ndarray) -> np.ndarray:
    y = np.linspace(0, frames.shape[1] - 1, 32).astype(np.int64)
    x = np.linspace(0, frames.shape[2] - 1, 32).astype(np.int64)
    return _normalize(frames[:, y][:, :, x].reshape(len(frames), -1).astype(np.float32))


def reference_centroid_predictions(
    features: np.ndarray,
    stages: np.ndarray,
    splits: np.ndarray,
) -> np.ndarray:
    features = _normalize(features)
    reference = splits == 0
    centers = _normalize(
        np.stack([features[reference & (stages == stage)].mean(axis=0) for stage in range(3)])
    )
    return np.argmax(features @ centers.T, axis=1).astype(np.int64)


def _true_object_positions(row: np.ndarray) -> list[tuple[int, int]]:
    return sorted(
        (int(x), int(y)) for x, y in row if int(x) >= 0 and int(y) >= 0
    )


def _draw_border(frame: np.ndarray, x: int, y: int, color: tuple[int, int, int]) -> None:
    x0, y0 = x * TILE_SIZE, y * TILE_SIZE
    frame[y0 : y0 + 3, x0 : x0 + TILE_SIZE] = color
    frame[y0 + TILE_SIZE - 3 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE] = color
    frame[y0 : y0 + TILE_SIZE, x0 : x0 + 3] = color
    frame[y0 : y0 + TILE_SIZE, x0 + TILE_SIZE - 3 : x0 + TILE_SIZE] = color


def _write_parser_sheet(
    path: Path,
    frames: np.ndarray,
    stages: np.ndarray,
    splits: np.ndarray,
    parsed: list[dict[str, object]],
) -> None:
    gap, marker, columns = 6, 10, 4
    sheet = np.full(
        (3 * 192 + 2 * gap, marker + columns * 192 + (columns - 1) * gap, 3),
        255,
        dtype=np.uint8,
    )
    colors = ((104, 117, 125), (40, 117, 164), (43, 137, 95))
    for stage in range(3):
        y = stage * (192 + gap)
        sheet[y : y + 192, :marker] = colors[stage]
        candidates = np.flatnonzero((splits == 1) & (stages == stage))[:columns]
        for column, index in enumerate(candidates):
            frame = frames[index].copy()
            for agent_x, agent_y, _ in parsed[index]["agents"]:
                _draw_border(frame, agent_x, agent_y, (255, 224, 64))
            for object_x, object_y, _, _ in parsed[index]["objects"]:
                _draw_border(frame, object_x, object_y, (64, 220, 120))
            x = marker + column * (192 + gap)
            sheet[y : y + 192, x : x + 192] = frame
    _write_png(path, sheet)


def _write_chart(path: Path, methods: dict[str, dict[str, object]]) -> None:
    labels = ("accuracy", *GOTOOBJECT_STAGES)
    colors = ("#68757d", "#2875a4", "#2b895f")
    width, height = 900, 430
    left, top, chart_width, chart_height = 70, 65, 760, 280
    elements = [f'<rect width="{width}" height="{height}" fill="#f8fafb"/>']
    for group, label in enumerate(labels):
        group_x = left + group * chart_width / len(labels)
        for method_index, (method, metrics) in enumerate(methods.items()):
            value = (
                float(metrics["accuracy"])
                if label == "accuracy"
                else float(metrics["recall_by_stage"][label])
            )
            x = group_x + 18 + method_index * 34
            bar_height = value * chart_height
            elements.append(
                f'<rect x="{x:.1f}" y="{top + chart_height - bar_height:.1f}" width="25" height="{bar_height:.1f}" fill="{colors[method_index]}"/>'
            )
        elements.append(
            f'<text x="{group_x + 55:.1f}" y="{top + chart_height + 24}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#26343d">{label}</text>'
        )
    elements.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#697780"/>',
            f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#697780"/>',
            '<text x="70" y="34" font-family="sans-serif" font-size="22" fill="#172027">GoToObject random-exploration visual metric</text>',
        ]
    )
    for index, method in enumerate(methods):
        x = 90 + index * 250
        elements.append(
            f'<rect x="{x}" y="392" width="14" height="14" fill="{colors[index]}"/><text x="{x + 22}" y="404" font-family="sans-serif" font-size="12" fill="#26343d">{method}</text>'
        )
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        + "".join(elements)
        + "</svg>\n",
        encoding="utf-8",
    )


def evaluate_dataset(
    dataset_path: Path,
    output_dir: Path,
    *,
    dino_embeddings_path: Path | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(dataset_path) as data:
        frames = data["frames"].astype(np.uint8)
        stages = data["stages"].astype(np.int64)
        splits = data["splits"].astype(np.int64)
        hashes = data["frame_hashes"]
        agent_positions = data["agent_positions"].astype(np.int64)
        agent_directions = data["agent_directions"].astype(np.int64)
        floor_counts = data["floor_counts"].astype(np.int64)
        floor_positions = data["floor_positions"].astype(np.int64)
        floor_types = data["floor_types"]
        floor_colors = data["floor_colors"]
    templates = build_tile_templates()
    parsed = [parse_object_graph(frame, templates=templates) for frame in frames]
    object_graph_predictions = np.asarray([row["stage"] for row in parsed], dtype=np.int64)
    predicted_agents = [row["agents"] for row in parsed]
    agent_exact = np.asarray(
        [
            len(rows) == 1
            and tuple(rows[0][:2]) == tuple(agent_positions[index])
            and rows[0][2] == agent_directions[index]
            for index, rows in enumerate(predicted_agents)
        ]
    )
    count_exact = np.asarray(
        [row["floor_count"] == floor_counts[index] for index, row in enumerate(parsed)]
    )
    positions_exact = np.asarray(
        [
            sorted((row[0], row[1]) for row in parsed[index]["objects"])
            == _true_object_positions(floor_positions[index])
            for index in range(len(parsed))
        ]
    )
    audit = splits == 1
    methods = {
        "raw_current_reference": classification_metrics(
            reference_centroid_predictions(raw_frame_features(frames), stages, splits)[audit],
            stages[audit],
        ),
        "rgb_template_object_graph": classification_metrics(
            object_graph_predictions[audit], stages[audit]
        ),
    }
    dino_metadata = None
    if dino_embeddings_path is not None:
        with np.load(dino_embeddings_path) as embeddings_data:
            embeddings = embeddings_data["embeddings"].astype(np.float32)
            embedding_hashes = embeddings_data["frame_hashes"]
        if embeddings.shape[0] != len(frames) or not np.array_equal(embedding_hashes, hashes):
            raise ValueError("DINO embeddings do not align with the RGB dataset")
        methods["dinov2_current_reference"] = classification_metrics(
            reference_centroid_predictions(embeddings, stages, splits)[audit],
            stages[audit],
        )
        dino_metadata = str(dino_embeddings_path.resolve())

    graph = methods["rgb_template_object_graph"]
    parser_gate = bool(
        agent_exact[audit].mean() >= 0.99
        and count_exact[audit].mean() >= 0.99
        and all(value > 0 for value in graph["predicted_class_sizes"])
    )
    semantic_gate = bool(
        graph["accuracy"] >= 0.98
        and all(value >= 0.98 for value in graph["recall_by_stage"].values())
        and graph["confusion"][0][2] / max(sum(graph["confusion"][0]), 1) <= 0.01
    )
    dino = methods.get("dinov2_current_reference")
    advantage_gate = (
        bool(graph["macro_recall"] >= dino["macro_recall"] + 0.10)
        if dino is not None
        else None
    )
    image_path = output_dir / "gotoobject_object_graph_audit.png"
    _write_parser_sheet(image_path, frames, stages, splits, parsed)
    chart_path = output_dir / "gotoobject_visual_metric.svg"
    chart_methods = {
        name: methods[name]
        for name in (
            "raw_current_reference",
            "dinov2_current_reference",
            "rgb_template_object_graph",
        )
        if name in methods
    }
    _write_chart(chart_path, chart_methods)
    output = {
        "dataset": str(dataset_path.resolve()),
        "dino_embeddings": dino_metadata,
        "stage_names": GOTOOBJECT_STAGES,
        "split_names": SPLIT_NAMES,
        "frame_count": len(frames),
        "audit_count": int(audit.sum()),
        "template_count": len(templates[0]),
        "dataset_coverage": {
            SPLIT_NAMES[split]: {
                "object_types": sorted(
                    set(floor_types[splits == split].ravel().tolist()) - {""}
                ),
                "object_colors": sorted(
                    set(floor_colors[splits == split].ravel().tolist()) - {""}
                ),
                "agent_directions": sorted(
                    set(agent_directions[splits == split].tolist())
                ),
                "agent_position_count": len(
                    set(map(tuple, agent_positions[splits == split].tolist()))
                ),
            }
            for split in range(2)
        },
        "parser": {
            "agent_exact_rate": float(agent_exact[audit].mean()),
            "floor_count_exact_rate": float(count_exact[audit].mean()),
            "floor_position_exact_rate": float(positions_exact[audit].mean()),
            "all_tiles_exact_template_rate": float(
                np.mean(
                    [
                        parsed[index]["exact_tile_fraction"] == 1.0
                        for index in np.flatnonzero(audit)
                    ]
                )
            ),
        },
        "methods": methods,
        "parser_gate_passed": parser_gate,
        "semantic_gate_passed": semantic_gate,
        "structure_advantage_gate_passed": advantage_gate,
        "final_gate_evaluated": dino is not None,
        "final_gate_passed": bool(parser_gate and semantic_gate and advantage_gate)
        if advantage_gate is not None
        else None,
        "manual_audit_image": str(image_path.resolve()),
        "chart": str(chart_path.resolve()),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["metrics"] = str(metrics_path.resolve())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--dino-embeddings", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/minigrid_gotoobject_visual"
    ) / f"gotoobject_object_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = evaluate_dataset(
        args.dataset,
        output_dir,
        dino_embeddings_path=args.dino_embeddings,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
