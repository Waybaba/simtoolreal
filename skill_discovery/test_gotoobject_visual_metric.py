"""Tests for GoToObject balanced RGB data and object-graph parsing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from minigrid.core.actions import Actions

from skill_discovery.evaluate_gotoobject_object_graph import (
    build_tile_templates,
    classification_metrics,
    parse_object_graph,
)
from skill_discovery.generate_gotoobject_visual_dataset import (
    GoToObjectVisualDatasetConfig,
    collect_scripted_reference,
    generate_dataset,
)
from skill_discovery.minigrid_doorkey import plan_to_face
from skill_discovery.minigrid_gotoobject import (
    floor_objects,
    make_gotoobject,
    plan_to_far,
    semantic_stage,
)


class GoToObjectVisualMetricTest(unittest.TestCase):
    def test_template_parser_recovers_scripted_rgb_stages(self) -> None:
        env = make_gotoobject(render_mode="rgb_array")
        templates = build_tile_templates()
        frames = []
        expected = []
        try:
            env.reset(seed=7)
            for action in plan_to_far(env):
                env.step(action)
            frames.append(env.render())
            expected.append(semantic_stage(env))
            selected = floor_objects(env)[0]
            for action in plan_to_face(env, selected.position):
                env.step(action)
            frames.append(env.render())
            expected.append(semantic_stage(env))
            env.step(int(Actions.pickup))
            frames.append(env.render())
            expected.append(semantic_stage(env))
        finally:
            env.close()
        parsed = [parse_object_graph(frame, templates=templates) for frame in frames]
        self.assertEqual(expected, [0, 1, 2])
        self.assertEqual([row["stage"] for row in parsed], expected)
        self.assertEqual([row["floor_count"] for row in parsed], [2, 2, 1])
        self.assertTrue(all(len(row["agents"]) == 1 for row in parsed))
        self.assertTrue(all(row["exact_tile_fraction"] == 1.0 for row in parsed))

    def test_small_dataset_is_balanced_unique_and_disjoint(self) -> None:
        config = GoToObjectVisualDatasetConfig(
            samples_per_stage=2,
            max_reference_episodes=100,
            max_audit_episodes=1_000,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = generate_dataset(config, Path(temporary) / "dataset")
            with np.load(output["dataset"]) as data:
                stages = data["stages"]
                splits = data["splits"]
                hashes = data["frame_hashes"]
        self.assertTrue(output["data_gate_passed"])
        self.assertEqual(output["split_stage_counts"], {
            "scripted_reference": [2, 2, 2],
            "random_exploration_audit": [2, 2, 2],
        })
        reference_hashes = set(hashes[splits == 0].tolist())
        audit_hashes = set(hashes[splits == 1].tolist())
        self.assertFalse(reference_hashes & audit_hashes)
        self.assertEqual(np.bincount(stages, minlength=3).tolist(), [4, 4, 4])

    def test_reference_skips_layout_without_far_state(self) -> None:
        config = GoToObjectVisualDatasetConfig(
            samples_per_stage=1,
            reference_seed_start=200_170,
            max_reference_episodes=10,
        )
        samples, hashes, attempts, skipped = collect_scripted_reference(config)
        self.assertEqual(len(samples), 3)
        self.assertEqual(len(hashes), 3)
        self.assertGreaterEqual(attempts, 2)
        self.assertEqual(skipped, 1)

    def test_classification_metrics_reports_invalid_predictions(self) -> None:
        metrics = classification_metrics(
            np.asarray([0, 1, -1]),
            np.asarray([0, 1, 2]),
        )
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["invalid_predictions"], 1)
        self.assertEqual(metrics["recall_by_stage"]["object_carried"], 0.0)


if __name__ == "__main__":
    unittest.main()
