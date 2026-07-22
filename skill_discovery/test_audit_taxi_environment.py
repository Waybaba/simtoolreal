"""Tests for the official Taxi-v4 environment audit."""

from __future__ import annotations

import unittest

import gymnasium as gym

from skill_discovery.audit_taxi_environment import (
    TaxiEnvironmentAuditConfig,
    enumerate_reachable_states,
    shortest_delivery_actions,
    taxi_semantic_stage,
)


class TaxiEnvironmentAuditTest(unittest.TestCase):
    def test_seeds_must_be_unique(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            TaxiEnvironmentAuditConfig(seeds=(7, 7))

    def test_official_reachable_state_counts(self) -> None:
        env = gym.make("Taxi-v4")
        try:
            output = enumerate_reachable_states(env.unwrapped)
        finally:
            env.close()
        self.assertEqual(output["initial_state_count"], 300)
        self.assertEqual(output["reachable_state_count"], 404)
        self.assertEqual(output["terminal_state_count"], 4)
        self.assertTrue(output["dry_transitions_deterministic"])

    def test_shortest_script_has_one_pickup_and_dropoff(self) -> None:
        env = gym.make("Taxi-v4")
        try:
            state, _ = env.reset(seed=7)
            base = env.unwrapped
            actions = shortest_delivery_actions(base, state)
            stages = [taxi_semantic_stage(base, state)]
            for action in actions:
                state, _, _, _, _ = env.step(action)
                stages.append(taxi_semantic_stage(base, state))
        finally:
            env.close()
        self.assertEqual(actions.count(4), 1)
        self.assertEqual(actions.count(5), 1)
        self.assertEqual(stages[0], 0)
        self.assertIn(1, stages)
        self.assertEqual(stages[-1], 2)


if __name__ == "__main__":
    unittest.main()
