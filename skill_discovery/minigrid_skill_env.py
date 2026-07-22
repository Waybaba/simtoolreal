"""Skill-conditioned wrappers and intrinsic rewards for MiniGrid DoorKey."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from minigrid.wrappers import FullyObsWrapper

from skill_discovery.minigrid_doorkey import DOORKEY_STAGES, semantic_stage


@dataclass(frozen=True)
class MiniGridSkillConfig:
    env_id: str = "MiniGrid-DoorKey-5x5-v0"
    num_skills: int = 4
    pseudocount: float = 2.0
    semantic_decay: float = 0.999
    semantic_coverage_weight: float = 1.0
    seed: int = 7


class OnlineIntrinsicReward:
    """Online DIAYN or semantic-spread reward shared by vector envs."""

    def __init__(self, objective: str, config: MiniGridSkillConfig):
        if objective not in {
            "random",
            "raw",
            "semantic",
            "semantic_spread",
            "semantic_balanced",
        }:
            raise ValueError("unknown objective")
        self.objective = objective
        self.config = config
        self.semantic_counts = np.full(
            (config.num_skills, len(DOORKEY_STAGES)),
            config.pseudocount,
            dtype=np.float64,
        )
        self.raw_counts: dict[int, np.ndarray] = {}
        self.episode_counts = np.zeros(config.num_skills, dtype=np.int64)
        self.stage_counts = np.zeros(len(DOORKEY_STAGES), dtype=np.int64)
        self.balanced_targets = np.random.default_rng(
            config.seed + 60000
        ).permutation(config.num_skills).astype(np.int64)

    def reward(self, skill: int, stage: int, raw_feature: int) -> tuple[float, dict[str, float]]:
        self.episode_counts[skill] += 1
        self.stage_counts[stage] += 1
        if self.objective == "random":
            return 0.0, {"diayn_reward": 0.0, "coverage_reward": 0.0}
        if self.objective == "semantic_balanced":
            reward = float(stage == self.balanced_targets[skill])
            return reward, {
                "diayn_reward": reward,
                "coverage_reward": 0.0,
            }
        if self.objective == "raw":
            counts = self.raw_counts.setdefault(
                raw_feature,
                np.full(self.config.num_skills, self.config.pseudocount, dtype=np.float64),
            )
            counts[skill] += 1.0
            posterior = counts[skill] / counts.sum()
            diayn_reward = math.log(max(posterior * self.config.num_skills, 1.0e-8))
            return diayn_reward, {"diayn_reward": diayn_reward, "coverage_reward": 0.0}

        self.semantic_counts *= self.config.semantic_decay
        self.semantic_counts[skill, stage] += 1.0
        posterior = self.semantic_counts[skill, stage] / self.semantic_counts[:, stage].sum()
        diayn_reward = math.log(max(posterior * self.config.num_skills, 1.0e-8))
        coverage_reward = 0.0
        if self.objective == "semantic_spread":
            stage_totals = self.semantic_counts.sum(axis=0)
            stage_probability = stage_totals[stage] / stage_totals.sum()
            coverage_reward = -math.log(
                max(len(DOORKEY_STAGES) * stage_probability, 1.0e-8)
            )
        total = diayn_reward + self.config.semantic_coverage_weight * coverage_reward
        return total, {
            "diayn_reward": diayn_reward,
            "coverage_reward": coverage_reward,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "semantic_counts": self.semantic_counts.tolist(),
            "raw_feature_count": len(self.raw_counts),
            "episode_counts": self.episode_counts.tolist(),
            "stage_counts": self.stage_counts.tolist(),
            "balanced_targets": self.balanced_targets.tolist()
            if self.objective == "semantic_balanced"
            else None,
        }


class DoorKeySkillWrapper(gym.Wrapper):
    """Expose a full object-centric grid plus a fixed latent skill."""

    def __init__(
        self,
        config: MiniGridSkillConfig,
        skill_id: int,
        reward_model: OnlineIntrinsicReward,
        *,
        render_mode: str | None = None,
    ):
        if skill_id < 0 or skill_id >= config.num_skills:
            raise ValueError("skill_id is out of range")
        base = gym.make(config.env_id, render_mode=render_mode)
        super().__init__(FullyObsWrapper(base))
        self.config = config
        self.skill_id = skill_id
        self.reward_model = reward_model
        self.furthest_stage = 0
        self._object_count = max(OBJECT_TO_IDX.values()) + 1
        self._color_count = max(COLOR_TO_IDX.values()) + 1
        grid_shape = self.env.observation_space["image"].shape
        self._grid_cells = int(grid_shape[0] * grid_shape[1])
        feature_count = self._grid_cells * (
            self._object_count + self._color_count + 3
        )
        feature_count += 4 + config.num_skills
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(feature_count,),
            dtype=np.float32,
        )

    def _encode(self, observation: dict[str, object]) -> np.ndarray:
        image = np.asarray(observation["image"], dtype=np.int64).reshape(-1, 3)
        object_features = np.eye(self._object_count, dtype=np.float32)[image[:, 0]]
        color_features = np.eye(self._color_count, dtype=np.float32)[image[:, 1]]
        state_features = np.eye(3, dtype=np.float32)[np.clip(image[:, 2], 0, 2)]
        direction = np.eye(4, dtype=np.float32)[int(observation["direction"])]
        skill = np.eye(self.config.num_skills, dtype=np.float32)[self.skill_id]
        return np.concatenate(
            (
                object_features.reshape(-1),
                color_features.reshape(-1),
                state_features.reshape(-1),
                direction,
                skill,
            )
        )

    @staticmethod
    def _raw_feature(observation: dict[str, object]) -> int:
        image = np.asarray(observation["image"], dtype=np.uint8)
        direction = int(observation["direction"]).to_bytes(1, "little")
        digest = hashlib.blake2b(image.tobytes() + direction, digest_size=8).digest()
        return int.from_bytes(digest, "little")

    def reset(self, **kwargs: object) -> tuple[np.ndarray, dict[str, object]]:
        observation, info = self.env.reset(**kwargs)
        self.furthest_stage = semantic_stage(self.env)
        return self._encode(observation), info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        observation, native_reward, terminated, truncated, info = self.env.step(action)
        stage = semantic_stage(
            self.env,
            terminated=terminated,
            reward=float(native_reward),
        )
        self.furthest_stage = max(self.furthest_stage, stage)
        training_reward = 0.0
        reward_parts = {"diayn_reward": 0.0, "coverage_reward": 0.0}
        if terminated or truncated:
            training_reward, reward_parts = self.reward_model.reward(
                self.skill_id,
                self.furthest_stage,
                self._raw_feature(observation),
            )
        info = {
            **info,
            "skill_id": self.skill_id,
            "semantic_stage": stage,
            "furthest_stage": self.furthest_stage,
            "native_reward": float(native_reward),
            "intrinsic_reward": float(training_reward),
            **reward_parts,
        }
        return (
            self._encode(observation),
            float(training_reward),
            terminated,
            truncated,
            info,
        )
