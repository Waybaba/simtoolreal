"""Train fixed DoorKey controls with MaskablePPO and option termination."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
import torch
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from minigrid.wrappers import FullyObsWrapper
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from skill_discovery.generate_point_cup_dataset import _write_png
from skill_discovery.minigrid_doorkey import DOORKEY_STAGES, semantic_stage
from skill_discovery.train_minigrid_doorkey_tabular import (
    DOORKEY_CONTROL_MATRIX,
    DOORKEY_POLICY_ACTIONS,
    final_state_success,
    state_changing_action_indices,
)


@dataclass(frozen=True)
class MaskedPPOConfig:
    env_id: str = "MiniGrid-DoorKey-5x5-v0"
    seed: int = 7
    num_skills: int = 4
    n_envs: int = 8
    total_timesteps: int = 250_000
    horizon: int = 64
    n_steps: int = 256
    batch_size: int = 256
    n_epochs: int = 4
    learning_rate: float = 2.5e-4
    ent_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    net_arch: tuple[int, int] = (256, 256)
    eval_interval: int = 50_000
    checkpoint_eval_episodes_per_skill: int = 64
    final_eval_episodes_per_skill: int = 256
    stability_checkpoints: int = 3
    stage_rate_gate: float = 0.80
    torch_threads: int = 4
    target_assignment: tuple[int, ...] = (0, 1, 2, 3)
    reward_matrix: tuple[tuple[float, ...], ...] = DOORKEY_CONTROL_MATRIX

    def __post_init__(self) -> None:
        if self.num_skills != len(DOORKEY_STAGES):
            raise ValueError("DoorKey masked PPO is fixed to four skills")
        if self.n_envs < self.num_skills or self.n_envs % self.num_skills != 0:
            raise ValueError("n_envs must be a positive multiple of four")
        if self.total_timesteps <= 0 or self.horizon <= 0:
            raise ValueError("training budget and horizon must be positive")
        if self.n_steps <= 0 or self.batch_size <= 0 or self.n_epochs <= 0:
            raise ValueError("PPO rollout and update sizes must be positive")
        if self.batch_size > self.n_steps * self.n_envs:
            raise ValueError("batch size cannot exceed rollout size")
        if self.eval_interval <= 0:
            raise ValueError("evaluation interval must be positive")
        if self.total_timesteps % self.eval_interval != 0:
            raise ValueError("total timesteps must be divisible by eval interval")
        if self.total_timesteps // self.eval_interval < self.stability_checkpoints:
            raise ValueError("not enough checkpoints for stability gate")
        if sorted(self.target_assignment) != list(range(self.num_skills)):
            raise ValueError("target assignment must be a stage permutation")
        matrix = np.asarray(self.reward_matrix, dtype=np.float64)
        if matrix.shape != (self.num_skills, len(DOORKEY_STAGES)):
            raise ValueError("reward matrix must have shape (4, 4)")
        if not np.isfinite(matrix).all():
            raise ValueError("reward matrix must be finite")
        if not 0 <= self.stage_rate_gate <= 1:
            raise ValueError("stage rate gate must be in [0, 1]")


class MaskedDoorKeyControlEnv(gym.Wrapper):
    """Expose full-grid features and a state-changing official action mask."""

    def __init__(
        self,
        config: MaskedPPOConfig,
        skill_id: int,
        *,
        render_mode: str | None = None,
    ):
        if skill_id < 0 or skill_id >= config.num_skills:
            raise ValueError("skill id is out of range")
        base = gym.make(config.env_id, render_mode=render_mode)
        super().__init__(FullyObsWrapper(base))
        self.config = config
        self.skill_id = skill_id
        self.target_stage = config.target_assignment[skill_id]
        self.furthest_stage = 0
        self.option_steps = 0
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

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=np.bool_)
        for action_index in state_changing_action_indices(self):
            mask[DOORKEY_POLICY_ACTIONS[action_index]] = True
        return mask

    def reset(self, **kwargs: object) -> tuple[np.ndarray, dict[str, object]]:
        observation, info = self.env.reset(**kwargs)
        self.furthest_stage = semantic_stage(self)
        self.option_steps = 0
        return self._encode(observation), info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if not self.action_masks()[int(action)]:
            raise ValueError("policy selected a masked DoorKey action")
        observation, native_reward, native_terminated, native_truncated, info = (
            self.env.step(int(action))
        )
        self.option_steps += 1
        stage = semantic_stage(
            self,
            terminated=bool(native_terminated),
            reward=float(native_reward),
        )
        self.furthest_stage = max(self.furthest_stage, stage)
        option_terminated = bool(
            self.target_stage in {1, 2}
            and self.furthest_stage == self.target_stage
        )
        horizon_truncated = bool(
            self.option_steps >= self.config.horizon
            and not native_terminated
            and not native_truncated
            and not option_terminated
        )
        terminated = bool(native_terminated or option_terminated)
        truncated = bool(native_truncated or horizon_truncated)
        reward = float(
            self.config.reward_matrix[self.skill_id][self.furthest_stage]
        )
        carrying = self.unwrapped.carrying
        carrying_value = None
        if carrying is not None:
            carrying_value = [carrying.type, carrying.color]
        objects = (
            self.unwrapped.grid.get(x, y)
            for x in range(self.unwrapped.width)
            for y in range(self.unwrapped.height)
        )
        door_open = any(
            obj is not None and obj.type == "door" and bool(obj.is_open)
            for obj in objects
        )
        info = {
            **info,
            "skill_id": self.skill_id,
            "target_stage": self.target_stage,
            "semantic_stage": stage,
            "furthest_stage": self.furthest_stage,
            "native_reward": float(native_reward),
            "native_success": bool(native_terminated and native_reward > 0),
            "native_terminated": bool(native_terminated),
            "option_terminated": option_terminated,
            "horizon_truncated": horizon_truncated,
            "carrying": carrying_value,
            "door_open": door_open,
            "masked_control_reward": reward,
        }
        return self._encode(observation), reward, terminated, truncated, info


def _make_vec_env(config: MaskedPPOConfig) -> DummyVecEnv:
    env_fns = []
    for env_index in range(config.n_envs):
        skill_id = env_index % config.num_skills

        def make_env(skill: int = skill_id) -> Monitor:
            return Monitor(MaskedDoorKeyControlEnv(config, skill))

        env_fns.append(make_env)
    vec_env = DummyVecEnv(env_fns)
    vec_env.seed(config.seed)
    return vec_env


def _rollout_model(
    model: MaskablePPO,
    config: MaskedPPOConfig,
    skill: int,
    *,
    seed: int,
    render: bool = False,
) -> dict[str, object]:
    env = MaskedDoorKeyControlEnv(
        config,
        skill,
        render_mode="rgb_array" if render else None,
    )
    frames = []
    actions = []
    try:
        observation, _ = env.reset(seed=seed)
        if render:
            frames.append(env.render())
        terminated = truncated = False
        final_info: dict[str, object] = {
            "furthest_stage": env.furthest_stage,
            "native_success": False,
            "carrying": None,
            "door_open": False,
            "option_terminated": False,
            "horizon_truncated": False,
        }
        while not (terminated or truncated):
            mask = env.action_masks()
            action, _ = model.predict(
                observation,
                deterministic=True,
                action_masks=mask,
            )
            action_id = int(np.asarray(action).item())
            if not mask[action_id]:
                raise ValueError("MaskablePPO predicted an invalid action")
            observation, _, terminated, truncated, final_info = env.step(action_id)
            actions.append(action_id)
            if render:
                frames.append(env.render())
        return {
            "skill": skill,
            "target_stage": config.target_assignment[skill],
            "stage": int(final_info["furthest_stage"]),
            "steps": len(actions),
            "actions": actions,
            "native_success": bool(final_info["native_success"]),
            "carrying": final_info["carrying"],
            "door_open": bool(final_info["door_open"]),
            "option_terminated": bool(final_info["option_terminated"]),
            "horizon_truncated": bool(final_info["horizon_truncated"]),
            "frames": frames,
        }
    finally:
        env.close()


def evaluate_model(
    model: MaskablePPO,
    config: MaskedPPOConfig,
    *,
    episodes_per_skill: int,
    seed: int,
) -> dict[str, object]:
    rates = np.zeros((4, 4), dtype=np.float64)
    target_rates = []
    final_state_rates = []
    native_success_rates = []
    mean_steps = []
    for skill in range(config.num_skills):
        target = config.target_assignment[skill]
        stages = []
        final_successes = []
        native_successes = []
        steps = []
        for episode in range(episodes_per_skill):
            rollout = _rollout_model(
                model,
                config,
                skill,
                seed=seed + episode,
            )
            stages.append(int(rollout["stage"]))
            final_successes.append(final_state_success(target, rollout))
            native_successes.append(bool(rollout["native_success"]))
            steps.append(int(rollout["steps"]))
        rates[skill] = np.bincount(stages, minlength=4) / len(stages)
        target_rates.append(float(rates[skill, target]))
        final_state_rates.append(float(np.mean(final_successes)))
        native_success_rates.append(float(np.mean(native_successes)))
        mean_steps.append(float(np.mean(steps)))
    goal_skill = config.target_assignment.index(3)
    gate = bool(
        all(rate >= config.stage_rate_gate for rate in target_rates)
        and all(rate >= config.stage_rate_gate for rate in final_state_rates)
        and native_success_rates[goal_skill] >= config.stage_rate_gate
    )
    return {
        "episodes_per_skill": episodes_per_skill,
        "target_assignment": list(config.target_assignment),
        "target_assignment_names": [
            DOORKEY_STAGES[target] for target in config.target_assignment
        ],
        "outcome_rates": rates.tolist(),
        "target_stage_rates": target_rates,
        "target_rate_by_stage": {
            DOORKEY_STAGES[target]: target_rates[skill]
            for skill, target in enumerate(config.target_assignment)
        },
        "final_state_rates": final_state_rates,
        "final_state_rate_by_target": {
            DOORKEY_STAGES[target]: final_state_rates[skill]
            for skill, target in enumerate(config.target_assignment)
        },
        "native_success_rates": native_success_rates,
        "goal_skill": goal_skill,
        "goal_native_success_rate": native_success_rates[goal_skill],
        "mean_episode_steps": mean_steps,
        "specialization_gate_passed": gate,
    }


class EvaluationCallback(BaseCallback):
    def __init__(self, config: MaskedPPOConfig):
        super().__init__(verbose=0)
        self.config = config
        self.next_evaluation = config.eval_interval
        self.evaluations: list[dict[str, object]] = []
        self.terminal_stage_counts = np.zeros((4, 4), dtype=np.int64)
        self.native_success_counts = np.zeros(4, dtype=np.int64)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if bool(done):
                skill = int(info["skill_id"])
                stage = int(info["furthest_stage"])
                self.terminal_stage_counts[skill, stage] += 1
                self.native_success_counts[skill] += int(info["native_success"])
        if self.num_timesteps >= self.next_evaluation:
            evaluation = evaluate_model(
                self.model,
                self.config,
                episodes_per_skill=self.config.checkpoint_eval_episodes_per_skill,
                seed=self.config.seed + 100_000,
            )
            self.evaluations.append(
                {"timesteps": self.next_evaluation, **evaluation}
            )
            self.next_evaluation += self.config.eval_interval
        return True

    def state_dict(self) -> dict[str, object]:
        return {
            "evaluations": self.evaluations,
            "terminal_stage_counts": self.terminal_stage_counts.tolist(),
            "native_success_counts": self.native_success_counts.tolist(),
        }


def _write_curves(path: Path, evaluations: list[dict[str, object]]) -> None:
    width, height = 900, 420
    left, top, chart_width, chart_height = 65, 55, 770, 290
    colors = ("#68757d", "#2875a4", "#d17031", "#2b895f")
    lines = []
    for stage, name in enumerate(DOORKEY_STAGES):
        points = []
        for index, row in enumerate(evaluations):
            value = row["target_rate_by_stage"][name]
            x = left + chart_width * index / max(len(evaluations) - 1, 1)
            y = top + chart_height * (1.0 - value)
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{colors[stage]}" stroke-width="2"/>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#fbfcfd"/>'
        '<text x="30" y="30" font-family="sans-serif" font-size="19" '
        'fill="#172b3a">DoorKey masked PPO target-stage rates</text>'
        f'<line x1="{left}" y1="{top + chart_height}" '
        f'x2="{left + chart_width}" y2="{top + chart_height}" '
        'stroke="#83919a"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{top + chart_height}" stroke="#83919a"/>'
        f'{"".join(lines)}</svg>'
    )
    path.write_text(svg, encoding="utf-8")


def _write_rollout_audit(
    output_dir: Path,
    model: MaskablePPO,
    config: MaskedPPOConfig,
    target_rates: list[float],
) -> list[dict[str, object]]:
    columns = 4
    gap = 6
    marker = 9
    colors = ((104, 117, 125), (40, 117, 164), (209, 112, 49), (43, 137, 95))
    rows = []
    manifest = []
    for skill in range(config.num_skills):
        target = config.target_assignment[skill]
        rate = target_rates[skill]
        attempts = 1 if rate <= 0 else min(512, max(32, int(10 / rate)))
        rollout = None
        fallback = None
        matched_offset = -1
        for offset in range(attempts):
            candidate = _rollout_model(
                model,
                config,
                skill,
                seed=config.seed + 990_000 + offset,
                render=True,
            )
            fallback = fallback or candidate
            if (
                int(candidate["stage"]) == target
                and final_state_success(target, candidate)
            ):
                rollout = candidate
                matched_offset = offset
                break
        if rollout is None:
            assert fallback is not None
            rollout = fallback
        frames = rollout.pop("frames")
        indices = np.linspace(0, len(frames) - 1, columns).astype(np.int64)
        rows.append(
            (
                int(rollout["stage"]),
                [frames[int(index)] for index in indices],
            )
        )
        manifest.append(
            {
                **rollout,
                "target_stage_name": DOORKEY_STAGES[target],
                "actual_stage_name": DOORKEY_STAGES[int(rollout["stage"])],
                "target_state_found": matched_offset >= 0,
                "matched_seed_offset": matched_offset,
            }
        )
    frame_height, frame_width = rows[0][1][0].shape[:2]
    sheet = np.full(
        (
            4 * frame_height + 3 * gap,
            marker + columns * frame_width + (columns - 1) * gap,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    for skill, (actual, frames) in enumerate(rows):
        y = skill * (frame_height + gap)
        sheet[y : y + frame_height, :marker] = colors[actual]
        for column, frame in enumerate(frames):
            x = marker + column * (frame_width + gap)
            sheet[y : y + frame_height, x : x + frame_width] = frame
    _write_png(output_dir / "policy_rollout_audit.png", sheet)
    (output_dir / "policy_rollout_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def train_run(config: MaskedPPOConfig, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    vec_env = _make_vec_env(config)
    callback = EvaluationCallback(config)
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        ent_coef=config.ent_coef,
        policy_kwargs={"net_arch": list(config.net_arch)},
        seed=config.seed,
        device="cpu",
        verbose=0,
    )
    started = time.monotonic()
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callback,
            use_masking=True,
        )
        elapsed = time.monotonic() - started
        model.save(output_dir / "masked_ppo_policy")
        final = evaluate_model(
            model,
            config,
            episodes_per_skill=config.final_eval_episodes_per_skill,
            seed=config.seed + 900_000,
        )
        manifest = _write_rollout_audit(
            output_dir,
            model,
            config,
            final["target_stage_rates"],
        )
    finally:
        vec_env.close()
    recent = callback.evaluations[-config.stability_checkpoints :]
    stability = bool(
        len(recent) == config.stability_checkpoints
        and all(row["specialization_gate_passed"] for row in recent)
    )
    output = {
        "config": asdict(config),
        "versions": {
            package: importlib.metadata.version(package)
            for package in (
                "minigrid",
                "gymnasium",
                "torch",
                "stable-baselines3",
                "sb3-contrib",
            )
        },
        "elapsed_seconds": elapsed,
        "training_callback": callback.state_dict(),
        "evaluations": callback.evaluations,
        "final_evaluation": final,
        "checkpoint_stability_passed": stability,
        "signal_gate_passed": bool(final["specialization_gate_passed"] and stability),
        "policy_rollout_manifest": manifest,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    _write_curves(output_dir / "stage_curves.svg", callback.evaluations)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default=MaskedPPOConfig.env_id)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--total-timesteps", type=int, default=250_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=64)
    parser.add_argument("--final-eval-episodes", type=int, default=256)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = MaskedPPOConfig(
        env_id=args.env_id,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        eval_interval=args.eval_interval,
        checkpoint_eval_episodes_per_skill=args.checkpoint_eval_episodes,
        final_eval_episodes_per_skill=args.final_eval_episodes,
    )
    size_tag = "8" if "8x8" in config.env_id else "5"
    run_id = (
        f"doorkey{size_tag}_masked_ppo_control_seed{config.seed}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = args.output_dir or Path(
        "outputs/skill_discovery/minigrid_doorkey_training"
    ) / run_id
    output = train_run(config, output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "elapsed_seconds": output["elapsed_seconds"],
                "final_target_stage_rates": output["final_evaluation"][
                    "target_stage_rates"
                ],
                "final_state_rates": output["final_evaluation"][
                    "final_state_rates"
                ],
                "goal_native_success_rate": output["final_evaluation"][
                    "goal_native_success_rate"
                ],
                "checkpoint_stability_passed": output[
                    "checkpoint_stability_passed"
                ],
                "signal_gate_passed": output["signal_gate_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
