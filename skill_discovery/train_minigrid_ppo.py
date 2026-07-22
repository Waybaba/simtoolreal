"""Train skill-conditioned PPO policies on official MiniGrid DoorKey."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from skill_discovery.minigrid_doorkey import DOORKEY_STAGES
from skill_discovery.minigrid_skill_env import (
    DoorKeySkillWrapper,
    MiniGridSkillConfig,
    OnlineIntrinsicReward,
)
from skill_discovery.train_pusher_diayn import _best_class_assignment


@dataclass(frozen=True)
class MiniGridPPOConfig:
    objective: str = "semantic_spread"
    env_id: str = "MiniGrid-DoorKey-5x5-v0"
    seed: int = 7
    num_skills: int = 4
    n_envs: int = 8
    total_timesteps: int = 250_000
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
    stochastic_final_eval_episodes_per_skill: int = 64
    class_rate_gate: float = 0.70
    torch_threads: int = 4

    def __post_init__(self) -> None:
        if self.objective not in {
            "random",
            "raw",
            "semantic",
            "semantic_spread",
            "semantic_balanced",
            "semantic_balanced_transition",
        }:
            raise ValueError("unknown objective")
        if self.num_skills != len(DOORKEY_STAGES):
            raise ValueError("DoorKey probe is fixed to four skills")
        if self.n_envs < self.num_skills or self.n_envs % self.num_skills != 0:
            raise ValueError("n_envs must be a positive multiple of num_skills")
        if self.batch_size > self.n_steps * self.n_envs:
            raise ValueError("batch size cannot exceed rollout size")
        if self.stochastic_final_eval_episodes_per_skill < 0:
            raise ValueError("stochastic evaluation episode count cannot be negative")


def _skill_config(config: MiniGridPPOConfig) -> MiniGridSkillConfig:
    return MiniGridSkillConfig(
        env_id=config.env_id,
        num_skills=config.num_skills,
        seed=config.seed,
    )


def _make_vec_env(
    config: MiniGridPPOConfig,
    reward_model: OnlineIntrinsicReward,
) -> DummyVecEnv:
    skill_config = _skill_config(config)
    env_fns = []
    for env_index in range(config.n_envs):
        skill_id = env_index % config.num_skills

        def make_env(skill: int = skill_id) -> Monitor:
            return Monitor(DoorKeySkillWrapper(skill_config, skill, reward_model))

        env_fns.append(make_env)
    vec_env = DummyVecEnv(env_fns)
    vec_env.seed(config.seed)
    return vec_env


def evaluate_model(
    model: PPO,
    config: MiniGridPPOConfig,
    episodes_per_skill: int,
    seed: int,
    *,
    deterministic: bool = True,
) -> dict[str, object]:
    rates = np.zeros((config.num_skills, len(DOORKEY_STAGES)), dtype=np.float64)
    mean_steps = []
    native_success_rates = []
    action_rates = []
    skill_config = _skill_config(config)
    for skill in range(config.num_skills):
        reward_model = OnlineIntrinsicReward("random", skill_config)
        env = DoorKeySkillWrapper(skill_config, skill, reward_model)
        stage_counts = np.zeros(len(DOORKEY_STAGES), dtype=np.int64)
        episode_steps = []
        successes = []
        action_counts = np.zeros(env.action_space.n, dtype=np.int64)
        try:
            for episode in range(episodes_per_skill):
                observation, _ = env.reset(seed=seed + skill * 1_000_000 + episode)
                terminated = truncated = False
                steps = 0
                final_info: dict[str, object] = {}
                while not (terminated or truncated):
                    action, _ = model.predict(
                        observation, deterministic=deterministic
                    )
                    action_id = int(np.asarray(action).item())
                    observation, _, terminated, truncated, final_info = env.step(action_id)
                    action_counts[action_id] += 1
                    steps += 1
                stage = int(final_info["furthest_stage"])
                stage_counts[stage] += 1
                episode_steps.append(steps)
                successes.append(float(final_info["native_reward"]) > 0)
        finally:
            env.close()
        rates[skill] = stage_counts / episodes_per_skill
        mean_steps.append(float(np.mean(episode_steps)))
        native_success_rates.append(float(np.mean(successes)))
        action_rates.append((action_counts / max(action_counts.sum(), 1)).tolist())

    assignment, matched_rates = _best_class_assignment(rates)
    by_stage = np.zeros(len(DOORKEY_STAGES), dtype=np.float64)
    for skill, stage in enumerate(assignment):
        by_stage[stage] = matched_rates[skill]
    goal_skill = assignment.index(3)
    return {
        "deterministic": deterministic,
        "episodes_per_skill": episodes_per_skill,
        "stage_rates": rates.tolist(),
        "stage_assignment": assignment,
        "stage_assignment_names": [DOORKEY_STAGES[index] for index in assignment],
        "matched_stage_rates": matched_rates,
        "matched_rate_by_stage": {
            name: float(by_stage[index]) for index, name in enumerate(DOORKEY_STAGES)
        },
        "matched_stage_rate_mean": float(np.mean(matched_rates)),
        "matched_stage_rate_min": float(np.min(matched_rates)),
        "goal_skill": goal_skill,
        "goal_skill_native_success_rate": native_success_rates[goal_skill],
        "native_success_rates": native_success_rates,
        "mean_episode_steps": mean_steps,
        "action_rates": action_rates,
        "specialization_gate_passed": bool(
            min(matched_rates) >= config.class_rate_gate
            and native_success_rates[goal_skill] >= config.class_rate_gate
        ),
    }


class EvaluationCallback(BaseCallback):
    def __init__(self, config: MiniGridPPOConfig):
        super().__init__(verbose=0)
        self.config = config
        self.next_evaluation = config.eval_interval
        self.evaluations: list[dict[str, object]] = []
        self.terminal_episode_counts = np.zeros(config.num_skills, dtype=np.int64)
        self.terminal_stage_counts = np.zeros(
            (config.num_skills, len(DOORKEY_STAGES)), dtype=np.int64
        )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if bool(done):
                skill = int(info["skill_id"])
                stage = int(info["furthest_stage"])
                self.terminal_episode_counts[skill] += 1
                self.terminal_stage_counts[skill, stage] += 1
        if self.num_timesteps >= self.next_evaluation:
            evaluation = evaluate_model(
                self.model,
                self.config,
                self.config.checkpoint_eval_episodes_per_skill,
                seed=self.config.seed + 100_000 + self.next_evaluation,
            )
            self.evaluations.append(
                {"timesteps": self.num_timesteps, **evaluation}
            )
            self.next_evaluation += self.config.eval_interval
        return True

    def state_dict(self) -> dict[str, object]:
        return {
            "evaluations": self.evaluations,
            "terminal_episode_counts": self.terminal_episode_counts.tolist(),
            "terminal_stage_counts": self.terminal_stage_counts.tolist(),
        }


def _write_curves(path: Path, evaluations: list[dict[str, object]]) -> None:
    width, height = 920, 440
    left, top, chart_width, chart_height = 65, 60, 790, 300
    colors = ("#68757d", "#2875a4", "#d17031", "#2b895f")
    lines = []
    for stage in range(len(DOORKEY_STAGES)):
        values = [row["matched_rate_by_stage"][DOORKEY_STAGES[stage]] for row in evaluations]
        points = []
        for index, value in enumerate(values):
            x = left + chart_width * index / max(len(values) - 1, 1)
            y = top + chart_height * (1.0 - value)
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[stage]}" stroke-width="2"/>'
        )
    legend = []
    for index, name in enumerate(DOORKEY_STAGES):
        x = 500 + index * 100
        legend.append(
            f'<rect x="{x}" y="20" width="12" height="4" fill="{colors[index]}"/>'
            f'<text x="{x + 17}" y="27" font-family="sans-serif" font-size="11">{name.split("_")[0]}</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fbfcfd"/>
<text x="30" y="32" font-family="sans-serif" font-size="20" fill="#172b3a">MiniGrid DoorKey: deterministic matched stage rates</text>
<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#83919a"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#83919a"/>
<line x1="{left}" y1="{top + 0.3 * chart_height}" x2="{left + chart_width}" y2="{top + 0.3 * chart_height}" stroke="#b8c2c8" stroke-dasharray="4 4"/>
<text x="22" y="{top + 5}" font-family="sans-serif" font-size="12">1.0</text>
<text x="22" y="{top + chart_height + 5}" font-family="sans-serif" font-size="12">0.0</text>
{''.join(lines)}
{''.join(legend)}
</svg>'''
    path.write_text(svg, encoding="utf-8")


def train_run(config: MiniGridPPOConfig, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    reward_model = OnlineIntrinsicReward(config.objective, _skill_config(config))
    vec_env = _make_vec_env(config, reward_model)
    callback = EvaluationCallback(config)
    model = PPO(
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
        model.learn(total_timesteps=config.total_timesteps, callback=callback)
        elapsed = time.monotonic() - started
        model.save(output_dir / "ppo_policy")
        final_evaluation = evaluate_model(
            model,
            config,
            config.final_eval_episodes_per_skill,
            seed=config.seed + 900_000,
        )
        final_stochastic_evaluation = None
        if config.stochastic_final_eval_episodes_per_skill:
            final_stochastic_evaluation = evaluate_model(
                model,
                config,
                config.stochastic_final_eval_episodes_per_skill,
                seed=config.seed + 950_000,
                deterministic=False,
            )
    finally:
        vec_env.close()

    recent = callback.evaluations[-min(3, len(callback.evaluations)) :]
    checkpoint_stability = bool(
        len(recent) >= 2
        and all(row["specialization_gate_passed"] for row in recent)
    )
    output = {
        "config": asdict(config),
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("minigrid", "gymnasium", "torch", "stable-baselines3")
        },
        "elapsed_seconds": elapsed,
        "reward_model": reward_model.state_dict(),
        "training_callback": callback.state_dict(),
        "final_evaluation": final_evaluation,
        "final_stochastic_evaluation": final_stochastic_evaluation,
        "checkpoint_stability_passed": checkpoint_stability,
        "signal_gate_passed": bool(
            final_evaluation["specialization_gate_passed"] and checkpoint_stability
        ),
    }
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    (output_dir / "evaluation_history.json").write_text(
        json.dumps(callback.evaluations, indent=2), encoding="utf-8"
    )
    if callback.evaluations:
        _write_curves(output_dir / "stage_curves.svg", callback.evaluations)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--objective",
        choices=(
            "random",
            "raw",
            "semantic",
            "semantic_spread",
            "semantic_balanced",
            "semantic_balanced_transition",
        ),
        default="semantic_spread",
    )
    parser.add_argument("--env-id", default="MiniGrid-DoorKey-5x5-v0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--total-timesteps", type=int, default=250_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=64)
    parser.add_argument("--final-eval-episodes", type=int, default=256)
    parser.add_argument("--stochastic-final-eval-episodes", type=int, default=64)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--net-arch", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = MiniGridPPOConfig(
        objective=args.objective,
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
        stochastic_final_eval_episodes_per_skill=args.stochastic_final_eval_episodes,
        ent_coef=args.ent_coef,
        net_arch=tuple(args.net_arch),
    )
    run_id = (
        f"doorkey5_ppo_{config.objective}_seed{config.seed}_"
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
                "final_evaluation": output["final_evaluation"],
                "final_stochastic_evaluation": output[
                    "final_stochastic_evaluation"
                ],
                "checkpoint_stability_passed": output["checkpoint_stability_passed"],
                "signal_gate_passed": output["signal_gate_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
