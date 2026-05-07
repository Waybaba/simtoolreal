#!/usr/bin/env python3
"""CleanRL-style SimToolReal trainer for the Isaac Lab direct environment.

This script ports the previous Isaac Gym/rl_games training recipe into an
explicit single-file PyTorch loop:

* PPO with 16-step horizons, clipped value loss, adaptive KL learning rate.
* LSTM actor with a learned SAPG coefficient embedding before the MLP.
* Asymmetric critic that consumes Isaac Lab ``obs["critic"]`` states.
* ``mixed_expl_learn_param`` entropy blocks, coefficient-conditioned sigma,
  and leader-follower off-policy batch augmentation.

The goal is algorithmic parity without depending on rl_games.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ISAAC_SRC = REPO_ROOT / "src" / "isaaclab_env"
if str(LOCAL_ISAAC_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_ISAAC_SRC))

from isaaclab.app import AppLauncher


class CleanRLAppLauncher(AppLauncher):
    """AppLauncher variant that can decouple Kit renderer and PhysX CUDA ordinals."""

    def __init__(
        self,
        launcher_args: argparse.Namespace,
        *,
        active_gpu_override: Optional[int] = None,
        physics_gpu_override: Optional[int] = None,
    ):
        self._active_gpu_override = active_gpu_override
        self._physics_gpu_override = physics_gpu_override
        super().__init__(launcher_args)

    def _resolve_device_settings(self, launcher_args: dict) -> None:
        super()._resolve_device_settings(launcher_args)
        if self._active_gpu_override is not None:
            launcher_args["active_gpu"] = self._active_gpu_override
        if self._physics_gpu_override is not None:
            launcher_args["physics_gpu"] = self._physics_gpu_override


parser = argparse.ArgumentParser(description="Train SimToolReal-Direct with a CleanRL-style PPO/SAPG implementation.")

# Environment / run control.
parser.add_argument("--task", type=str, default="SimToolReal-Direct-v0")
parser.add_argument("--num_envs", type=int, default=12288)
parser.add_argument("--num_steps", type=int, default=16, help="PPO horizon length.")
parser.add_argument("--total_updates", type=int, default=1_000_000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "outputs" / "isaaclab_cleanrl_train")
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--disable_fabric", action="store_true")
parser.add_argument("--env_device", type=str, default=None)
parser.add_argument("--kit_active_gpu", type=int, default=None)
parser.add_argument("--kit_physics_gpu", type=int, default=None)

# PPO config matched to SimToolRealLSTMAsymmetricPPO.yaml / launch_training.py.
parser.add_argument("--learning_rate", type=float, default=1.0e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--clip_coef", type=float, default=0.1)
parser.add_argument("--update_epochs", type=int, default=2)
parser.add_argument("--minibatch_size", type=int, default=98304)
parser.add_argument("--seq_length", type=int, default=16)
parser.add_argument("--norm_adv", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--norm_obs", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--norm_value", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--clip_value", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--reward_scale", type=float, default=0.01)
parser.add_argument("--critic_coef", type=float, default=4.0)
parser.add_argument("--bounds_loss_coef", type=float, default=1.0e-4)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--kl_threshold", type=float, default=0.016)
parser.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction, default=True)

# Network config matched to old LSTM actor / asymmetric critic.
parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[1024, 1024, 512, 512])
parser.add_argument("--lstm_hidden_size", type=int, default=1024)
parser.add_argument("--lstm_layers", type=int, default=1)
parser.add_argument("--extra_param_size", type=int, default=32)

# SAPG / mixed exploration config from launch_training.py.
parser.add_argument("--expl_type", choices=("none", "mixed_expl_learn_param"), default="mixed_expl_learn_param")
parser.add_argument("--sapg_num_blocks", type=int, default=6)
parser.add_argument("--expl_coef_block_size", type=int, default=None)
parser.add_argument("--expl_reward_type", choices=("entropy", "none"), default="entropy")
parser.add_argument("--expl_reward_coef_scale", type=float, default=0.005)
parser.add_argument("--use_others_experience", choices=("none", "lf", "all"), default="lf")
parser.add_argument("--off_policy_ratio", type=float, default=1.0)
parser.add_argument("--fixed_sigma", choices=("fixed", "coef_cond"), default="coef_cond")

# Checkpoints, W&B, and video.
parser.add_argument("--save_frequency", type=int, default=3000)
parser.add_argument("--save_best_after", type=int, default=100)
parser.add_argument("--checkpoint", type=Path, default=None)
parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--wandb_project", type=str, default="simtoolreal")
parser.add_argument("--wandb_entity", type=str, default="waybabag")
parser.add_argument("--wandb_group", type=str, default=datetime.now().strftime("%Y-%m-%d"))
parser.add_argument("--wandb_tags", type=str, nargs="*", default=[])
parser.add_argument("--wandb_notes", type=str, default="")
parser.add_argument("--wandb_logcode_dir", type=str, default=".")
parser.add_argument("--wandb_log_diff", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--capture_video", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--capture_video_freq", type=int, default=6000, help="Capture cadence in env control steps, matching the old task.env.capture_video_freq.")
parser.add_argument("--capture_video_len", type=int, default=600, help="Number of rendered frames per video, matching the old task.env.capture_video_len.")
parser.add_argument("--capture_video_env_id", type=int, default=0, help="Env id viewed by the training video camera.")
parser.add_argument("--capture_video_start_on_reset", action=argparse.BooleanOptionalAction, default=True, help="Match old behavior: arm capture at the cadence, then start recording when the viewed env resets.")
parser.add_argument("--video_fps", type=int, default=0, help="0 matches the old env video fps: int(1 / control_dt).")

# Common task overrides used by the old launch helper.
parser.add_argument("--object_scale_noise_min", type=float, default=0.9)
parser.add_argument("--object_scale_noise_max", type=float, default=1.1)
parser.add_argument("--force_consecutive_near_goal_steps", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--force_scale", type=float, default=20.0)
parser.add_argument("--torque_scale", type=float, default=2.0)
parser.add_argument("--object_ang_vel_penalty_scale", type=float, default=0.0)

CleanRLAppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, enable_cameras=False)
args_cli = parser.parse_args()
if args_cli.capture_video:
    args_cli.enable_cameras = True

app_launcher = CleanRLAppLauncher(
    args_cli,
    active_gpu_override=args_cli.kit_active_gpu,
    physics_gpu_override=args_cli.kit_physics_gpu,
)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.distributions.normal import Normal  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import isaaclab_env.tasks  # noqa: F401, E402


@dataclass
class TrainStats:
    update: int
    global_step: int
    fps: float
    reward_mean: float
    episodic_success_mean: float
    actor_loss: float
    critic_loss: float
    entropy: float
    approx_kl: float
    clipfrac: float
    bounds_loss: float
    learning_rate: float
    sigma_mean: float
    video_path: str | None = None


class RunningMeanStd(nn.Module):
    def __init__(self, shape: tuple[int, ...], epsilon: float = 1.0e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float32))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float()
        if x.numel() == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(x.shape[0]), dtype=torch.float32, device=x.device)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta.square() * self.count * batch_count / total_count
        self.mean.copy_(new_mean)
        self.var.copy_(torch.clamp(m_2 / total_count, min=1.0e-6))
        self.count.copy_(total_count)

    def normalize(self, x: torch.Tensor, clip: float = 5.0) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1.0e-8), -clip, clip)


class ValueNormalizer(nn.Module):
    def __init__(self, epsilon: float = 1.0e-4):
        super().__init__()
        self.rms = RunningMeanStd((1,), epsilon=epsilon)

    @torch.no_grad()
    def update(self, returns: torch.Tensor) -> None:
        self.rms.update(returns.reshape(-1, 1))

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.rms.mean) / torch.sqrt(self.rms.var + 1.0e-8)

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.sqrt(self.rms.var + 1.0e-8) + self.rms.mean


def _init_linear(layer: nn.Linear, std: float = math.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _make_mlp(input_dim: int, hidden_sizes: list[int], output_dim: int | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_size in hidden_sizes:
        layers.append(_init_linear(nn.Linear(last_dim, hidden_size)))
        layers.append(nn.ELU())
        last_dim = hidden_size
    if output_dim is not None:
        layers.append(_init_linear(nn.Linear(last_dim, output_dim), std=1.0))
    return nn.Sequential(*layers)


class ExtraParamInput(nn.Module):
    """Replace an appended SAPG coefficient id with a learned parameter vector."""

    def __init__(self, base_dim: int, coef_ids: torch.Tensor | None, extra_param_size: int):
        super().__init__()
        self.base_dim = base_dim
        if coef_ids is None:
            self.register_buffer("coef_ids", torch.empty(0))
            self.extra_params = None
            self.output_dim = base_dim
        else:
            self.register_buffer("coef_ids", coef_ids.float().clone())
            self.extra_params = nn.Parameter(torch.randn((len(coef_ids), extra_param_size), dtype=torch.float32))
            self.output_dim = base_dim + extra_param_size

    def forward(self, obs_with_optional_coef: torch.Tensor) -> torch.Tensor:
        if self.extra_params is None:
            return obs_with_optional_coef
        coef = obs_with_optional_coef[:, self.base_dim].reshape(-1, 1)
        block_ids = (coef == self.coef_ids.reshape(1, -1)).float().argmax(dim=1)
        return torch.cat([obs_with_optional_coef[:, : self.base_dim], self.extra_params[block_ids]], dim=-1)


class LSTMActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int],
        lstm_hidden_size: int,
        lstm_layers: int,
        coef_ids: torch.Tensor | None,
        extra_param_size: int,
        fixed_sigma: str,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.fixed_sigma = fixed_sigma
        self.input_adapter = ExtraParamInput(obs_dim, coef_ids, extra_param_size)
        self.lstm = nn.LSTM(self.input_adapter.output_dim, lstm_hidden_size, num_layers=lstm_layers)
        self.layer_norm = nn.LayerNorm(lstm_hidden_size)
        self.mlp = _make_mlp(lstm_hidden_size, hidden_sizes)
        self.mu = _init_linear(nn.Linear(hidden_sizes[-1], action_dim), std=0.01)
        if fixed_sigma == "coef_cond":
            if coef_ids is None:
                raise ValueError("coef_cond sigma requires SAPG coefficient ids")
            self.register_buffer("sigma_ids", coef_ids.float().clone())
            self.log_sigma = nn.Parameter(torch.zeros((len(coef_ids), action_dim), dtype=torch.float32))
        else:
            self.register_buffer("sigma_ids", torch.empty(0))
            self.log_sigma = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

    def initial_state(self, num_envs: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros((self.lstm.num_layers, num_envs, self.lstm.hidden_size), dtype=torch.float32, device=device)
        c = torch.zeros_like(h)
        return h, c

    def _sigma(self, obs_with_optional_coef: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        if self.fixed_sigma == "coef_cond":
            coef = obs_with_optional_coef[:, -1].reshape(-1, 1)
            block_ids = (coef == self.sigma_ids.reshape(1, -1)).float().argmax(dim=1)
            return torch.exp(self.log_sigma[block_ids])
        return torch.exp(self.log_sigma).expand_as(mu)

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        initial_state: tuple[torch.Tensor, torch.Tensor],
        dones_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward B sequences of length T.

        Args:
            obs_seq: Tensor of shape ``[B, T, obs_dim]``.
            initial_state: LSTM state with batch dimension B.
            dones_seq: Optional tensor ``[B, T]``. If set, hidden state is zeroed
                before consuming observations whose env was done.
        """
        batch_size, seq_len, obs_dim = obs_seq.shape
        flat_obs = obs_seq.reshape(batch_size * seq_len, obs_dim)
        adapted = self.input_adapter(flat_obs).reshape(batch_size, seq_len, -1).transpose(0, 1)
        state = initial_state
        outputs = []
        if dones_seq is None:
            out, state = self.lstm(adapted, state)
            outputs = [out]
        else:
            dones_t = dones_seq.transpose(0, 1)
            h, c = state
            for t in range(seq_len):
                not_done = (1.0 - dones_t[t].float()).reshape(1, batch_size, 1)
                h = h * not_done
                c = c * not_done
                out, (h, c) = self.lstm(adapted[t : t + 1], (h, c))
                outputs.append(out)
            state = (h, c)
        lstm_out = torch.cat(outputs, dim=0).transpose(0, 1).reshape(batch_size * seq_len, -1)
        features = self.mlp(self.layer_norm(lstm_out))
        mu = self.mu(features)
        sigma = self._sigma(flat_obs, mu)
        return mu.reshape(batch_size, seq_len, -1), sigma.reshape(batch_size, seq_len, -1), state

    def forward_step(
        self,
        obs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        done: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        mu, sigma, next_state = self.forward_sequence(obs.unsqueeze(1), state, done.reshape(-1, 1))
        return mu[:, 0], sigma[:, 0], next_state


class AsymmetricCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_sizes: list[int], coef_ids: torch.Tensor | None, extra_param_size: int):
        super().__init__()
        self.input_adapter = ExtraParamInput(state_dim, coef_ids, extra_param_size)
        self.net = _make_mlp(self.input_adapter.output_dim, hidden_sizes, output_dim=1)

    def forward(self, state_obs: torch.Tensor) -> torch.Tensor:
        return self.net(self.input_adapter(state_obs)).squeeze(-1)


class SimToolRealAgent(nn.Module):
    def __init__(
        self,
        policy_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int],
        lstm_hidden_size: int,
        lstm_layers: int,
        coef_ids: torch.Tensor | None,
        extra_param_size: int,
        fixed_sigma: str,
        value_normalizer: ValueNormalizer | None,
    ):
        super().__init__()
        self.actor = LSTMActor(
            policy_obs_dim,
            action_dim,
            hidden_sizes,
            lstm_hidden_size,
            lstm_layers,
            coef_ids,
            extra_param_size,
            fixed_sigma,
        )
        self.critic = AsymmetricCritic(critic_obs_dim, hidden_sizes, coef_ids, extra_param_size)
        self.value_normalizer = value_normalizer

    def get_value(self, critic_obs: torch.Tensor, *, denormalize: bool = True) -> torch.Tensor:
        value = self.critic(critic_obs)
        if denormalize and self.value_normalizer is not None:
            value = self.value_normalizer.denormalize(value)
        return value

    def get_action_and_value(
        self,
        policy_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actor_state: tuple[torch.Tensor, torch.Tensor],
        done: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        mu, sigma, next_state = self.actor.forward_step(policy_obs, actor_state, done)
        dist = Normal(mu, sigma)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.get_value(critic_obs, denormalize=True)
        return action, logprob, entropy, value, mu, sigma, next_state


def _policy_kl(old_mu: torch.Tensor, old_sigma: torch.Tensor, new_mu: torch.Tensor, new_sigma: torch.Tensor) -> torch.Tensor:
    kl = torch.log(new_sigma / old_sigma + 1.0e-5)
    kl += (old_sigma.square() + (old_mu - new_mu).square()) / (2.0 * new_sigma.square() + 1.0e-5) - 0.5
    return kl.sum(dim=-1)


def _bound_loss(mu: torch.Tensor) -> torch.Tensor:
    soft_bound = 1.1
    high = torch.clamp_min(mu - soft_bound, 0.0).square()
    low = torch.clamp_max(mu + soft_bound, 0.0).square()
    return (high + low).sum(dim=-1)


def _make_sapg(args: argparse.Namespace, num_envs: int, device: torch.device) -> dict[str, Any]:
    if args.expl_type == "none":
        return {
            "enabled": False,
            "num_blocks": 1,
            "block_size": num_envs,
            "coef_ids": None,
            "env_coef": None,
            "env_entropy_coef": torch.full((num_envs,), 0.0, device=device),
        }

    block_size = args.expl_coef_block_size
    if block_size is None:
        block_size = num_envs // args.sapg_num_blocks if num_envs % args.sapg_num_blocks == 0 else num_envs
    if num_envs % block_size != 0:
        raise ValueError(f"num_envs={num_envs} must be divisible by expl_coef_block_size={block_size}")
    num_blocks = num_envs // block_size
    coef_ids = torch.linspace(50.0, 0.0, num_blocks, device=device)
    block_ids = torch.arange(num_blocks, device=device).repeat_interleave(block_size)
    env_coef = coef_ids[block_ids].unsqueeze(-1)
    if args.expl_reward_type == "entropy":
        entropy_candidates = torch.linspace(0.5, 0.0, num_blocks, device=device) * args.expl_reward_coef_scale
    else:
        entropy_candidates = torch.zeros((num_blocks,), device=device)
    env_entropy_coef = entropy_candidates[block_ids]
    return {
        "enabled": True,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "coef_ids": coef_ids,
        "env_coef": env_coef,
        "env_entropy_coef": env_entropy_coef,
        "entropy_candidates": entropy_candidates,
    }


def _append_coef(obs: torch.Tensor, env_coef: torch.Tensor | None) -> torch.Tensor:
    if env_coef is None:
        return obs
    return torch.cat([obs, env_coef], dim=-1)


def _append_coef_sequence(obs: torch.Tensor, env_coef: torch.Tensor | None) -> torch.Tensor:
    if env_coef is None:
        return obs
    return torch.cat([obs, env_coef.unsqueeze(0).expand(obs.shape[0], -1, -1)], dim=-1)


def _normalize_base_and_append(
    obs: torch.Tensor,
    rms: RunningMeanStd | None,
    env_coef: torch.Tensor | None,
) -> torch.Tensor:
    if rms is not None:
        obs = rms.normalize(obs)
    return _append_coef(obs, env_coef)


def _normalize_base_sequence_and_append(
    obs: torch.Tensor,
    rms: RunningMeanStd | None,
    env_coef: torch.Tensor | None,
) -> torch.Tensor:
    if rms is not None:
        flat = obs.reshape(-1, obs.shape[-1])
        obs = rms.normalize(flat).reshape_as(obs)
    return _append_coef_sequence(obs, env_coef)


def _compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_gae_lam = torch.zeros_like(next_value)
    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            next_non_terminal = 1.0 - next_done.float()
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[t + 1].float()
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages[t] = last_gae_lam
    returns = advantages + values
    return advantages, returns


def _make_sequence_batch(
    *,
    args: argparse.Namespace,
    agent: SimToolRealAgent,
    sapg: dict[str, Any],
    policy_obs: torch.Tensor,
    critic_obs: torch.Tensor,
    actions: torch.Tensor,
    old_logprobs: torch.Tensor,
    old_values: torch.Tensor,
    old_mu: torch.Tensor,
    old_sigma: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    initial_actor_state: tuple[torch.Tensor, torch.Tensor],
    actor_dones: torch.Tensor,
    obs_rms: RunningMeanStd | None,
    state_rms: RunningMeanStd | None,
    next_critic_obs: torch.Tensor,
    next_done: torch.Tensor,
) -> dict[str, Any]:
    """Create sequence-major PPO data and optional LF off-policy augmentation."""
    # Convert from [T, N, ...] to [N, T, ...].
    base = {
        "actions": actions.transpose(0, 1).contiguous(),
        "old_logprobs": old_logprobs.transpose(0, 1).contiguous(),
        "old_values": old_values.transpose(0, 1).contiguous(),
        "old_mu": old_mu.transpose(0, 1).contiguous(),
        "old_sigma": old_sigma.transpose(0, 1).contiguous(),
        "advantages": advantages.transpose(0, 1).contiguous(),
        "returns": returns.transpose(0, 1).contiguous(),
        "dones": actor_dones.transpose(0, 1).contiguous(),
        "off_policy": torch.zeros((policy_obs.shape[1], policy_obs.shape[0]), dtype=torch.bool, device=policy_obs.device),
        "h0": initial_actor_state[0].transpose(0, 1).contiguous(),
        "c0": initial_actor_state[1].transpose(0, 1).contiguous(),
        "entropy_coef": sapg["env_entropy_coef"].reshape(-1, 1).expand(-1, policy_obs.shape[0]).contiguous(),
    }

    env_coef = sapg["env_coef"]
    policy_sequences = [_normalize_base_sequence_and_append(policy_obs, obs_rms, env_coef).transpose(0, 1).contiguous()]
    critic_sequences = [_normalize_base_sequence_and_append(critic_obs, state_rms, env_coef).transpose(0, 1).contiguous()]
    pieces = {key: [value] for key, value in base.items()}

    if sapg["enabled"] and args.use_others_experience != "none" and sapg["num_blocks"] > 1:
        num_repeat = min(sapg["num_blocks"], int(args.off_policy_ratio) + 1)
        repeat_idxs = [0]
        if num_repeat > 1:
            repeat_idxs.extend(random.sample(range(1, sapg["num_blocks"]), k=num_repeat - 1))

        block_size = sapg["block_size"]
        for repeat_idx in repeat_idxs[1:]:
            rolled_coef = torch.roll(env_coef, shifts=block_size * repeat_idx, dims=0)
            rolled_entropy = torch.roll(sapg["env_entropy_coef"], shifts=block_size * repeat_idx, dims=0)
            if args.use_others_experience == "lf":
                env_ids = torch.arange((repeat_idx - 1) * block_size, repeat_idx * block_size, device=policy_obs.device)
            else:
                env_ids = torch.arange(policy_obs.shape[1], device=policy_obs.device)

            alt_policy = _normalize_base_sequence_and_append(policy_obs[:, env_ids], obs_rms, rolled_coef[env_ids])
            alt_critic = _normalize_base_sequence_and_append(critic_obs[:, env_ids], state_rms, rolled_coef[env_ids])
            policy_sequences.append(alt_policy.transpose(0, 1).contiguous())
            critic_sequences.append(alt_critic.transpose(0, 1).contiguous())

            for key in ("actions", "old_logprobs", "old_mu", "old_sigma", "dones"):
                pieces[key].append(base[key][env_ids])
            pieces["off_policy"].append(torch.ones_like(base["off_policy"][env_ids]))
            pieces["h0"].append(base["h0"][env_ids])
            pieces["c0"].append(base["c0"][env_ids])
            pieces["entropy_coef"].append(rolled_entropy[env_ids].reshape(-1, 1).expand(-1, policy_obs.shape[0]).contiguous())

            # Match the rl_games LF augmentation path: recompute values/one-step
            # returns under the alternate coefficient, while keeping old policy
            # log-probs/mu/sigma from the behavior policy.
            with torch.no_grad():
                alt_critic_flat = alt_critic.reshape(-1, alt_critic.shape[-1])
                alt_values = agent.get_value(alt_critic_flat, denormalize=True).reshape(policy_obs.shape[0], -1)
                alt_next_critic = _normalize_base_and_append(next_critic_obs[env_ids], state_rms, rolled_coef[env_ids])
                alt_next_value = agent.get_value(alt_next_critic, denormalize=True)
                value_plus_next = torch.cat([alt_values, alt_next_value.unsqueeze(0)], dim=0)
                done_plus_next = torch.cat([dones[:, env_ids], next_done[env_ids].unsqueeze(0)], dim=0)
                one_step_returns = rewards[:, env_ids] + args.gamma * value_plus_next[1:] * (1.0 - done_plus_next[1:].float())
                off_adv = one_step_returns - alt_values
            pieces["old_values"].append(alt_values.transpose(0, 1).contiguous())
            pieces["returns"].append(one_step_returns.transpose(0, 1).contiguous())
            pieces["advantages"].append(off_adv.transpose(0, 1).contiguous())

    batch = {
        "policy_obs": torch.cat(policy_sequences, dim=0),
        "critic_obs": torch.cat(critic_sequences, dim=0),
    }
    for key, values in pieces.items():
        batch[key] = torch.cat(values, dim=0)
    return batch


def _set_lr(optimizer: optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _update_adaptive_lr(current_lr: float, kl_threshold: float, approx_kl: float) -> float:
    if approx_kl > 2.0 * kl_threshold:
        return max(current_lr / 1.5, 1.0e-6)
    if approx_kl < 0.5 * kl_threshold:
        return min(current_lr * 1.5, 1.0e-2)
    return current_lr


def _set_camera(env) -> None:
    origins = env.unwrapped.scene.env_origins[: env.unwrapped.num_envs].detach().cpu()
    center = origins.mean(dim=0)
    spread = torch.linalg.norm(origins.max(dim=0).values - origins.min(dim=0).values).item()
    distance = max(2.8, spread * 1.4)
    eye = center + torch.tensor([distance, -distance, 1.65])
    target = center + torch.tensor([0.0, 0.38, 0.52])
    env.unwrapped.sim.set_camera_view(
        eye=eye.tolist(),
        target=target.tolist(),
        camera_prim_path=env.unwrapped.cfg.viewer.cam_prim_path,
    )


def _write_video_frame(writer, env, post_step_renders: int = 2) -> None:
    for _ in range(post_step_renders):
        env.unwrapped.sim.render()
    frame = np.asarray(env.render())
    if frame.size == 0:
        raise RuntimeError("env.render() returned an empty frame.")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    writer.append_data(frame)


def _video_fps(env, args: argparse.Namespace) -> int:
    if args.video_fps > 0:
        return args.video_fps
    return int(round(1.0 / env.unwrapped.control_dt))


def _flatten_scalar_dict(values: dict[str, Any], prefix: str = "") -> dict[str, float | int]:
    flat: dict[str, float | int] = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_scalar_dict(value, name))
        elif isinstance(value, torch.Tensor):
            if value.numel() == 1:
                flat[name] = float(value.detach().cpu().item())
        elif isinstance(value, (float, int)):
            flat[name] = value
    return flat


def _try_init_wandb(args: argparse.Namespace, run_dir: Path, config: dict[str, Any]):
    if not args.wandb:
        return None
    try:
        import wandb
    except Exception as exc:
        print(f"[CLEANRL][WARN] wandb requested but unavailable: {exc}", flush=True)
        return None
    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            id=f"uid_{run_dir.name}",
            name=run_dir.name,
            resume="allow",
            tags=args.wandb_tags,
            notes=args.wandb_notes,
            config=config,
            dir=str(run_dir),
            sync_tensorboard=False,
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.define_metric("*", step_metric="global_step")
        code_root = Path(args.wandb_logcode_dir)
        if str(code_root):
            run.log_code(root=str(code_root))
        if args.wandb_log_diff:
            diff_path = run_dir / "diff.patch"
            import subprocess

            diff = subprocess.run(["git", "diff"], cwd=REPO_ROOT, check=False, capture_output=True, text=True)
            diff_path.write_text(diff.stdout, encoding="utf-8")
            artifact = wandb.Artifact("diff", type="file", description="Git diff")
            artifact.add_file(str(diff_path))
            run.log_artifact(artifact)
        print(f"[CLEANRL] wandb_url={run.url}", flush=True)
        return run
    except Exception as exc:
        print(f"[CLEANRL][WARN] Could not initialize W&B: {exc}", flush=True)
        return None


def _save_checkpoint(
    path: Path,
    agent: SimToolRealAgent,
    optimizer: optim.Optimizer,
    obs_rms: RunningMeanStd | None,
    state_rms: RunningMeanStd | None,
    value_normalizer: ValueNormalizer | None,
    update: int,
    global_step: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "agent": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "obs_rms": None if obs_rms is None else obs_rms.state_dict(),
            "state_rms": None if state_rms is None else state_rms.state_dict(),
            "value_normalizer": None if value_normalizer is None else value_normalizer.state_dict(),
            "update": update,
            "global_step": global_step,
            "args": vars(args),
        },
        path,
    )


def _load_checkpoint(
    path: Path,
    agent: SimToolRealAgent,
    optimizer: optim.Optimizer,
    obs_rms: RunningMeanStd | None,
    state_rms: RunningMeanStd | None,
    value_normalizer: ValueNormalizer | None,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    agent.load_state_dict(checkpoint["agent"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if obs_rms is not None and checkpoint.get("obs_rms") is not None:
        obs_rms.load_state_dict(checkpoint["obs_rms"])
    if state_rms is not None and checkpoint.get("state_rms") is not None:
        state_rms.load_state_dict(checkpoint["state_rms"])
    if value_normalizer is not None and checkpoint.get("value_normalizer") is not None:
        value_normalizer.load_state_dict(checkpoint["value_normalizer"])
    return int(checkpoint.get("update", 0)), int(checkpoint.get("global_step", 0))


def main() -> None:
    if args_cli.num_steps % args_cli.seq_length != 0:
        raise ValueError("--num_steps must be divisible by --seq_length for recurrent PPO")
    if args_cli.minibatch_size % args_cli.seq_length != 0:
        raise ValueError("--minibatch_size must be divisible by --seq_length for recurrent PPO")

    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    env_device = args_cli.env_device or args_cli.device
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args_cli.run_name or f"cleanrl_simtoolreal_{timestamp}"
    run_dir = args_cli.output_root / args_cli.wandb_project / args_cli.wandb_group / run_name
    ckpt_dir = run_dir / "checkpoints"
    video_dir = run_dir / "videos" / "train"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CLEANRL] task={args_cli.task} app_device={args_cli.device} env_device={env_device}", flush=True)
    print(f"[CLEANRL] run_dir={run_dir}", flush=True)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=env_device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.viewer.resolution = (1280, 720)
    env_cfg.viewer.cam_prim_path = "/OmniverseKit_Persp"
    env_cfg.log_dir = str(run_dir)

    # Match old launch_training.py overrides.
    env_cfg.object_scale_noise_multiplier_range = (args_cli.object_scale_noise_min, args_cli.object_scale_noise_max)
    env_cfg.force_consecutive_near_goal_steps = args_cli.force_consecutive_near_goal_steps
    env_cfg.force_scale = args_cli.force_scale
    env_cfg.torque_scale = args_cli.torque_scale
    env_cfg.object_ang_vel_penalty_scale = args_cli.object_ang_vel_penalty_scale

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.capture_video else None)
    wandb_run = None
    video_writer = None
    try:
        obs, _ = env.reset(seed=args_cli.seed)
        if args_cli.capture_video:
            _set_camera(env)

        device = torch.device(env.unwrapped.device)
        policy_obs_dim = int(obs["policy"].shape[-1])
        critic_obs_dim = int(obs["critic"].shape[-1])
        action_dim = int(env.action_space.shape[-1])
        sapg = _make_sapg(args_cli, args_cli.num_envs, device)
        coef_ids = sapg["coef_ids"] if sapg["enabled"] else None

        obs_rms = RunningMeanStd((policy_obs_dim,), epsilon=1.0e-4).to(device) if args_cli.norm_obs else None
        state_rms = RunningMeanStd((critic_obs_dim,), epsilon=1.0e-4).to(device) if args_cli.norm_obs else None
        value_normalizer = ValueNormalizer(epsilon=1.0e-4).to(device) if args_cli.norm_value else None
        agent = SimToolRealAgent(
            policy_obs_dim=policy_obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=action_dim,
            hidden_sizes=list(args_cli.hidden_sizes),
            lstm_hidden_size=args_cli.lstm_hidden_size,
            lstm_layers=args_cli.lstm_layers,
            coef_ids=coef_ids,
            extra_param_size=args_cli.extra_param_size if sapg["enabled"] else 0,
            fixed_sigma=args_cli.fixed_sigma if sapg["enabled"] else "fixed",
            value_normalizer=value_normalizer,
        ).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args_cli.learning_rate, eps=1.0e-8)
        scaler = torch.amp.GradScaler("cuda", enabled=args_cli.mixed_precision and device.type == "cuda")

        config = {
            **vars(args_cli),
            "run_dir": str(run_dir),
            "policy_obs_dim": policy_obs_dim,
            "critic_obs_dim": critic_obs_dim,
            "action_dim": action_dim,
            "sapg": {
                key: (value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value)
                for key, value in sapg.items()
                if key != "env_coef"
            },
        }
        config_path = run_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
        wandb_run = _try_init_wandb(args_cli, run_dir, config)

        start_update = 0
        global_step = 0
        if args_cli.checkpoint is not None:
            start_update, global_step = _load_checkpoint(args_cli.checkpoint, agent, optimizer, obs_rms, state_rms, value_normalizer)
            print(f"[CLEANRL] restored checkpoint={args_cli.checkpoint} update={start_update} global_step={global_step}", flush=True)

        policy_obs = obs["policy"].to(device)
        critic_obs = obs["critic"].to(device)
        next_done = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
        actor_state = agent.actor.initial_state(args_cli.num_envs, device)
        current_lr = args_cli.learning_rate
        best_reward = -float("inf")
        start_time = time.time()

        if args_cli.capture_video_env_id < 0 or args_cli.capture_video_env_id >= args_cli.num_envs:
            raise ValueError(f"--capture_video_env_id must be in [0, {args_cli.num_envs}), got {args_cli.capture_video_env_id}")
        control_step = global_step // args_cli.num_envs
        video_fps = _video_fps(env, args_cli)
        video_path: Path | None = None
        video_frames_written = 0
        # Old Isaac Gym behavior arms recording at step 0/frequency, but starts
        # the actual mp4 only when the viewed env resets.
        video_capture_pending = bool(args_cli.capture_video)

        batch_size = args_cli.num_envs * args_cli.num_steps
        sequences_per_minibatch = max(1, args_cli.minibatch_size // args_cli.seq_length)

        for update in range(start_update + 1, args_cli.total_updates + 1):
            rollout_start = time.time()
            completed_video_paths: list[str] = []
            initial_actor_state = (actor_state[0].detach().clone(), actor_state[1].detach().clone())

            obs_buf = torch.zeros((args_cli.num_steps, args_cli.num_envs, policy_obs_dim), device=device)
            critic_obs_buf = torch.zeros((args_cli.num_steps, args_cli.num_envs, critic_obs_dim), device=device)
            actions_buf = torch.zeros((args_cli.num_steps, args_cli.num_envs, action_dim), device=device)
            old_logprobs_buf = torch.zeros((args_cli.num_steps, args_cli.num_envs), device=device)
            rewards_buf = torch.zeros((args_cli.num_steps, args_cli.num_envs), device=device)
            dones_buf = torch.zeros((args_cli.num_steps, args_cli.num_envs), dtype=torch.bool, device=device)
            actor_dones_buf = torch.zeros_like(dones_buf)
            values_buf = torch.zeros((args_cli.num_steps, args_cli.num_envs), device=device)
            mu_buf = torch.zeros((args_cli.num_steps, args_cli.num_envs, action_dim), device=device)
            sigma_buf = torch.zeros((args_cli.num_steps, args_cli.num_envs, action_dim), device=device)

            for step in range(args_cli.num_steps):
                obs_buf[step] = policy_obs
                critic_obs_buf[step] = critic_obs
                dones_buf[step] = next_done
                actor_dones_buf[step] = next_done

                if obs_rms is not None:
                    obs_rms.update(policy_obs)
                    state_rms.update(critic_obs)
                norm_policy = _normalize_base_and_append(policy_obs, obs_rms, sapg["env_coef"])
                norm_critic = _normalize_base_and_append(critic_obs, state_rms, sapg["env_coef"])

                with torch.no_grad():
                    action, logprob, _entropy, value, mu, sigma, actor_state = agent.get_action_and_value(
                        norm_policy,
                        norm_critic,
                        actor_state,
                        next_done,
                    )
                actions_buf[step] = action
                old_logprobs_buf[step] = logprob
                values_buf[step] = value
                mu_buf[step] = mu
                sigma_buf[step] = sigma

                obs, reward, terminated, truncated, _info = env.step(torch.clamp(action, -1.0, 1.0))
                next_done = terminated | truncated
                rewards_buf[step] = reward * args_cli.reward_scale
                policy_obs = obs["policy"].to(device)
                critic_obs = obs["critic"].to(device)
                global_step += args_cli.num_envs
                control_step += 1

                if next_done.any():
                    done_mask = (1.0 - next_done.float()).reshape(1, args_cli.num_envs, 1)
                    actor_state = (actor_state[0] * done_mask, actor_state[1] * done_mask)

                if args_cli.capture_video:
                    if (
                        video_writer is None
                        and not video_capture_pending
                        and args_cli.capture_video_freq > 0
                        and control_step % args_cli.capture_video_freq == 0
                    ):
                        print(
                            "-" * 80
                            + f"\nAt control_step={control_step}, should start video capture at start of next episode\n"
                            + "-" * 80,
                            flush=True,
                        )
                        video_capture_pending = True

                    should_start_video = video_writer is None and video_capture_pending
                    if should_start_video and args_cli.capture_video_start_on_reset:
                        should_start_video = bool(next_done[args_cli.capture_video_env_id].item())
                    if should_start_video:
                        video_path = video_dir / f"{timestamp}_video_{control_step}.mp4"
                        print("-" * 80, flush=True)
                        print(f"Starting to capture video frames: {video_path}", flush=True)
                        print("-" * 80, flush=True)
                        video_writer = imageio.get_writer(str(video_path), fps=video_fps, macro_block_size=1)
                        video_frames_written = 0
                        video_capture_pending = False

                    if video_writer is not None:
                        _write_video_frame(video_writer, env)
                        video_frames_written += 1
                        if video_frames_written >= args_cli.capture_video_len:
                            assert video_path is not None
                            video_writer.close()
                            video_writer = None
                            completed_video_paths.append(str(video_path))
                            print("-" * 80, flush=True)
                            print(f"Saved video to {video_path}", flush=True)
                            print("-" * 80, flush=True)
                            video_path = None
                            video_frames_written = 0

            with torch.no_grad():
                norm_next_critic = _normalize_base_and_append(critic_obs, state_rms, sapg["env_coef"])
                next_value = agent.get_value(norm_next_critic, denormalize=True)
                advantages, returns = _compute_gae(
                    rewards_buf,
                    dones_buf,
                    values_buf,
                    next_value,
                    next_done,
                    args_cli.gamma,
                    args_cli.gae_lambda,
                )
                if value_normalizer is not None:
                    value_normalizer.update(returns)

            batch = _make_sequence_batch(
                args=args_cli,
                agent=agent,
                sapg=sapg,
                policy_obs=obs_buf,
                critic_obs=critic_obs_buf,
                actions=actions_buf,
                old_logprobs=old_logprobs_buf,
                old_values=values_buf,
                old_mu=mu_buf,
                old_sigma=sigma_buf,
                rewards=rewards_buf,
                dones=dones_buf,
                advantages=advantages,
                returns=returns,
                initial_actor_state=initial_actor_state,
                actor_dones=actor_dones_buf,
                obs_rms=obs_rms,
                state_rms=state_rms,
                next_critic_obs=critic_obs,
                next_done=next_done,
            )

            if args_cli.norm_adv:
                adv = batch["advantages"]
                batch["advantages"] = (adv - adv.mean()) / (adv.std(unbiased=False) + 1.0e-8)

            num_sequences = batch["policy_obs"].shape[0]
            sequence_indices = torch.arange(num_sequences, device=device)
            actor_losses = []
            critic_losses = []
            entropies = []
            approx_kls = []
            clipfracs = []
            bounds_losses = []

            for _epoch in range(args_cli.update_epochs):
                sequence_indices = sequence_indices[torch.randperm(num_sequences, device=device)]
                epoch_kls = []
                for start in range(0, num_sequences, sequences_per_minibatch):
                    mb_inds = sequence_indices[start : start + sequences_per_minibatch]
                    mb_policy = batch["policy_obs"][mb_inds]
                    mb_critic = batch["critic_obs"][mb_inds]
                    mb_actions = batch["actions"][mb_inds]
                    mb_old_logprobs = batch["old_logprobs"][mb_inds]
                    mb_old_values = batch["old_values"][mb_inds]
                    mb_old_mu = batch["old_mu"][mb_inds]
                    mb_old_sigma = batch["old_sigma"][mb_inds]
                    mb_returns = batch["returns"][mb_inds]
                    mb_advantages = batch["advantages"][mb_inds]
                    mb_dones = batch["dones"][mb_inds]
                    mb_entropy_coef = batch["entropy_coef"][mb_inds]
                    mb_h0 = batch["h0"][mb_inds].transpose(0, 1).contiguous()
                    mb_c0 = batch["c0"][mb_inds].transpose(0, 1).contiguous()

                    with torch.amp.autocast("cuda", enabled=args_cli.mixed_precision and device.type == "cuda"):
                        new_mu, new_sigma, _ = agent.actor.forward_sequence(mb_policy, (mb_h0, mb_c0), mb_dones)
                        dist = Normal(new_mu, new_sigma)
                        new_logprobs = dist.log_prob(mb_actions).sum(dim=-1)
                        entropy = dist.entropy().sum(dim=-1)
                        ratio = (new_logprobs - mb_old_logprobs).exp()
                        pg_loss_1 = -mb_advantages * ratio
                        pg_loss_2 = -mb_advantages * torch.clamp(ratio, 1.0 - args_cli.clip_coef, 1.0 + args_cli.clip_coef)
                        actor_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                        flat_critic = mb_critic.reshape(-1, mb_critic.shape[-1])
                        new_value_raw = agent.critic(flat_critic).reshape_as(mb_returns)
                        if value_normalizer is not None:
                            new_value = value_normalizer.denormalize(new_value_raw)
                            target_value = value_normalizer.normalize(mb_returns.reshape(-1)).reshape_as(mb_returns)
                        else:
                            new_value = new_value_raw
                            target_value = mb_returns
                        if args_cli.clip_value:
                            v_clipped = mb_old_values + torch.clamp(new_value - mb_old_values, -args_cli.clip_coef, args_cli.clip_coef)
                            if value_normalizer is not None:
                                v_clipped_raw = value_normalizer.normalize(v_clipped.reshape(-1)).reshape_as(mb_returns)
                                v_loss_unclipped = (new_value_raw - target_value).square()
                                v_loss_clipped = (v_clipped_raw - target_value).square()
                            else:
                                v_loss_unclipped = (new_value - mb_returns).square()
                                v_loss_clipped = (v_clipped - mb_returns).square()
                            critic_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                        else:
                            critic_loss = 0.5 * (new_value_raw - target_value).square().mean()

                        entropy_loss = (mb_entropy_coef * entropy).mean()
                        bounds_loss = _bound_loss(new_mu).mean()
                        loss = actor_loss + args_cli.critic_coef * critic_loss - entropy_loss + args_cli.bounds_loss_coef * bounds_loss

                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(agent.parameters(), args_cli.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    with torch.no_grad():
                        kl = _policy_kl(mb_old_mu, mb_old_sigma, new_mu.detach(), new_sigma.detach()).mean()
                        clipfrac = ((ratio - 1.0).abs() > args_cli.clip_coef).float().mean()
                    actor_losses.append(actor_loss.detach())
                    critic_losses.append(critic_loss.detach())
                    entropies.append(entropy.detach().mean())
                    approx_kls.append(kl.detach())
                    epoch_kls.append(kl.detach())
                    clipfracs.append(clipfrac.detach())
                    bounds_losses.append(bounds_loss.detach())

                if epoch_kls:
                    mean_epoch_kl = torch.stack(epoch_kls).mean().item()
                    current_lr = _update_adaptive_lr(current_lr, args_cli.kl_threshold, mean_epoch_kl)
                    _set_lr(optimizer, current_lr)

            rollout_time = time.time() - rollout_start
            reward_mean = float((rewards_buf / args_cli.reward_scale).mean().item())
            episodic_success = float(env.unwrapped.successes.float().mean().item())
            stats = TrainStats(
                update=update,
                global_step=global_step,
                fps=batch_size / max(rollout_time, 1.0e-6),
                reward_mean=reward_mean,
                episodic_success_mean=episodic_success,
                actor_loss=float(torch.stack(actor_losses).mean().item()),
                critic_loss=float(torch.stack(critic_losses).mean().item()),
                entropy=float(torch.stack(entropies).mean().item()),
                approx_kl=float(torch.stack(approx_kls).mean().item()),
                clipfrac=float(torch.stack(clipfracs).mean().item()),
                bounds_loss=float(torch.stack(bounds_losses).mean().item()),
                learning_rate=current_lr,
                sigma_mean=float(batch["old_sigma"].mean().item()),
                video_path=completed_video_paths[-1] if completed_video_paths else None,
            )

            print(
                "[CLEANRL] "
                f"update={stats.update} step={stats.global_step} fps={stats.fps:.0f} "
                f"reward_mean={stats.reward_mean:.4f} success={stats.episodic_success_mean:.4f} "
                f"actor_loss={stats.actor_loss:.5f} critic_loss={stats.critic_loss:.5f} "
                f"entropy={stats.entropy:.4f} kl={stats.approx_kl:.5f} lr={stats.learning_rate:.2e}",
                flush=True,
            )

            if wandb_run is not None:
                log_payload = {f"train/{key}": value for key, value in asdict(stats).items() if key != "video_path"}
                log_payload.update(
                    {
                        "global_step": stats.global_step,
                        "epoch": stats.update,
                        "performance/step_fps": stats.fps,
                        "performance/step_inference_rl_update_fps": stats.fps,
                        "losses/actor_loss": stats.actor_loss,
                        "losses/critic_loss": stats.critic_loss,
                        "losses/bounds_loss": stats.bounds_loss,
                        "losses/entropy": stats.entropy,
                        "losses/approx_kl": stats.approx_kl,
                        "info/last_lr": stats.learning_rate,
                        "info/sigma_mean": stats.sigma_mean,
                    }
                )
                log_payload.update(_flatten_scalar_dict(env.unwrapped.extras.get("log", {}), prefix="env"))
                if stats.video_path is not None:
                    import wandb

                    # Old SimToolReal logs under key "video"; keep that exact key
                    # so W&B dashboards/media panels match the Isaac Gym runs.
                    log_payload["video"] = wandb.Video(stats.video_path, fps=video_fps, format="mp4")
                    log_payload["train/video_path"] = stats.video_path
                wandb_run.log(log_payload, step=global_step)

            if update % args_cli.save_frequency == 0:
                _save_checkpoint(ckpt_dir / f"checkpoint_{update:06d}.pt", agent, optimizer, obs_rms, state_rms, value_normalizer, update, global_step, args_cli)
            if update >= args_cli.save_best_after and reward_mean > best_reward:
                best_reward = reward_mean
                _save_checkpoint(ckpt_dir / "best.pt", agent, optimizer, obs_rms, state_rms, value_normalizer, update, global_step, args_cli)
            _save_checkpoint(ckpt_dir / "latest.pt", agent, optimizer, obs_rms, state_rms, value_normalizer, update, global_step, args_cli)

        elapsed = time.time() - start_time
        print(f"[CLEANRL] finished updates={args_cli.total_updates} elapsed_sec={elapsed:.1f}", flush=True)
        print(f"[CLEANRL] latest_checkpoint={ckpt_dir / 'latest.pt'}", flush=True)
    finally:
        if video_writer is not None:
            video_writer.close()
        if wandb_run is not None:
            wandb_run.finish()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"[CLEANRL][ERROR] {type(err).__name__}: {err}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
