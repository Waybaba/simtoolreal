"""Shared CleanRL policy loader for frozen SimToolReal low-level controllers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    def __init__(self, shape: tuple[int, ...], epsilon: float = 1.0e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float32))

    def normalize(self, x: torch.Tensor, clip: float = 5.0) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1.0e-8), -clip, clip)


class ValueNormalizer(nn.Module):
    def __init__(self, epsilon: float = 1.0e-4):
        super().__init__()
        self.rms = RunningMeanStd((1,), epsilon=epsilon)

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
        self.input_adapter = ExtraParamInput(obs_dim, coef_ids, extra_param_size)
        self.lstm = nn.LSTM(self.input_adapter.output_dim, lstm_hidden_size, num_layers=lstm_layers)
        self.layer_norm = nn.LayerNorm(lstm_hidden_size)
        self.mlp = _make_mlp(lstm_hidden_size, hidden_sizes)
        self.mu = _init_linear(nn.Linear(hidden_sizes[-1], action_dim), std=0.01)
        self.fixed_sigma = fixed_sigma
        if fixed_sigma == "coef_cond":
            assert coef_ids is not None
            self.register_buffer("sigma_ids", coef_ids.float().clone())
            self.log_sigma = nn.Parameter(torch.zeros((len(coef_ids), action_dim), dtype=torch.float32))
        else:
            self.register_buffer("sigma_ids", torch.empty(0))
            self.log_sigma = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

    def initial_state(self, num_envs: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros((self.lstm.num_layers, num_envs, self.lstm.hidden_size), dtype=torch.float32, device=device)
        return h, torch.zeros_like(h)

    def _sigma(self, obs_with_optional_coef: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        if self.fixed_sigma == "coef_cond":
            coef = obs_with_optional_coef[:, -1].reshape(-1, 1)
            block_ids = (coef == self.sigma_ids.reshape(1, -1)).float().argmax(dim=1)
            return torch.exp(self.log_sigma[block_ids])
        return torch.exp(self.log_sigma).expand_as(mu)

    def forward_step(
        self,
        obs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        done: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size = obs.shape[0]
        h, c = state
        not_done = (1.0 - done.float()).reshape(1, batch_size, 1)
        adapted = self.input_adapter(obs).unsqueeze(0)
        out, next_state = self.lstm(adapted, (h * not_done, c * not_done))
        features = self.mlp(self.layer_norm(out.squeeze(0)))
        mu = self.mu(features)
        return mu, self._sigma(obs, mu), next_state


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


def _append_coef(obs: torch.Tensor, env_coef: torch.Tensor | None) -> torch.Tensor:
    if env_coef is None:
        return obs
    return torch.cat([obs, env_coef], dim=-1)


def make_sapg(ckpt_args: dict[str, Any], num_envs: int, device: torch.device) -> dict[str, Any]:
    if ckpt_args.get("expl_type", "mixed_expl_learn_param") == "none":
        return {"enabled": False, "coef_ids": None, "env_coef": None}
    num_blocks = int(ckpt_args.get("sapg_num_blocks", 6))
    if num_envs % num_blocks != 0:
        raise ValueError(f"--num_envs={num_envs} must be divisible by SAPG block count {num_blocks}")
    block_size = num_envs // num_blocks
    coef_ids = torch.linspace(50.0, 0.0, num_blocks, device=device)
    block_ids = torch.arange(num_blocks, device=device).repeat_interleave(block_size)
    return {"enabled": True, "coef_ids": coef_ids, "env_coef": coef_ids[block_ids].unsqueeze(-1)}


@dataclass
class LoadedLowLevelPolicy:
    checkpoint: dict[str, Any]
    ckpt_args: dict[str, Any]
    agent: SimToolRealAgent
    obs_rms: RunningMeanStd | None
    state_rms: RunningMeanStd | None
    value_normalizer: ValueNormalizer | None
    env_coef: torch.Tensor | None

    def initial_state(self, num_envs: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.agent.actor.initial_state(num_envs, device)

    def act(
        self,
        policy_obs: torch.Tensor,
        actor_state: tuple[torch.Tensor, torch.Tensor],
        done: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        norm_policy = self.obs_rms.normalize(policy_obs) if self.obs_rms is not None else policy_obs
        norm_policy = _append_coef(norm_policy, self.env_coef)
        mu, _sigma, actor_state = self.agent.actor.forward_step(norm_policy, actor_state, done)
        return torch.clamp(mu, -1.0, 1.0), actor_state


def load_low_level_policy(
    checkpoint_path: Path,
    *,
    policy_obs_dim: int,
    critic_obs_dim: int,
    action_dim: int,
    num_envs: int,
    device: torch.device,
) -> LoadedLowLevelPolicy:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    sapg = make_sapg(ckpt_args, num_envs, device)
    coef_ids = sapg["coef_ids"] if sapg["enabled"] else None
    obs_rms = RunningMeanStd((policy_obs_dim,), epsilon=1.0e-4).to(device) if ckpt_args.get("norm_obs", True) else None
    state_rms = RunningMeanStd((critic_obs_dim,), epsilon=1.0e-4).to(device) if ckpt_args.get("norm_obs", True) else None
    value_normalizer = ValueNormalizer(epsilon=1.0e-4).to(device) if ckpt_args.get("norm_value", True) else None
    agent = SimToolRealAgent(
        policy_obs_dim=policy_obs_dim,
        critic_obs_dim=critic_obs_dim,
        action_dim=action_dim,
        hidden_sizes=list(ckpt_args.get("hidden_sizes", [1024, 1024, 512, 512])),
        lstm_hidden_size=int(ckpt_args.get("lstm_hidden_size", 1024)),
        lstm_layers=int(ckpt_args.get("lstm_layers", 1)),
        coef_ids=coef_ids,
        extra_param_size=int(ckpt_args.get("extra_param_size", 32)) if sapg["enabled"] else 0,
        fixed_sigma=ckpt_args.get("fixed_sigma", "coef_cond") if sapg["enabled"] else "fixed",
        value_normalizer=value_normalizer,
    ).to(device)
    agent.load_state_dict(checkpoint["agent"])
    if obs_rms is not None and checkpoint.get("obs_rms") is not None:
        obs_rms.load_state_dict(checkpoint["obs_rms"])
    if state_rms is not None and checkpoint.get("state_rms") is not None:
        state_rms.load_state_dict(checkpoint["state_rms"])
    if value_normalizer is not None and checkpoint.get("value_normalizer") is not None:
        value_normalizer.load_state_dict(checkpoint["value_normalizer"])
    agent.eval()
    return LoadedLowLevelPolicy(
        checkpoint=checkpoint,
        ckpt_args=ckpt_args,
        agent=agent,
        obs_rms=obs_rms,
        state_rms=state_rms,
        value_normalizer=value_normalizer,
        env_coef=sapg["env_coef"],
    )
