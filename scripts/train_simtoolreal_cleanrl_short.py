#!/usr/bin/env python3
"""CleanRL-style PPO smoke training for the SimToolReal Isaac Lab direct env."""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional


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


parser = argparse.ArgumentParser(description="Short CleanRL-style PPO training with rollout video.")
parser.add_argument("--task", type=str, default="SimToolReal-Direct-Debug-v0", help="Gym task ID to train.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments.")
parser.add_argument("--num_steps", type=int, default=16, help="Rollout steps per PPO update.")
parser.add_argument("--num_updates", type=int, default=2, help="Number of PPO updates.")
parser.add_argument("--learning_rate", type=float, default=3.0e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--update_epochs", type=int, default=2)
parser.add_argument("--num_minibatches", type=int, default=2)
parser.add_argument("--clip_coef", type=float, default=0.2)
parser.add_argument("--ent_coef", type=float, default=0.01)
parser.add_argument("--vf_coef", type=float, default=1.0)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--video_length", type=int, default=64, help="Max frames to write from the training rollout.")
parser.add_argument("--video_fps", type=int, default=15)
parser.add_argument("--disable_fabric", action="store_true", help="Disable Fabric scene cloning/render path.")
parser.add_argument("--env_device", type=str, default=None, help="Optional env device override.")
parser.add_argument("--kit_active_gpu", type=int, default=None, help="Optional Kit renderer activeGpu override.")
parser.add_argument("--kit_physics_gpu", type=int, default=None, help="Optional Kit /physics/cudaDevice override.")
parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "outputs" / "isaaclab_cleanrl_short_train")
parser.add_argument("--seed", type=int, default=7, help="Seed for env and PPO.")
CleanRLAppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, enable_cameras=True)
args_cli = parser.parse_args()

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


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOAgent(nn.Module):
    def __init__(self, policy_obs_dim: int, critic_obs_dim: int, action_dim: int):
        super().__init__()
        self.actor_mean = nn.Sequential(
            _layer_init(nn.Linear(policy_obs_dim, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(critic_obs_dim, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            _layer_init(nn.Linear(256, 1), std=1.0),
        )

    def get_value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_obs)

    def get_action_and_value(
        self,
        policy_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean = self.actor_mean(policy_obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(critic_obs),
        )


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


def _write_frame(writer, env) -> None:
    frame = np.asarray(env.render())
    if frame.size == 0:
        raise RuntimeError("env.render() returned an empty frame.")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    writer.append_data(frame)


def main() -> None:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    env_device = args_cli.env_device or args_cli.device
    run_dir = args_cli.output_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_cleanrl_video")
    video_dir = run_dir / "videos" / "train"
    video_path = video_dir / "cleanrl_training_rollout.mp4"
    run_dir.mkdir(parents=True, exist_ok=True)
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

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    writer = None
    try:
        obs, _ = env.reset(seed=args_cli.seed)
        _set_camera(env)
        policy_obs = obs["policy"].to(env.unwrapped.device)
        critic_obs = obs["critic"].to(env.unwrapped.device)
        next_done = torch.zeros(args_cli.num_envs, device=env.unwrapped.device)

        policy_obs_dim = policy_obs.shape[-1]
        critic_obs_dim = critic_obs.shape[-1]
        action_dim = env.action_space.shape[-1]
        agent = PPOAgent(policy_obs_dim, critic_obs_dim, action_dim).to(env.unwrapped.device)
        optimizer = optim.Adam(agent.parameters(), lr=args_cli.learning_rate, eps=1e-5)

        obs_buf = torch.zeros((args_cli.num_steps, args_cli.num_envs, policy_obs_dim), device=env.unwrapped.device)
        critic_obs_buf = torch.zeros((args_cli.num_steps, args_cli.num_envs, critic_obs_dim), device=env.unwrapped.device)
        actions_buf = torch.zeros((args_cli.num_steps, args_cli.num_envs, action_dim), device=env.unwrapped.device)
        logprobs_buf = torch.zeros((args_cli.num_steps, args_cli.num_envs), device=env.unwrapped.device)
        rewards_buf = torch.zeros((args_cli.num_steps, args_cli.num_envs), device=env.unwrapped.device)
        dones_buf = torch.zeros((args_cli.num_steps, args_cli.num_envs), device=env.unwrapped.device)
        values_buf = torch.zeros((args_cli.num_steps, args_cli.num_envs), device=env.unwrapped.device)

        writer = imageio.get_writer(str(video_path), fps=args_cli.video_fps, macro_block_size=1)
        frames_written = 0
        batch_size = args_cli.num_envs * args_cli.num_steps
        minibatch_size = batch_size // args_cli.num_minibatches
        batch_indices = np.arange(batch_size)

        for update in range(args_cli.num_updates):
            for step in range(args_cli.num_steps):
                obs_buf[step] = policy_obs
                critic_obs_buf[step] = critic_obs
                dones_buf[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(policy_obs, critic_obs)
                actions_buf[step] = action
                logprobs_buf[step] = logprob
                values_buf[step] = value.flatten()

                obs, reward, terminated, truncated, info = env.step(torch.clamp(action, -1.0, 1.0))
                next_done = (terminated | truncated).float()
                rewards_buf[step] = reward
                policy_obs = obs["policy"].to(env.unwrapped.device)
                critic_obs = obs["critic"].to(env.unwrapped.device)

                if frames_written < args_cli.video_length:
                    _write_frame(writer, env)
                    frames_written += 1

            with torch.no_grad():
                next_value = agent.get_value(critic_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards_buf)
                last_gae_lam = 0.0
                for t in reversed(range(args_cli.num_steps)):
                    if t == args_cli.num_steps - 1:
                        next_non_terminal = 1.0 - next_done
                        next_values = next_value
                    else:
                        next_non_terminal = 1.0 - dones_buf[t + 1]
                        next_values = values_buf[t + 1]
                    delta = rewards_buf[t] + args_cli.gamma * next_values * next_non_terminal - values_buf[t]
                    advantages[t] = last_gae_lam = delta + args_cli.gamma * args_cli.gae_lambda * next_non_terminal * last_gae_lam
                returns = advantages + values_buf

            flat_obs = obs_buf.reshape((-1, policy_obs_dim))
            flat_critic_obs = critic_obs_buf.reshape((-1, critic_obs_dim))
            flat_actions = actions_buf.reshape((-1, action_dim))
            flat_logprobs = logprobs_buf.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)
            flat_values = values_buf.reshape(-1)

            for _ in range(args_cli.update_epochs):
                np.random.shuffle(batch_indices)
                for start in range(0, batch_size, minibatch_size):
                    mb_inds = batch_indices[start : start + minibatch_size]
                    _, new_logprob, entropy, new_value = agent.get_action_and_value(
                        flat_obs[mb_inds],
                        flat_critic_obs[mb_inds],
                        flat_actions[mb_inds],
                    )
                    logratio = new_logprob - flat_logprobs[mb_inds]
                    ratio = logratio.exp()
                    mb_advantages = flat_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    pg_loss_1 = -mb_advantages * ratio
                    pg_loss_2 = -mb_advantages * torch.clamp(ratio, 1 - args_cli.clip_coef, 1 + args_cli.clip_coef)
                    pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()
                    new_value = new_value.view(-1)
                    v_loss_unclipped = (new_value - flat_returns[mb_inds]) ** 2
                    v_clipped = flat_values[mb_inds] + torch.clamp(
                        new_value - flat_values[mb_inds], -args_cli.clip_coef, args_cli.clip_coef
                    )
                    v_loss_clipped = (v_clipped - flat_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    entropy_loss = entropy.mean()
                    loss = pg_loss - args_cli.ent_coef * entropy_loss + args_cli.vf_coef * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args_cli.max_grad_norm)
                    optimizer.step()

            print(
                "[CLEANRL] "
                f"update={update + 1}/{args_cli.num_updates} "
                f"reward_mean={float(rewards_buf.mean().item()):.4f} "
                f"value_loss={float(v_loss.item()):.4f} "
                f"policy_loss={float(pg_loss.item()):.4f} "
                f"frames={frames_written}",
                flush=True,
            )

        writer.close()
        writer = None
        torch.save(agent.state_dict(), run_dir / "cleanrl_agent.pt")
        if frames_written == 0:
            raise RuntimeError("No video frames were recorded.")
        print(f"[CLEANRL] video_path={video_path}", flush=True)
        print(f"[CLEANRL] checkpoint={run_dir / 'cleanrl_agent.pt'}", flush=True)
    finally:
        if writer is not None:
            writer.close()
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
