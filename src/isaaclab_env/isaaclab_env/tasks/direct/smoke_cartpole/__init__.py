"""Simple direct Cartpole task registration."""

import gymnasium as gym


gym.register(
    id="SimToolReal-Smoke-Cartpole-Direct-v0",
    entry_point="isaaclab_tasks.direct.cartpole.cartpole_env:CartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env:SimToolRealSmokeCartpoleDirectEnvCfg",
    },
)

