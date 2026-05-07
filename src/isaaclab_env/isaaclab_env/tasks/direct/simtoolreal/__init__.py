"""SimToolReal direct task registrations."""

import gymnasium as gym


gym.register(
    id="SimToolReal-Direct-v0",
    entry_point=f"{__name__}.env:SimToolRealDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env:SimToolRealDirectEnvCfg",
    },
)

gym.register(
    id="SimToolReal-Direct-Debug-v0",
    entry_point=f"{__name__}.env:SimToolRealDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env:SimToolRealDirectDebugEnvCfg",
    },
)

gym.register(
    id="SimToolReal-Direct-Play-v0",
    entry_point=f"{__name__}.env:SimToolRealDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env:SimToolRealDirectPlayEnvCfg",
    },
)
