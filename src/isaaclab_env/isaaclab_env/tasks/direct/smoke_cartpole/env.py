"""Direct Cartpole config used to verify the local Isaac Lab extension loads."""

from isaaclab.utils import configclass
from isaaclab_tasks.direct.cartpole.cartpole_env import CartpoleEnvCfg


@configclass
class SimToolRealSmokeCartpoleDirectEnvCfg(CartpoleEnvCfg):
    """Small direct Cartpole config for fast local smoke checks."""

    def __post_init__(self) -> None:
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.0
        self.scene.clone_in_fabric = False
