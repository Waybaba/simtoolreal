"""Cluster-only training smoke test that exits cleanly after success."""

import os
import sys

import hydra
import yaml
from omegaconf import DictConfig

from isaacgymenvs.train import launch_rlg_hydra
from isaacgymenvs.utils.reformat import omegaconf_to_dict


@hydra.main(version_base="1.1", config_name="config", config_path="../isaacgymenvs/cfg")
def launch(cfg: DictConfig) -> None:
    if os.path.exists(cfg.checkpoint_yaml):
        with open(cfg.checkpoint_yaml, "r") as f:
            checkpoint_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        for param, value in checkpoint_cfg["params"].items():
            path = param.split(".")
            temp_cfg = cfg
            for param_name in path[:-1]:
                temp_cfg = temp_cfg[param_name]
            temp_cfg[path[-1]] = value

    vec_env = None
    while True:
        cfg_n_env = launch_rlg_hydra(cfg, vec_env)
        if not isinstance(cfg_n_env, tuple):
            break

        if len(cfg_n_env) != 2 or not hasattr(cfg_n_env[1], "change_on_restart"):
            break

        cfg, vec_env = cfg_n_env
        vec_env.change_on_restart(omegaconf_to_dict(cfg.task))

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    launch()
