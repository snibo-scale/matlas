"""Termination terms for Matlas skill tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def base_height_out_of_range(
    env: ManagerBasedRlEnv,
    minimum: float,
    maximum: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    height = asset.data.root_link_pos_w[:, 2]
    return (height < minimum) | (height > maximum)


def dof_velocity_out_of_limit(
    env: ManagerBasedRlEnv,
    max_velocity: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    speeds = torch.cat(
        [
            asset.data.joint_vel,
            asset.data.root_link_lin_vel_b,
            asset.data.root_link_ang_vel_b,
        ],
        dim=1,
    )
    return speeds.abs().amax(dim=1) > max_velocity
