"""Co-design reward / cost terms (pair with negative weights).

Electrical energy is covered by mjlab's built-in ``mdp.electrical_power_cost``.
Here we add gearbox friction loss and an over-speed penalty enforcing the
design's (soft) motor velocity limit, since MuJoCo motors have no native speed
clamp.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from matlas.codesign.events import get_state

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def over_speed_cost(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
    """mean(relu(|q̇| - design_velocity_limit)^2) over actuated joints."""
    state = get_state(env, asset_cfg.name)
    asset: Entity = env.scene[asset_cfg.name]
    excess = torch.relu(asset.data.joint_vel.abs() - state.vel_limit)
    return torch.mean(excess**2, dim=1)


def friction_power_cost(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
    """sum(dof_frictionloss * |q̇|) over actuated joints (gearbox Coulomb loss)."""
    state = get_state(env, asset_cfg.name)
    asset: Entity = env.scene[asset_cfg.name]
    friction = env.sim.model.dof_frictionloss[:, state.v_adr]
    speed = asset.data.joint_vel[:, state.joint_cols].abs()
    return torch.sum(friction * speed, dim=1)
