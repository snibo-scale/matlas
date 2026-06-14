"""Reward terms for the Matlas standing task.

These reproduce ``envs/stand_env.py::MatlasStandEnv._reward`` term-for-term in
batched torch form. Shaping terms (``upright``/``height``/``posture``/
``stillness``/``angular_stillness``) return a positive [0, 1] tolerance and take
a positive weight. Cost terms (``torque_cost``/``joint_speed_cost``/
``action_rate_cost``) return a positive magnitude and take a negative weight.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")
_ALL_JOINTS = SceneEntityCfg("robot", joint_names=(".*",))

# Matches the legacy max_qvel clamp applied before stillness terms.
_QVEL_CLAMP = 40.0

# Cache of per-actuator effort limits, keyed by (id(env), device).
_EFFORT_LIMITS: dict[tuple[int, str], torch.Tensor] = {}


def _wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return (x + torch.pi) % (2.0 * torch.pi) - torch.pi


def upright(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """exp(4 * (R22 - 1)); R22 is the base-z component of world up."""
    asset: Entity = env.scene[asset_cfg.name]
    r22 = -asset.data.projected_gravity_b[:, 2]
    return torch.exp(4.0 * (r22 - 1.0))


def height_tracking(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """exp(-20 * height_error^2) about the nominal standing height."""
    asset: Entity = env.scene[asset_cfg.name]
    err = asset.data.root_link_pos_w[:, 2] - target_height
    return torch.exp(-20.0 * err**2)


def posture(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ALL_JOINTS
) -> torch.Tensor:
    """exp(-3 * mean(wrapped(q - q_default)^2)) over controlled joints."""
    asset: Entity = env.scene[asset_cfg.name]
    jids = asset_cfg.joint_ids
    err = _wrap_angle(asset.data.joint_pos[:, jids] - asset.data.default_joint_pos[:, jids])
    return torch.exp(-3.0 * torch.mean(err**2, dim=1))


def stillness(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """exp(-2 * mean(base_linear_velocity^2)) in the base frame."""
    asset: Entity = env.scene[asset_cfg.name]
    vel = asset.data.root_link_lin_vel_b.clamp(-_QVEL_CLAMP, _QVEL_CLAMP)
    return torch.exp(-2.0 * torch.mean(vel**2, dim=1))


def angular_stillness(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
    """exp(-2 * mean(base_angular_velocity^2)) in the base frame."""
    asset: Entity = env.scene[asset_cfg.name]
    ang = asset.data.root_link_ang_vel_b.clamp(-_QVEL_CLAMP, _QVEL_CLAMP)
    return torch.exp(-2.0 * torch.mean(ang**2, dim=1))


def _effort_limits(env: ManagerBasedRlEnv, asset: Entity) -> torch.Tensor:
    key = (id(env), str(asset.data.actuator_force.device))
    limits = _EFFORT_LIMITS.get(key)
    if limits is None:
        vals = [
            max(abs(float(a.forcerange[0])), abs(float(a.forcerange[1])))
            for a in asset.spec.actuators
        ]
        limits = torch.tensor(
            vals, device=asset.data.actuator_force.device, dtype=torch.float
        )
        _EFFORT_LIMITS[key] = limits
    return limits


def torque_cost(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """mean((actuator_force / effort_limit)^2) -- pair with a negative weight."""
    asset: Entity = env.scene[asset_cfg.name]
    normalized = asset.data.actuator_force / _effort_limits(env, asset).clamp_min(1e-6)
    return torch.mean(normalized**2, dim=1)


def joint_speed_cost(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ALL_JOINTS
) -> torch.Tensor:
    """mean(joint_velocity^2) -- pair with a negative weight."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.mean(asset.data.joint_vel[:, asset_cfg.joint_ids] ** 2, dim=1)


def action_rate_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
    """mean((action - prev_action)^2) -- pair with a negative weight."""
    delta = env.action_manager.action - env.action_manager.prev_action
    return torch.mean(delta**2, dim=1)
