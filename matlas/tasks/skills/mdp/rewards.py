"""Reward terms for Matlas actuation-characterization skills."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")
_ALL_JOINTS = SceneEntityCfg("robot", joint_names=(".*",))


def _asset(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> Entity:
    return env.scene[asset_cfg.name]


def _phase(env: ManagerBasedRlEnv) -> torch.Tensor:
    return (env.episode_length_buf.float() / max(env.max_episode_length, 1)).clamp(
        0.0, 1.0
    )


def _wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return (x + torch.pi) % (2.0 * torch.pi) - torch.pi


def _pitch_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(dim=1)
    sinp = 2.0 * (w * y - z * x)
    return torch.asin(sinp.clamp(-1.0, 1.0))


def _effort_limits(env: ManagerBasedRlEnv, asset: Entity) -> torch.Tensor:
    vals = [
        max(abs(float(a.forcerange[0])), abs(float(a.forcerange[1])))
        for a in asset.spec.actuators
    ]
    return torch.tensor(vals, device=asset.data.actuator_force.device, dtype=torch.float)


def height_schedule(
    env: ManagerBasedRlEnv,
    standing_height: float,
    low_height: float,
    phase_down: float,
    phase_hold: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Track a stand-to-low-to-stand height schedule."""
    asset = _asset(env, asset_cfg)
    phase = _phase(env)
    down = torch.clamp(phase / max(phase_down, 1e-6), 0.0, 1.0)
    up = torch.clamp((phase - phase_hold) / max(1.0 - phase_hold, 1e-6), 0.0, 1.0)
    target = standing_height + (low_height - standing_height) * down
    target = target + (standing_height - low_height) * up
    err = asset.data.root_link_pos_w[:, 2] - target
    return torch.exp(-25.0 * err**2)


def posture_tracking(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ALL_JOINTS
) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    jids = asset_cfg.joint_ids
    err = _wrap_angle(asset.data.joint_pos[:, jids] - asset.data.default_joint_pos[:, jids])
    return torch.exp(-2.5 * torch.mean(err**2, dim=1))


def upright(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """Reward the root body z-axis staying aligned with world up."""
    asset = _asset(env, asset_cfg)
    root_up_dot_world_up = -asset.data.projected_gravity_b[:, 2]
    return torch.exp(4.0 * (root_up_dot_world_up - 1.0))


def velocity_tracking(
    env: ManagerBasedRlEnv,
    target_vx: float,
    target_vy: float = 0.0,
    target_yaw_rate: float = 0.0,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    lin = asset.data.root_link_lin_vel_b
    ang = asset.data.root_link_ang_vel_b
    err = (lin[:, 0] - target_vx) ** 2 + 0.5 * (lin[:, 1] - target_vy) ** 2
    err = err + 0.25 * (ang[:, 2] - target_yaw_rate) ** 2
    return torch.exp(-4.0 * err)


def forward_progress(
    env: ManagerBasedRlEnv,
    target_x: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    err = asset.data.root_link_pos_w[:, 0] - target_x
    return torch.exp(-6.0 * err**2)


def lateral_stability(
    env: ManagerBasedRlEnv,
    max_y: float = 0.08,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    y = asset.data.root_link_pos_w[:, 1]
    return torch.exp(-((y / max(max_y, 1e-6)) ** 2))


def ballistic_jump(
    env: ManagerBasedRlEnv,
    target_height: float,
    target_x: float,
    launch_phase: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Reward upward launch early, then height/forward displacement."""
    asset = _asset(env, asset_cfg)
    phase = _phase(env)
    launch = (phase < launch_phase).float()
    z_vel = asset.data.root_link_lin_vel_w[:, 2]
    height = asset.data.root_link_pos_w[:, 2]
    x = asset.data.root_link_pos_w[:, 0]
    launch_reward = torch.clamp(z_vel, min=0.0) / 3.0
    flight_reward = torch.exp(-8.0 * (height - target_height) ** 2)
    progress = torch.exp(-4.0 * (x - target_x) ** 2)
    return launch * launch_reward + (1.0 - launch) * 0.5 * (flight_reward + progress)


def flip_rotation(
    env: ManagerBasedRlEnv,
    target_pitch_rate: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    pitch_rate = asset.data.root_link_ang_vel_b[:, 1]
    return torch.exp(-0.35 * (pitch_rate - target_pitch_rate) ** 2)


def landing_upright(
    env: ManagerBasedRlEnv,
    landing_phase: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    phase = _phase(env)
    weight = torch.clamp((phase - landing_phase) / max(1.0 - landing_phase, 1e-6), 0.0, 1.0)
    upright = -asset.data.projected_gravity_b[:, 2]
    pitch = _pitch_from_quat_wxyz(asset.data.root_link_quat_w)
    return weight * torch.exp(3.0 * (upright - 1.0)) * torch.exp(-2.0 * pitch**2)


def task_complete(
    env: ManagerBasedRlEnv,
    min_phase: float,
    target_x: float,
    max_x_error: float,
    min_upright: float,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    phase = _phase(env)
    upright = -asset.data.projected_gravity_b[:, 2]
    x_ok = torch.abs(asset.data.root_link_pos_w[:, 0] - target_x) < max_x_error
    return ((phase > min_phase) & x_ok & (upright > min_upright)).float()


def torque_cost(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    normalized = asset.data.actuator_force / _effort_limits(env, asset).clamp_min(1e-6)
    return torch.mean(normalized**2, dim=1)
