"""Actuator-envelope randomization for skill exploration.

This is intentionally simpler than the catalog co-design path: each reset
samples a per-env torque scale and velocity scale around the XML actuator
limits. The policy observes the envelope, while rewards penalize saturation and
torque-speed violations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_STATE_ATTR = "_skill_actuation"


@dataclass(frozen=True)
class ActuatorEnvelopeCfg:
    torque_scale: tuple[float, float]
    velocity_scale: tuple[float, float]
    continuous_torque_fraction: float = 0.45


class _ActuationState:
    def __init__(self, env: ManagerBasedRlEnv, asset: Entity) -> None:
        device = env.device
        act = asset.actuators[0]
        self.ctrl_ids = act.global_ctrl_ids.to(device).long()
        self.joint_cols = act.target_ids.to(device).long()
        self.v_adr = asset.indexing.joint_v_adr[act.target_ids].to(device).long()

        base_effort = [
            max(abs(float(a.forcerange[0])), abs(float(a.forcerange[1])))
            for a in asset.spec.actuators
        ]
        self.base_effort = torch.tensor(base_effort, device=device, dtype=torch.float)

        # A conservative nominal no-load-speed proxy. Replace with datasheet values
        # per joint group once the motor choices are known.
        self.base_velocity = torch.full_like(self.base_effort, 30.0)

        n = env.num_envs
        self.effort_limit = self.base_effort.unsqueeze(0).repeat(n, 1)
        self.velocity_limit = self.base_velocity.unsqueeze(0).repeat(n, 1)
        self.continuous_limit = 0.45 * self.effort_limit
        self.obs = torch.zeros(n, 3, device=device)


def get_state(env: ManagerBasedRlEnv, asset_name: str = "robot") -> _ActuationState:
    state = getattr(env, _STATE_ATTR, None)
    if state is None:
        state = _ActuationState(env, env.scene[asset_name])
        setattr(env, _STATE_ATTR, state)
    return state


@requires_model_fields(
    "actuator_forcerange",
    recompute=RecomputeLevel.set_const_0,
)
def randomize_actuator_envelope(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    cfg: ActuatorEnvelopeCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    state = get_state(env, asset_cfg.name)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(env.device).long()

    n = len(env_ids)
    torque_lo, torque_hi = cfg.torque_scale
    vel_lo, vel_hi = cfg.velocity_scale
    torque_scale = torch.empty(n, 1, device=env.device).uniform_(torque_lo, torque_hi)
    velocity_scale = torch.empty(n, 1, device=env.device).uniform_(vel_lo, vel_hi)

    effort = state.base_effort.unsqueeze(0) * torque_scale
    velocity = state.base_velocity.unsqueeze(0) * velocity_scale
    continuous = effort * cfg.continuous_torque_fraction

    rows = env_ids[:, None]
    model = env.sim.model
    model.actuator_forcerange[rows, state.ctrl_ids, 0] = -effort
    model.actuator_forcerange[rows, state.ctrl_ids, 1] = effort

    state.effort_limit[env_ids] = effort
    state.velocity_limit[env_ids] = velocity
    state.continuous_limit[env_ids] = continuous
    state.obs[env_ids] = torch.cat(
        [
            torque_scale,
            velocity_scale,
            torch.full((n, 1), cfg.continuous_torque_fraction, device=env.device),
        ],
        dim=1,
    )


def actuator_envelope_obs(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return get_state(env, asset_cfg.name).obs


def saturation_cost(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    state = get_state(env, asset_cfg.name)
    asset: Entity = env.scene[asset_cfg.name]
    usage = asset.data.actuator_force[:, state.ctrl_ids].abs() / state.effort_limit.clamp_min(1e-6)
    return torch.mean(torch.relu(usage - 0.90) ** 2, dim=1)


def continuous_torque_cost(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    state = get_state(env, asset_cfg.name)
    asset: Entity = env.scene[asset_cfg.name]
    excess = torch.relu(
        asset.data.actuator_force[:, state.ctrl_ids].abs() - state.continuous_limit
    )
    return torch.mean((excess / state.effort_limit.clamp_min(1e-6)) ** 2, dim=1)


def over_speed_cost(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    state = get_state(env, asset_cfg.name)
    asset: Entity = env.scene[asset_cfg.name]
    speed = asset.data.joint_vel[:, state.joint_cols].abs()
    excess = torch.relu(speed - state.velocity_limit)
    return torch.mean((excess / state.velocity_limit.clamp_min(1e-6)) ** 2, dim=1)


def torque_speed_violation_cost(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Linear torque-speed envelope violation.

    A crude BLDC proxy: available torque decreases linearly to zero at no-load
    speed. This is a soft feasibility penalty; MuJoCo still enforces hard peak
    torque through actuator_forcerange.
    """
    state = get_state(env, asset_cfg.name)
    asset: Entity = env.scene[asset_cfg.name]
    speed_ratio = (
        asset.data.joint_vel[:, state.joint_cols].abs()
        / state.velocity_limit.clamp_min(1e-6)
    )
    available = state.effort_limit * torch.relu(1.0 - speed_ratio)
    torque = asset.data.actuator_force[:, state.ctrl_ids].abs()
    return torch.mean((torch.relu(torque - available) / state.effort_limit.clamp_min(1e-6)) ** 2, dim=1)

