"""Per-env actuator-design events.

``set_actuator_design`` is a reset event that gives each parallel env its own
actuator design. It writes the per-world model fields mjlab expands for domain
randomization (``actuator_forcerange``, ``dof_armature``, ``dof_frictionloss``)
and stashes the design (normalized obs, mass, per-joint velocity limit) on the
env for the observation and reward terms to read.

Design source:
- training: sample a random (motor, gear) per group per env from the catalog;
- evaluation: read ``env._codesign_override`` (a [num_envs, 2*N_GROUPS] long
  tensor of motor/gear indices) injected by the Pareto-search script.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from matlas.codesign import catalog as cat

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_STATE_ATTR = "_codesign"
_OVERRIDE_ATTR = "_codesign_override"


class _CodesignState:
    """Per-env design buffers and the joint/actuator index maps."""

    def __init__(self, env: ManagerBasedRlEnv, asset: Entity) -> None:
        device = env.device
        self.device = device
        act = asset.actuators[0]  # single XmlActuator wrapping all 21 motors
        self.names: list[str] = list(act.target_names)
        self.n_act = len(self.names)
        self.ctrl_ids = act.global_ctrl_ids.to(device).long()
        self.joint_cols = act.target_ids.to(device).long()  # joint_vel column ids
        self.v_adr = asset.indexing.joint_v_adr[act.target_ids].to(device).long()
        # Actuator -> design-group index.
        self.group_idx = torch.tensor(
            [cat.GROUPS.index(cat.group_of(n)) for n in self.names],
            device=device,
            dtype=torch.long,
        )
        # Lookup table [n_motor, n_gear, 5]: effort, vel, armature, friction, mass.
        table = torch.zeros(
            len(cat.MOTOR_CATALOG), len(cat.GEAR_OPTIONS), 5, device=device
        )
        for mi in range(len(cat.MOTOR_CATALOG)):
            for gi, gear in enumerate(cat.GEAR_OPTIONS):
                p = cat.actuator_params(mi, gear)
                table[mi, gi] = torch.tensor(
                    [p.effort_limit, p.velocity_limit, p.armature, p.frictionloss, p.mass],
                    device=device,
                )
        self.table = table

        n = env.num_envs
        self.design_obs = torch.zeros(n, 3 * cat.N_GROUPS, device=device)
        self.mass = torch.zeros(n, device=device)
        # Velocity limit aligned to joint_vel columns (large where unactuated).
        self.vel_limit = torch.full(
            (n, asset.num_joints), float("inf"), device=device
        )
        self.genome = torch.zeros(n, cat.GENOME_LEN, device=device, dtype=torch.long)


def get_state(env: ManagerBasedRlEnv, asset_name: str = "robot") -> _CodesignState:
    state = getattr(env, _STATE_ATTR, None)
    if state is None:
        state = _CodesignState(env, env.scene[asset_name])
        setattr(env, _STATE_ATTR, state)
    return state


def _sample_design_indices(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-env (motor_idx, gear_idx) of shape [len(env_ids), N_GROUPS]."""
    override = getattr(env, _OVERRIDE_ATTR, None)
    g = cat.N_GROUPS
    if override is not None:
        rows = override.to(env.device).long()[env_ids]
        return rows[:, :g], rows[:, g:]
    e = len(env_ids)
    motor_idx = torch.randint(
        0, len(cat.MOTOR_CATALOG), (e, g), device=env.device
    )
    gear_idx = torch.randint(0, len(cat.GEAR_OPTIONS), (e, g), device=env.device)
    return motor_idx, gear_idx


@requires_model_fields(
    "actuator_forcerange",
    "dof_armature",
    "dof_frictionloss",
    recompute=RecomputeLevel.set_const_0,
)
def set_actuator_design(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    state = get_state(env, asset_cfg.name)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(env.device).long()

    motor_idx, gear_idx = _sample_design_indices(env, env_ids)  # [E, G]
    state.genome[env_ids] = torch.cat([motor_idx, gear_idx], dim=1)

    # Per-group params [E, G, 5] -> per-actuator [E, n_act, 5] via group_idx.
    grouped = state.table[motor_idx, gear_idx]  # [E, G, 5]
    per_act = grouped[:, state.group_idx, :]  # [E, n_act, 5]
    effort = per_act[..., 0]
    vel = per_act[..., 1]
    armature = per_act[..., 2]
    friction = per_act[..., 3]
    mass = per_act[..., 4]

    rows = env_ids[:, None]
    model = env.sim.model
    model.actuator_forcerange[rows, state.ctrl_ids, 0] = -effort
    model.actuator_forcerange[rows, state.ctrl_ids, 1] = effort
    model.dof_armature[rows, state.v_adr] = armature
    model.dof_frictionloss[rows, state.v_adr] = friction

    # Buffers for obs / rewards / objectives.
    state.mass[env_ids] = mass.sum(dim=1)
    state.vel_limit[rows, state.joint_cols] = vel
    obs = torch.stack(
        [
            effort / cat.MAX_EFFORT,
            armature / cat.MAX_ARMATURE,
            vel / cat.MAX_VELOCITY,
        ],
        dim=-1,
    )  # [E, n_act, 3]
    # Reduce to per-group (mean over the actuators in each group) -> [E, G, 3].
    group_obs = torch.zeros(len(env_ids), cat.N_GROUPS, 3, device=env.device)
    counts = torch.zeros(cat.N_GROUPS, device=env.device)
    group_obs.index_add_(1, state.group_idx, obs)
    counts.index_add_(0, state.group_idx, torch.ones_like(state.group_idx, dtype=torch.float))
    group_obs /= counts.clamp_min(1.0)[None, :, None]
    state.design_obs[env_ids] = group_obs.reshape(len(env_ids), -1)
