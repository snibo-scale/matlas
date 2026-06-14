"""Standing-balance task configuration for the Matlas biped (mjlab).

Ports ``envs/stand_env.py`` to mjlab's manager-based env. The policy commands
joint torques (``JointEffortActionCfg``) scaled to 0.15x the actuator limits,
matching the legacy ``control_mode="direct"``. The 3-stage curriculum
(``STAND_STAGE_CONFIGS``) is preserved as three registered task variants that
differ in gravity, reset noise, episode length, and termination tolerances.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as mdp
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from matlas.robots.matlas_constants import (
    MATLAS_ACTION_SCALE,
    STANDING_HEIGHT,
    get_matlas_robot_cfg,
)
from matlas.tasks.stand import mdp as stand_mdp

CONTROL_DT = 0.02
PHYSICS_DT = 0.002
DECIMATION = round(CONTROL_DT / PHYSICS_DT)  # 10
MAX_QVEL = 40.0


@dataclass(frozen=True)
class StandStageCfg:
    """Per-stage curriculum knobs, mirroring STAND_STAGE_CONFIGS."""

    gravity_scale: float
    reset_noise: float
    episode_seconds: float
    healthy_height: tuple[float, float]
    max_tilt_rad: float


STAND_STAGES: dict[int, StandStageCfg] = {
    1: StandStageCfg(0.1, 0.0, 6.0, (0.30, 1.0), 1.2),
    2: StandStageCfg(1.0, 0.002, 8.0, (0.35, 1.0), 0.85),
    3: StandStageCfg(1.0, 0.02, 10.0, (0.35, 1.0), 0.75),
}


def _observations() -> dict[str, ObservationGroupCfg]:
    terms = {
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
        "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
        "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }
    return {
        "actor": ObservationGroupCfg(terms=terms, concatenate_terms=True),
        "critic": ObservationGroupCfg(terms={**terms}, concatenate_terms=True),
    }


def _rewards() -> dict[str, RewardTermCfg]:
    return {
        "alive": RewardTermCfg(func=mdp.is_alive, weight=1.0),
        "upright": RewardTermCfg(func=stand_mdp.upright, weight=1.5),
        "height": RewardTermCfg(
            func=stand_mdp.height_tracking,
            weight=0.8,
            params={"target_height": STANDING_HEIGHT},
        ),
        "posture": RewardTermCfg(func=stand_mdp.posture, weight=1.2),
        "stillness": RewardTermCfg(func=stand_mdp.stillness, weight=0.5),
        "angular_stillness": RewardTermCfg(
            func=stand_mdp.angular_stillness, weight=0.5
        ),
        "torque_cost": RewardTermCfg(func=stand_mdp.torque_cost, weight=-0.003),
        "joint_speed_cost": RewardTermCfg(
            func=stand_mdp.joint_speed_cost, weight=-0.002
        ),
        "action_rate_cost": RewardTermCfg(
            func=stand_mdp.action_rate_cost, weight=-0.03
        ),
        "fall": RewardTermCfg(func=mdp.is_terminated, weight=-8.0),
    }


def _events(reset_noise: float) -> dict[str, EventTermCfg]:
    noise = (-reset_noise, reset_noise)
    return {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {axis: noise for axis in ("x", "y", "z")},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": noise,
                "velocity_range": noise,
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }


def _terminations(stage: StandStageCfg) -> dict[str, TerminationTermCfg]:
    return {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": stage.max_tilt_rad},
        ),
        "base_height": TerminationTermCfg(
            func=stand_mdp.base_height_out_of_range,
            params={
                "minimum": stage.healthy_height[0],
                "maximum": stage.healthy_height[1],
            },
        ),
        "dof_velocity": TerminationTermCfg(
            func=stand_mdp.dof_velocity_out_of_limit,
            params={"max_velocity": MAX_QVEL},
        ),
        "nan": TerminationTermCfg(func=mdp.nan_detection, time_out=True),
    }


def make_stand_env_cfg(stage: int = 3, play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = STAND_STAGES[stage]
    actions: dict[str, ActionTermCfg] = {
        "effort": JointEffortActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=dict(MATLAS_ACTION_SCALE),
        )
    }
    env_cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_matlas_robot_cfg()},
            num_envs=1,
            env_spacing=2.0,
        ),
        observations=_observations(),
        actions=actions,
        events=_events(cfg.reset_noise),
        rewards=_rewards(),
        terminations=_terminations(cfg),
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="base",
            distance=3.0,
            elevation=-10.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=PHYSICS_DT,
                gravity=(0.0, 0.0, -9.81 * cfg.gravity_scale),
            ),
        ),
        decimation=DECIMATION,
        episode_length_s=cfg.episode_seconds,
    )
    if play:
        env_cfg.episode_length_s = 1e9
    return env_cfg
