"""Skill-curriculum environments for Matlas actuation characterization."""

from __future__ import annotations

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
from matlas.tasks.skills import mdp as skill_mdp

CONTROL_DT = 0.02
PHYSICS_DT = 0.002
DECIMATION = round(CONTROL_DT / PHYSICS_DT)
MAX_QVEL = 70.0
_ALL_JOINTS = SceneEntityCfg("robot", joint_names=(".*",))


@dataclass(frozen=True)
class SkillTaskCfg:
    task_id_suffix: str
    episode_seconds: float
    actuator_envelope: skill_mdp.ActuatorEnvelopeCfg
    reset_noise: float
    healthy_height: tuple[float, float]
    max_tilt_rad: float | None
    rewards: dict[str, RewardTermCfg]


def _common_observations() -> dict[str, ObservationGroupCfg]:
    terms = {
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
        "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
        "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "phase": ObservationTermCfg(func=skill_mdp.task_phase),
        "actuator_envelope": ObservationTermCfg(func=skill_mdp.actuator_envelope_obs),
    }
    return {
        "actor": ObservationGroupCfg(terms=terms, concatenate_terms=True),
        "critic": ObservationGroupCfg(terms={**terms}, concatenate_terms=True),
    }


def _events(
    reset_noise: float,
    actuator_envelope: skill_mdp.ActuatorEnvelopeCfg,
) -> dict[str, EventTermCfg]:
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
                "asset_cfg": _ALL_JOINTS,
            },
        ),
        "randomize_actuator_envelope": EventTermCfg(
            func=skill_mdp.randomize_actuator_envelope,
            mode="reset",
            params={
                "cfg": actuator_envelope,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
    }


def _base_rewards() -> dict[str, RewardTermCfg]:
    return {
        "alive": RewardTermCfg(func=mdp.is_alive, weight=0.5),
        "posture": RewardTermCfg(func=skill_mdp.posture_tracking, weight=0.5),
        "lateral_stability": RewardTermCfg(
            func=skill_mdp.lateral_stability,
            weight=0.3,
            params={"max_y": 0.10},
        ),
        "torque_cost": RewardTermCfg(func=skill_mdp.torque_cost, weight=-0.004),
        "saturation_cost": RewardTermCfg(
            func=skill_mdp.saturation_cost, weight=-0.08
        ),
        "continuous_torque_cost": RewardTermCfg(
            func=skill_mdp.continuous_torque_cost, weight=-0.04
        ),
        "over_speed_cost": RewardTermCfg(
            func=skill_mdp.over_speed_cost, weight=-0.10
        ),
        "torque_speed_violation": RewardTermCfg(
            func=skill_mdp.torque_speed_violation_cost, weight=-0.12
        ),
        "joint_speed_cost": RewardTermCfg(func=mdp.joint_vel_l2, weight=-2.0e-4),
        "action_rate_cost": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02),
        "fall": RewardTermCfg(func=mdp.is_terminated, weight=-8.0),
    }


def _task_rewards(*terms: tuple[str, RewardTermCfg]) -> dict[str, RewardTermCfg]:
    rewards = _base_rewards()
    rewards.update(dict(terms))
    return rewards


SKILL_TASKS: dict[str, SkillTaskCfg] = {
    "stand_balance": SkillTaskCfg(
        task_id_suffix="StandBalance",
        episode_seconds=10.0,
        actuator_envelope=skill_mdp.ActuatorEnvelopeCfg(
            torque_scale=(0.8, 1.2),
            velocity_scale=(0.9, 1.1),
        ),
        reset_noise=0.02,
        healthy_height=(0.35, 1.0),
        max_tilt_rad=0.75,
        rewards=_task_rewards(
            (
                "upright",
                RewardTermCfg(func=skill_mdp.upright, weight=1.2),
            ),
            (
                "height",
                RewardTermCfg(
                    func=skill_mdp.height_schedule,
                    weight=1.2,
                    params={
                        "standing_height": STANDING_HEIGHT,
                        "low_height": STANDING_HEIGHT,
                        "phase_down": 0.3,
                        "phase_hold": 0.7,
                    },
                ),
            )
        ),
    ),
    "squat": SkillTaskCfg(
        task_id_suffix="Squat",
        episode_seconds=6.0,
        actuator_envelope=skill_mdp.ActuatorEnvelopeCfg(
            torque_scale=(1.0, 1.4),
            velocity_scale=(0.9, 1.15),
        ),
        reset_noise=0.015,
        healthy_height=(0.25, 1.0),
        max_tilt_rad=0.8,
        rewards=_task_rewards(
            (
                "height_schedule",
                RewardTermCfg(
                    func=skill_mdp.height_schedule,
                    weight=2.0,
                    params={
                        "standing_height": STANDING_HEIGHT,
                        "low_height": 0.43,
                        "phase_down": 0.35,
                        "phase_hold": 0.65,
                    },
                ),
            )
        ),
    ),
    "loaded_squat": SkillTaskCfg(
        task_id_suffix="LoadedSquat",
        episode_seconds=7.0,
        actuator_envelope=skill_mdp.ActuatorEnvelopeCfg(
            torque_scale=(1.0, 1.5),
            velocity_scale=(0.9, 1.15),
        ),
        reset_noise=0.02,
        healthy_height=(0.25, 1.05),
        max_tilt_rad=0.85,
        rewards=_task_rewards(
            (
                "height_schedule",
                RewardTermCfg(
                    func=skill_mdp.height_schedule,
                    weight=2.2,
                    params={
                        "standing_height": STANDING_HEIGHT,
                        "low_height": 0.39,
                        "phase_down": 0.4,
                        "phase_hold": 0.68,
                    },
                ),
            )
        ),
    ),
    "single_step": SkillTaskCfg(
        task_id_suffix="SingleStep",
        episode_seconds=5.0,
        actuator_envelope=skill_mdp.ActuatorEnvelopeCfg(
            torque_scale=(1.0, 1.5),
            velocity_scale=(0.9, 1.2),
        ),
        reset_noise=0.025,
        healthy_height=(0.30, 1.05),
        max_tilt_rad=0.9,
        rewards=_task_rewards(
            (
                "velocity",
                RewardTermCfg(
                    func=skill_mdp.velocity_tracking,
                    weight=1.2,
                    params={"target_vx": 0.22},
                ),
            ),
            (
                "progress",
                RewardTermCfg(
                    func=skill_mdp.forward_progress,
                    weight=1.2,
                    params={"target_x": 0.22},
                ),
            ),
        ),
    ),
    "stair_step": SkillTaskCfg(
        task_id_suffix="StairStep",
        episode_seconds=6.0,
        actuator_envelope=skill_mdp.ActuatorEnvelopeCfg(
            torque_scale=(1.1, 1.6),
            velocity_scale=(0.9, 1.2),
        ),
        reset_noise=0.025,
        healthy_height=(0.35, 1.15),
        max_tilt_rad=0.95,
        rewards=_task_rewards(
            (
                "velocity",
                RewardTermCfg(
                    func=skill_mdp.velocity_tracking,
                    weight=1.0,
                    params={"target_vx": 0.18},
                ),
            ),
            (
                "progress",
                RewardTermCfg(
                    func=skill_mdp.forward_progress,
                    weight=1.0,
                    params={"target_x": 0.18},
                ),
            ),
            (
                "height",
                RewardTermCfg(
                    func=skill_mdp.height_schedule,
                    weight=1.0,
                    params={
                        "standing_height": STANDING_HEIGHT,
                        "low_height": STANDING_HEIGHT + 0.08,
                        "phase_down": 0.45,
                        "phase_hold": 0.65,
                    },
                ),
            ),
        ),
    ),
    "jump_forward": SkillTaskCfg(
        task_id_suffix="JumpForward",
        episode_seconds=4.0,
        actuator_envelope=skill_mdp.ActuatorEnvelopeCfg(
            torque_scale=(1.3, 2.0),
            velocity_scale=(0.95, 1.25),
        ),
        reset_noise=0.03,
        healthy_height=(0.20, 1.55),
        max_tilt_rad=1.15,
        rewards=_task_rewards(
            (
                "jump",
                RewardTermCfg(
                    func=skill_mdp.ballistic_jump,
                    weight=2.2,
                    params={
                        "target_height": STANDING_HEIGHT + 0.22,
                        "target_x": 0.35,
                        "launch_phase": 0.35,
                    },
                ),
            ),
            (
                "complete",
                RewardTermCfg(
                    func=skill_mdp.task_complete,
                    weight=3.0,
                    params={
                        "min_phase": 0.75,
                        "target_x": 0.35,
                        "max_x_error": 0.22,
                        "min_upright": 0.75,
                    },
                ),
            ),
        ),
    ),
    "back_flip": SkillTaskCfg(
        task_id_suffix="BackFlip",
        episode_seconds=4.5,
        actuator_envelope=skill_mdp.ActuatorEnvelopeCfg(
            torque_scale=(1.5, 2.5),
            velocity_scale=(1.0, 1.35),
        ),
        reset_noise=0.03,
        healthy_height=(0.18, 1.75),
        max_tilt_rad=None,
        rewards=_task_rewards(
            (
                "jump",
                RewardTermCfg(
                    func=skill_mdp.ballistic_jump,
                    weight=1.2,
                    params={
                        "target_height": STANDING_HEIGHT + 0.35,
                        "target_x": -0.05,
                        "launch_phase": 0.32,
                    },
                ),
            ),
            (
                "flip_rate",
                RewardTermCfg(
                    func=skill_mdp.flip_rotation,
                    weight=1.8,
                    params={"target_pitch_rate": -7.0},
                ),
            ),
            (
                "landing",
                RewardTermCfg(
                    func=skill_mdp.landing_upright,
                    weight=2.5,
                    params={"landing_phase": 0.68},
                ),
            ),
            (
                "complete",
                RewardTermCfg(
                    func=skill_mdp.task_complete,
                    weight=5.0,
                    params={
                        "min_phase": 0.82,
                        "target_x": -0.05,
                        "max_x_error": 0.30,
                        "min_upright": 0.80,
                    },
                ),
            ),
        ),
    ),
}


def _terminations(task: SkillTaskCfg) -> dict[str, TerminationTermCfg]:
    terms = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "base_height": TerminationTermCfg(
            func=skill_mdp.base_height_out_of_range,
            params={
                "minimum": task.healthy_height[0],
                "maximum": task.healthy_height[1],
            },
        ),
        "dof_velocity": TerminationTermCfg(
            func=skill_mdp.dof_velocity_out_of_limit,
            params={"max_velocity": MAX_QVEL},
        ),
        "nan": TerminationTermCfg(func=mdp.nan_detection, time_out=True),
    }
    if task.max_tilt_rad is not None:
        terms["fell_over"] = TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": task.max_tilt_rad},
        )
    return terms


def make_skill_env_cfg(
    task_name: str,
    play: bool = False,
    actuator_envelope: skill_mdp.ActuatorEnvelopeCfg | None = None,
) -> ManagerBasedRlEnvCfg:
    task = SKILL_TASKS[task_name]
    envelope = actuator_envelope or task.actuator_envelope
    # Policy actions can request the top of the exploration envelope; per-env
    # randomized forcerange clamps each rollout to its sampled actuator limit.
    action_torque_scale = envelope.torque_scale[1]
    actions: dict[str, ActionTermCfg] = {
        "effort": JointEffortActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale={
                joint: action_torque_scale / 0.15 * scale
                for joint, scale in MATLAS_ACTION_SCALE.items()
            },
        )
    }
    env_cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_matlas_robot_cfg()},
            num_envs=1,
            env_spacing=2.0,
        ),
        observations=_common_observations(),
        actions=actions,
        events=_events(task.reset_noise, envelope),
        rewards=task.rewards,
        terminations=_terminations(task),
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
                gravity=(0.0, 0.0, -9.81),
            ),
        ),
        decimation=DECIMATION,
        episode_length_s=task.episode_seconds,
    )
    if play:
        env_cfg.episode_length_s = 1e9
    return env_cfg
