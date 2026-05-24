"""Standing balance task for the full Matlas MuJoCo assembly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from envs.walk_env import MatlasWalkEnv, WalkEnvConfig, _quat_to_rotmat, _wrap_angle


@dataclass(frozen=True)
class StandEnvConfig(WalkEnvConfig):
    """Configuration for standing balance training."""

    episode_seconds: float = 10.0
    command_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    controlled_joints: str = "all"
    control_mode: str = "pd"
    gravity_scale: float = 1.0
    action_scale: float = 0.15
    kp: float = 12.0
    kd: float = 1.0
    reset_noise: float = 0.005
    healthy_height: tuple[float, float] = (0.35, 1.0)
    max_tilt_rad: float = 0.75
    max_qvel: float = 40.0
    alive_weight: float = 1.0
    posture_weight: float = 1.2
    stillness_weight: float = 0.5
    angular_stillness_weight: float = 0.5
    height_tracking_weight: float = 0.8
    upright_weight: float = 1.5
    torque_cost: float = 0.003
    action_rate_cost: float = 0.03
    joint_speed_cost: float = 0.002
    xy_drift_weight: float = 0.0
    foot_height_weight: float = 0.0
    fall_penalty: float = 8.0


STAND_STAGE_CONFIGS = {
    1: StandEnvConfig(
        controlled_joints="legs",
        control_mode="direct",
        gravity_scale=0.1,
        reset_noise=0.0,
        episode_seconds=6.0,
        healthy_height=(0.30, 1.0),
        max_tilt_rad=1.2,
    ),
    2: StandEnvConfig(
        gravity_scale=1.0,
        reset_noise=0.002,
        episode_seconds=8.0,
        healthy_height=(0.35, 1.0),
        max_tilt_rad=0.85,
    ),
    3: StandEnvConfig(
        gravity_scale=1.0,
        reset_noise=0.02,
        episode_seconds=10.0,
        healthy_height=(0.35, 1.0),
        max_tilt_rad=0.75,
    ),
}


class MatlasStandEnv(MatlasWalkEnv):
    """Standing task with reward terms for balance and pose holding."""

    config: StandEnvConfig

    def __init__(self, config: StandEnvConfig | None = None, seed: int | None = None):
        super().__init__(config or StandEnvConfig(), seed=seed)
        self.model.opt.gravity[:] = np.asarray([0.0, 0.0, -9.81]) * self.config.gravity_scale
        self.reset_xy = np.zeros(2, dtype=np.float64)
        self.left_foot_body_id = self._body_id("foot_1")
        self.right_foot_body_id = self._body_id("foot")

    def _body_id(self, name: str) -> int:
        import mujoco

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise RuntimeError(f"Required body not found: {name}")
        return body_id

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        self.reset_xy[:] = self.data.qpos[:2]
        return obs, info

    def _reward(
        self, action: np.ndarray, terminated: bool
    ) -> tuple[float, dict[str, float]]:
        rot = _quat_to_rotmat(self.data.qpos[3:7])
        clipped_qvel = np.clip(self.data.qvel, -self.config.max_qvel, self.config.max_qvel)
        base_vel_body = rot.T @ clipped_qvel[0:3]
        base_ang_body = rot.T @ clipped_qvel[3:6]
        upright = rot[2, 2]
        height_error = self.data.qpos[2] - self.default_qpos[2]
        xy_error = self.data.qpos[:2] - self.reset_xy
        foot_height_error = (
            self.data.xpos[self.left_foot_body_id, 2]
            - self.data.xpos[self.right_foot_body_id, 2]
        )
        posture_error = _wrap_angle(
            self.data.qpos[self.controlled_qposadr]
            - self.default_qpos[self.controlled_qposadr]
        )
        normalized_torque = self.last_ctrl / np.maximum(self.torque_limit, 1e-6)

        terms = {
            "alive": 0.0 if terminated else self.config.alive_weight,
            "upright": float(self.config.upright_weight * np.exp(4.0 * (upright - 1.0))),
            "height": float(
                self.config.height_tracking_weight * np.exp(-20.0 * height_error**2)
            ),
            "posture": float(
                self.config.posture_weight * np.exp(-3.0 * np.mean(posture_error**2))
            ),
            "stillness": float(
                self.config.stillness_weight * np.exp(-2.0 * np.mean(base_vel_body**2))
            ),
            "angular_stillness": float(
                self.config.angular_stillness_weight
                * np.exp(-2.0 * np.mean(base_ang_body**2))
            ),
            "torque_cost": float(
                -self.config.torque_cost * np.mean(normalized_torque**2)
            ),
            "action_rate_cost": float(
                -self.config.action_rate_cost * np.mean((action - self.previous_action) ** 2)
            ),
            "joint_speed_cost": float(
                -self.config.joint_speed_cost * np.mean(clipped_qvel[6:] ** 2)
            ),
            "xy_drift": float(
                self.config.xy_drift_weight * np.exp(-20.0 * np.dot(xy_error, xy_error))
            ),
            "foot_height": float(
                self.config.foot_height_weight * np.exp(-80.0 * foot_height_error**2)
            ),
            "fall": float(-self.config.fall_penalty if terminated else 0.0),
        }
        return float(sum(terms.values())), terms

    def _is_healthy(self) -> bool:
        if not super()._is_healthy():
            return False
        if np.max(np.abs(self.data.qvel)) > self.config.max_qvel:
            return False
        return bool(np.isfinite(self.data.qacc).all())
