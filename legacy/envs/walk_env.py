"""Walking task scaffold for the full Matlas MuJoCo assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from components.full_assembly import build_full_assembly
from components.poses import named_pose_qpos


try:
    from gymnasium import Env, spaces
except ModuleNotFoundError:
    Env = object
    spaces = None


LEG_JOINT_KEYS = (
    "hip_pitch",
    "hip_roll",
    "yaw",
    "knee",
    "ankle_pitch",
    "ankle_roll",
)


@dataclass(frozen=True)
class WalkEnvConfig:
    """Parameters that define the walking task and actuator wrapper."""

    nominal_pose: str = "standing"
    control_dt: float = 0.02
    episode_seconds: float = 8.0
    command_velocity: tuple[float, float, float] = (0.25, 0.0, 0.0)
    controlled_joints: str = "legs"
    control_mode: str = "pd"
    action_scale: float = 0.45
    kp: float = 28.0
    kd: float = 1.2
    reset_noise: float = 0.01
    healthy_height: tuple[float, float] = (0.45, 1.25)
    max_tilt_rad: float = 0.9
    forward_weight: float = 2.0
    lateral_weight: float = 0.5
    yaw_weight: float = 0.25
    upright_weight: float = 0.8
    height_weight: float = 0.2
    torque_cost: float = 0.002
    action_rate_cost: float = 0.02
    joint_speed_cost: float = 0.001
    fall_penalty: float = 5.0


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Return a 3x3 world-from-body rotation matrix for a MuJoCo quaternion."""
    mat = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, q)
    return mat.reshape(3, 3)


def _wrap_angle(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


class MatlasWalkEnv(Env):
    """
    Minimal Gymnasium-compatible walking environment.

    The policy outputs normalized target offsets for the 12 leg joints. A
    software PD controller converts those targets to torque commands and clips
    them by the MuJoCo actuator ranges already defined in the robot model.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: WalkEnvConfig | None = None, seed: int | None = None):
        self.config = config or WalkEnvConfig()
        self.model = build_full_assembly()
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.frame_skip = max(1, round(self.config.control_dt / self.model.opt.timestep))
        self.max_steps = max(1, round(self.config.episode_seconds / self.config.control_dt))
        self.step_count = 0

        self.controlled_actuator_ids = self._find_controlled_actuators()
        self.controlled_joint_ids = self.model.actuator_trnid[
            self.controlled_actuator_ids, 0
        ]
        self.controlled_qposadr = self.model.jnt_qposadr[self.controlled_joint_ids]
        self.controlled_dofadr = self.model.jnt_dofadr[self.controlled_joint_ids]
        self.controlled_actuator_gear = self.model.actuator_gear[
            self.controlled_actuator_ids, 0
        ]
        self.controlled_joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id))
            for joint_id in self.controlled_joint_ids
        ]
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self.torque_limit = np.maximum(np.abs(self.ctrl_low), np.abs(self.ctrl_high))
        self.default_qpos = self._nominal_qpos()
        self.previous_action = np.zeros(
            len(self.controlled_actuator_ids), dtype=np.float64
        )
        self.last_ctrl = np.zeros(self.model.nu, dtype=np.float64)

        obs_size = self._get_obs().shape[0]
        if spaces is not None:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(len(self.controlled_actuator_ids),),
                dtype=np.float32,
            )
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_size,),
                dtype=np.float32,
            )

    @property
    def leg_actuator_ids(self) -> np.ndarray:
        """Backward-compatible alias for scripts that report leg metrics."""
        return self._find_leg_actuators()

    def _find_leg_actuators(self) -> np.ndarray:
        actuator_ids: list[int] = []
        for actuator_id in range(self.model.nu):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
            )
            if name is None:
                continue
            joint_name = name.removesuffix("_motor")
            base_name = joint_name.removesuffix("_1")
            if base_name in LEG_JOINT_KEYS:
                actuator_ids.append(actuator_id)
        if len(actuator_ids) != 12:
            raise RuntimeError(f"Expected 12 leg actuators, found {len(actuator_ids)}")
        return np.asarray(actuator_ids, dtype=np.int32)

    def _find_controlled_actuators(self) -> np.ndarray:
        if self.config.controlled_joints == "legs":
            return self._find_leg_actuators()
        if self.config.controlled_joints == "all":
            return np.arange(self.model.nu, dtype=np.int32)
        raise ValueError(
            f"Unknown controlled_joints={self.config.controlled_joints!r}; "
            "choose 'legs' or 'all'"
        )

    def _nominal_qpos(self) -> np.ndarray:
        return named_pose_qpos(self.model, self.config.nominal_pose)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.default_qpos
        self.data.qpos[self.controlled_qposadr] += self.rng.normal(
            0.0, self.config.reset_noise, size=len(self.controlled_qposadr)
        )
        self.data.qvel[:] = self.rng.normal(
            0.0, self.config.reset_noise, size=self.model.nv
        )
        self.previous_action.fill(0.0)
        self.last_ctrl.fill(0.0)
        self.step_count = 0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), self._info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)
        ctrl = self._pd_control(action)
        self.data.ctrl[:] = ctrl
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.last_ctrl[:] = ctrl
        self.step_count += 1

        obs = self._get_obs()
        terminated = not self._is_healthy()
        truncated = self.step_count >= self.max_steps
        reward, reward_terms = self._reward(action, terminated)
        info = self._info()
        info["reward_terms"] = reward_terms
        self.previous_action[:] = action
        return obs, reward, terminated, truncated, info

    def _pd_control(self, action: np.ndarray) -> np.ndarray:
        if self.config.control_mode == "direct":
            ctrl = np.zeros(self.model.nu, dtype=np.float64)
            limit = np.maximum(
                np.abs(self.ctrl_low[self.controlled_actuator_ids]),
                np.abs(self.ctrl_high[self.controlled_actuator_ids]),
            )
            # action_scale bounds how much torque a unit-magnitude action can
            # command. Without it, |action|=1 -> full actuator limit, which
            # destabilizes the robot under high-entropy initial policies (SAC).
            ctrl[self.controlled_actuator_ids] = self.config.action_scale * limit * action
            return np.clip(ctrl, self.ctrl_low, self.ctrl_high)
        if self.config.control_mode != "pd":
            raise ValueError(
                f"Unknown control_mode={self.config.control_mode!r}; "
                "choose 'pd' or 'direct'"
            )

        q = _wrap_angle(self.data.qpos[self.controlled_qposadr])
        qd = self.data.qvel[self.controlled_dofadr]
        q_des = (
            self.default_qpos[self.controlled_qposadr]
            + self.config.action_scale * self.controlled_actuator_gear * action
        )
        joint_torque = self.config.kp * _wrap_angle(q_des - q) - self.config.kd * qd
        actuator_ctrl = self.controlled_actuator_gear * joint_torque

        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        ctrl[self.controlled_actuator_ids] = actuator_ctrl
        return np.clip(ctrl, self.ctrl_low, self.ctrl_high)

    def _get_obs(self) -> np.ndarray:
        rot = _quat_to_rotmat(self.data.qpos[3:7])
        gravity_body = rot.T @ np.array([0.0, 0.0, -1.0])
        command = np.asarray(self.config.command_velocity, dtype=np.float64)
        joint_q = _wrap_angle(self.data.qpos[7:])
        joint_qd = np.clip(self.data.qvel[6:], -30.0, 30.0)
        base_vel_body = rot.T @ self.data.qvel[0:3]
        base_ang_body = rot.T @ self.data.qvel[3:6]
        obs = np.concatenate(
            [
                gravity_body,
                command,
                base_vel_body,
                base_ang_body,
                joint_q,
                joint_qd,
                self.previous_action,
            ]
        )
        return obs.astype(np.float32)

    def _reward(
        self, action: np.ndarray, terminated: bool
    ) -> tuple[float, dict[str, float]]:
        rot = _quat_to_rotmat(self.data.qpos[3:7])
        base_vel_body = rot.T @ self.data.qvel[0:3]
        base_ang_body = rot.T @ self.data.qvel[3:6]
        cmd = np.asarray(self.config.command_velocity)
        velocity_error = base_vel_body[:2] - cmd[:2]
        yaw_error = base_ang_body[2] - cmd[2]
        upright = rot[2, 2]
        height = self.data.qpos[2]
        height_target = 0.85
        normalized_torque = self.last_ctrl / np.maximum(self.torque_limit, 1e-6)
        terms = {
            "velocity": float(
                self.config.forward_weight * np.exp(-4.0 * velocity_error[0] ** 2)
                + self.config.lateral_weight * np.exp(-4.0 * velocity_error[1] ** 2)
            ),
            "yaw": float(self.config.yaw_weight * np.exp(-2.0 * yaw_error**2)),
            "upright": float(self.config.upright_weight * upright),
            "height": float(
                self.config.height_weight * np.exp(-8.0 * (height - height_target) ** 2)
            ),
            "torque_cost": float(
                -self.config.torque_cost * np.mean(normalized_torque**2)
            ),
            "action_rate_cost": float(
                -self.config.action_rate_cost * np.mean((action - self.previous_action) ** 2)
            ),
            "joint_speed_cost": float(
                -self.config.joint_speed_cost * np.mean(self.data.qvel[6:] ** 2)
            ),
            "fall": float(-self.config.fall_penalty if terminated else 0.0),
        }
        return float(sum(terms.values())), terms

    def _is_healthy(self) -> bool:
        height_min, height_max = self.config.healthy_height
        height = self.data.qpos[2]
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            return False
        if height < height_min or height > height_max:
            return False
        rot = _quat_to_rotmat(self.data.qpos[3:7])
        return bool(rot[2, 2] > np.cos(self.config.max_tilt_rad))

    def _info(self) -> dict[str, Any]:
        torque = self.last_ctrl.copy()
        dof_vel = self.data.qvel[self.model.jnt_dofadr[self.model.actuator_trnid[:, 0]]]
        limit = np.maximum(self.torque_limit, 1e-6)
        return {
            "base_height": float(self.data.qpos[2]),
            "base_x": float(self.data.qpos[0]),
            "mean_abs_torque": float(np.mean(np.abs(torque))),
            "mean_mechanical_power": float(np.mean(np.abs(torque * dof_vel))),
            "saturation_fraction": float(np.mean(np.abs(torque) >= 0.98 * limit)),
            "leg_saturation_fraction": float(
                np.mean(np.abs(torque[self.leg_actuator_ids]) >= 0.98 * limit[self.leg_actuator_ids])
            ),
        }
