"""Named joint poses for the Matlas full assembly."""

from __future__ import annotations

import mujoco
import numpy as np

from components.urdf import flip_mounted_joint_axes, joint_pose_sign


LEG_PITCH_JOINTS = (
    "hip_pitch_1",
    "knee_1",
    "ankle_pitch_1",
    "hip_pitch",
    "knee",
    "ankle_pitch",
)

POSES = {
    "standing": {
        "base_height": 0.61,
        "joints": {
            "hip_pitch_1": -0.15,
            "knee_1": 0.30,
            "ankle_pitch_1": 0.15,
            "hip_pitch": -0.15,
            "knee": 0.30,
            "ankle_pitch": 0.15,
        },
    },
    "crouch": {
        "base_height": 0.56,
        "joints": {
            "hip_pitch_1": -0.45,
            "knee_1": 1.40,
            "ankle_pitch_1": 0.45,
            "hip_pitch": -0.6,
            "knee": 1.40,
            "ankle_pitch": 0.45,
        },
    },
}


def joint_direction(joint_name: str) -> float:
    """Return semantic-to-physical sign for a joint pose."""
    return joint_pose_sign(joint_name)


def semantic_to_physical(joint_name: str, angle: float) -> float:
    return joint_direction(joint_name) * angle


def named_pose_qpos(model: mujoco.MjModel, pose_name: str = "standing") -> np.ndarray:
    """Return a full qpos vector for a named semantic pose."""
    if pose_name not in POSES:
        names = ", ".join(sorted(POSES))
        raise ValueError(f"Unknown pose {pose_name!r}; choose one of: {names}")

    pose = POSES[pose_name]
    qpos = np.zeros(model.nq, dtype=np.float64)
    qpos[2] = pose["base_height"]
    qpos[3] = 1.0
    for joint_name, angle in pose["joints"].items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            raise ValueError(f"Joint not found in model: {joint_name}")
        qpos[model.jnt_qposadr[joint_id]] = angle
    return qpos
