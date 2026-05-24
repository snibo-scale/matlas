"""Joint motor definitions."""

from __future__ import annotations

import mujoco

LEG_TORQUE_LIMITS = {
    "hip_pitch": 36.0,
    "hip_roll": 36.0,
    "yaw": 36.0,
    "knee": 36.0,
    "ankle_pitch": 12.0,
    "ankle_roll": 12.0,
}

DEFAULT_TORQUE_LIMIT = 36.0


def torque_limit(joint_name: str, profile: str = "default") -> float:
    if profile == "leg":
        base = joint_name.removesuffix("_1")
        return LEG_TORQUE_LIMITS.get(base, DEFAULT_TORQUE_LIMIT)
    return DEFAULT_TORQUE_LIMIT


def add_joint_actuators(
    scene: mujoco.MjSpec,
    joint_targets: list[str],
    *,
    name_prefix: str = "",
    profile: str = "default",
) -> None:
    """Add torque motors targeting already-prefixed joint names."""
    for target in joint_targets:
        joint_key = target.rsplit("/", 1)[-1]
        limit = torque_limit(joint_key, profile)
        actuator_name = (
            f"{name_prefix}_{joint_key}_motor" if name_prefix else f"{joint_key}_motor"
        )
        act = scene.add_actuator(name=actuator_name)
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.target = target
        act.ctrlrange = [-limit, limit]
        act.forcelimited = True
        act.forcerange = [-limit, limit]
