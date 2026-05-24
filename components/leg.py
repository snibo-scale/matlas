"""Leg assembly component."""

from __future__ import annotations

from components.base import Component, Mount

# Offset from the leg component origin to the hip link (from URDF fixed joint).
HIP_OFFSET = (-0.076637, -0.0162221, -0.0319212)

# Pelvis geom position inside the torso component worldbody.
PELVIS_IN_TORSO = (-0.0414369, 0.013591, -0.00683823)


def _mount_pos(hip_in_pelvis: tuple[float, float, float]) -> tuple[float, float, float]:
    hip_in_torso = tuple(
        pelvis + hip for pelvis, hip in zip(PELVIS_IN_TORSO, hip_in_pelvis)
    )
    return tuple(
        hip - offset for hip, offset in zip(hip_in_torso, HIP_OFFSET)
    )


class LegAssembly(Component):
    """Single leg with 6 actuated joints."""

    RIGHT_MOUNT = Mount(
        parent="base",
        pos=_mount_pos((0.03, -0.07, -0.0791376)),
    )
    LEFT_MOUNT = Mount(
        parent="base",
        pos=_mount_pos((-0.098, 0.0, 0.318)),
        euler=(0.0, 0.0, 3.14159),
    )

    @classmethod
    def load(cls) -> LegAssembly:
        return cls.from_package("leg_assembly", actuator_profile="leg")
