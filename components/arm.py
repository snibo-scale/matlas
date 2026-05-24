"""Arm assembly component."""

from __future__ import annotations

from components.base import Component, Mount

# Offset from the arm component origin to part_1 (from URDF fixed joint).
PART1_OFFSET = (0.00801965, -0.0943723, 0.0672566)


def _mount_pos(fastened_in_part2: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(
        fastened - offset for fastened, offset in zip(fastened_in_part2, PART1_OFFSET)
    )


class ArmAssembly(Component):
    """Four-DOF arm attached to the torso part_2 body."""

    RIGHT_MOUNT = Mount(
        parent="torso/part_2",
        pos=_mount_pos((0.163045, 0.0301369, 0.164551)),
    )
    LEFT_MOUNT = Mount(
        parent="torso/part_2",
        pos=_mount_pos((-0.0491293, 0.07, -0.0538186)),
        euler=(0.0, 0.0, 3.14159),
    )

    @classmethod
    def load(cls) -> ArmAssembly:
        return cls.from_package("arm_assembly")
