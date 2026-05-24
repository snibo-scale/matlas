"""Compose components into a full simulation scene."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco

from components.arm import ArmAssembly
from components.base import Component, Mount
from components.leg import LegAssembly
from components.torso import TorsoAssembly

INITIAL_HEIGHT = 1.0


@dataclass
class SystemBuilder:
    """
    Xacro-like scene composer.

    Example::

        builder = SystemBuilder()
        builder.add_floor()
        builder.add_floating_base()
        builder.add_torso()
        builder.add_legs()
        builder.add_arms()
        model = builder.compile()
    """

    scene: mujoco.MjSpec = field(default_factory=mujoco.MjSpec)
    _joint_targets: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.scene.option.gravity = [0, 0, -9.81]
        self.scene.option.timestep = 0.002

    def add_floor(self) -> None:
        floor = self.scene.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [2, 2, 0.01]
        floor.friction = [1.0, 0.005, 0.0001]
        floor.condim = 3

    def add_floating_base(
        self, name: str = "base", height: float = INITIAL_HEIGHT
    ) -> str:
        base = self.scene.worldbody.add_body(name=name)
        base.add_freejoint(name=f"{name}_free")
        base.pos = [0, 0, height]
        return name

    def mount_component(
        self,
        component: Component,
        mount: Mount,
        prefix: str,
        *,
        add_actuators: bool = True,
    ) -> list[str]:
        joint_targets = component.attach(self.scene, mount, prefix)
        if add_actuators:
            component.add_actuators(
                self.scene, joint_targets, prefix=prefix.rstrip("/")
            )
        self._joint_targets.extend(joint_targets)
        return joint_targets

    def add_torso(self, base_name: str = "base") -> list[str]:
        torso = TorsoAssembly.load()
        mount = Mount(parent=base_name)
        return self.mount_component(torso, mount, prefix="torso/")

    def add_legs(self, base_name: str = "base") -> list[str]:
        joints: list[str] = []
        right = LegAssembly.load()
        right_mount = Mount(parent=base_name, pos=LegAssembly.RIGHT_MOUNT.pos)
        joints.extend(
            self.mount_component(right, right_mount, prefix="right_leg/")
        )
        left = LegAssembly.load()
        left_mount = Mount(
            parent=base_name,
            pos=LegAssembly.LEFT_MOUNT.pos,
            euler=LegAssembly.LEFT_MOUNT.euler,
        )
        joints.extend(
            self.mount_component(left, left_mount, prefix="left_leg/")
        )
        return joints

    def add_arms(self) -> list[str]:
        joints: list[str] = []
        right = ArmAssembly.load()
        joints.extend(
            self.mount_component(right, ArmAssembly.RIGHT_MOUNT, prefix="right_arm/")
        )
        left = ArmAssembly.load()
        joints.extend(
            self.mount_component(left, ArmAssembly.LEFT_MOUNT, prefix="left_arm/")
        )
        return joints

    def compile(self) -> mujoco.MjModel:
        return self.scene.compile()


def build_full_system() -> mujoco.MjModel:
    """Assemble torso, two legs, and two arms on a floating base."""
    builder = SystemBuilder()
    builder.add_floor()
    builder.add_floating_base()
    builder.add_torso()
    builder.add_legs()
    builder.add_arms()
    return builder.compile()
