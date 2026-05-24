"""Complete biped URDF loader (models/full_assembly 2)."""

from __future__ import annotations

from pathlib import Path

import mujoco

from components.base import MODELS_ROOT, Component, Mount
from components.system import INITIAL_HEIGHT, SystemBuilder

FULL_ASSEMBLY_DIR = MODELS_ROOT / "full_assembly 3"
# URDF mesh tags use this ROS package name, not the folder name.
PACKAGE_PREFIX = "package://full_assembly/meshes/"


class FullAssembly(Component):
    """Full robot with two legs, torso, and two arms."""

    @classmethod
    def load(cls) -> FullAssembly:
        return cls.from_model_dir(
            FULL_ASSEMBLY_DIR,
            urdf_name="full_assembly",
            package_prefix=PACKAGE_PREFIX,
            actuator_profile="leg",
        )


def build_full_assembly(height: float = INITIAL_HEIGHT) -> mujoco.MjModel:
    """Load full_assembly 2 on a floating base with actuators."""
    builder = SystemBuilder()
    builder.add_floor()
    builder.add_floating_base(height=height)
    robot = FullAssembly.load()
    builder.mount_component(robot, Mount(parent="base"), prefix="")
    return builder.compile()
