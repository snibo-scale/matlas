"""Xacro-style reusable component loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import mujoco

from components.actuators import add_joint_actuators
from components.urdf import hinge_joint_names, load_urdf_spec

COMPONENTS_ROOT = Path(__file__).resolve().parent.parent / "models" / "components"
MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"


@dataclass(frozen=True)
class Mount:
    """Attachment frame relative to a parent body (like a xacro origin tag)."""

    parent: str
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Component:
    """
    Loaded URDF sub-assembly with actuators.

    Similar to a xacro macro: call ``load_spec()`` for a fresh copy, attach it
    with ``attach()``, then register motors with ``add_actuators()``.
    """

    name: str
    package_dir: Path
    joint_names: list[str] = field(default_factory=list)
    actuator_profile: str = "default"

    @classmethod
    def from_package(
        cls,
        package_name: str,
        *,
        actuator_profile: str = "default",
        components_root: Path = COMPONENTS_ROOT,
    ) -> Self:
        package_dir = components_root / package_name
        urdf_path = package_dir / "urdf" / f"{package_name}.urdf"
        mesh_dir = package_dir / "meshes"
        package_prefix = f"package://{package_name}/meshes/"

        spec = load_urdf_spec(urdf_path, mesh_dir, package_prefix)
        return cls(
            name=package_name,
            package_dir=package_dir,
            joint_names=hinge_joint_names(spec),
            actuator_profile=actuator_profile,
        )

    @classmethod
    def from_model_dir(
        cls,
        model_dir: Path | str,
        *,
        urdf_name: str | None = None,
        package_prefix: str | None = None,
        actuator_profile: str = "default",
    ) -> Self:
        """Load a URDF from any models/ subdirectory (e.g. ``full_assembly 2``)."""
        model_dir = Path(model_dir)
        robot_name = urdf_name or model_dir.name.replace(" ", "_")
        urdf_path = model_dir / "urdf" / f"{robot_name}.urdf"
        if not urdf_path.exists():
            # Fall back to any single URDF in the urdf folder.
            urdf_files = list((model_dir / "urdf").glob("*.urdf"))
            if len(urdf_files) != 1:
                raise FileNotFoundError(f"No unique URDF found in {model_dir / 'urdf'}")
            urdf_path = urdf_files[0]
            robot_name = urdf_path.stem

        mesh_dir = model_dir / "meshes"
        if package_prefix is None:
            package_prefix = f"package://{robot_name}/meshes/"

        spec = load_urdf_spec(urdf_path, mesh_dir, package_prefix)
        return cls(
            name=robot_name,
            package_dir=model_dir,
            joint_names=hinge_joint_names(spec),
            actuator_profile=actuator_profile,
        )

    def load_spec(self) -> mujoco.MjSpec:
        """Return a new MjSpec instance (required for multiple attachments)."""
        urdf_path = self.package_dir / "urdf" / f"{self.name}.urdf"
        if not urdf_path.exists():
            urdf_files = list((self.package_dir / "urdf").glob("*.urdf"))
            urdf_path = urdf_files[0]
        mesh_dir = self.package_dir / "meshes"
        package_prefix = f"package://{self.name}/meshes/"
        return load_urdf_spec(urdf_path, mesh_dir, package_prefix)

    def attach(
        self,
        scene: mujoco.MjSpec,
        mount: Mount,
        prefix: str,
    ) -> list[str]:
        """
        Attach this component to the scene and return prefixed joint names.

        ``prefix`` should end with ``/`` (e.g. ``"left_leg/"``).
        """
        parent_name = mount.parent.rstrip("/")
        parent_body = scene.body(parent_name)
        if parent_body is None:
            raise ValueError(f"Mount parent body not found: {mount.parent!r}")

        frame = parent_body.add_frame(
            name=f"{prefix.rstrip('/')}_mount",
            pos=list(mount.pos),
            euler=list(mount.euler),
        )
        scene.attach(self.load_spec(), frame=frame, prefix=prefix)
        if prefix and not prefix.endswith("/"):
            prefix = f"{prefix}/"
        return [f"{prefix}{joint_name}" for joint_name in self.joint_names]

    def add_actuators(
        self,
        scene: mujoco.MjSpec,
        joint_targets: list[str],
        *,
        prefix: str = "",
    ) -> None:
        add_joint_actuators(
            scene,
            joint_targets,
            name_prefix=prefix,
            profile=self.actuator_profile,
        )
