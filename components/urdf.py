"""URDF preprocessing utilities for MuJoCo."""

from __future__ import annotations

import re
from pathlib import Path

import mujoco

# Joints mirrored across the sagittal plane use the ``_1`` suffix in the full
# assembly URDF. Other ``_1`` names (e.g. ``revolute_1``) are not side-specific.
BILATERAL_JOINT_BASES = frozenset(
    {
        "hip_pitch",
        "hip_roll",
        "yaw",
        "knee",
        "ankle_pitch",
        "ankle_roll",
        "shoulder_yaw",
        "shoulder_roll",
        "shoulder_pitch",
        "elbow",
    }
)


def bilateral_joint_base(joint_name: str) -> str | None:
    """Return the shared base name for a bilateral joint, if any."""
    key = joint_name.rsplit("/", 1)[-1]
    if key.endswith("_1"):
        base = key.removesuffix("_1")
        return base if base in BILATERAL_JOINT_BASES else None
    return key if key in BILATERAL_JOINT_BASES else None


def should_flip_joint_axis(joint_name: str, *, mount_prefix: str = "") -> bool:
    """True for right-side copies that need a negated joint axis."""
    if mount_prefix.startswith("right_"):
        return bilateral_joint_base(joint_name) is not None
    return bilateral_joint_base(joint_name) is not None and joint_name.rsplit(
        "/", 1
    )[-1].endswith("_1")


def joint_pose_sign(joint_name: str, *, mount_prefix: str = "") -> float:
    """Map semantic pose angles to joint coordinates after axis conventions."""
    from components.actuators import ACTUATOR_GEAR_SIGNS

    key = joint_name.rsplit("/", 1)[-1]
    base = key.removesuffix("_1")
    sign = ACTUATOR_GEAR_SIGNS.get(base, 1.0)
    if should_flip_joint_axis(joint_name, mount_prefix=mount_prefix):
        sign *= -1.0
    return sign


def _negate_axis_tag(match: re.Match[str]) -> str:
    x, y, z = (float(v) for v in match.group(1).split())
    return f'<axis xyz="{-x:g} {-y:g} {-z:g}" />'


def flip_right_side_joint_axes(text: str) -> str:
    """Negate joint axes for mirrored ``_1`` bilateral joints in URDF text."""

    def process_joint(match: re.Match[str]) -> str:
        joint = match.group(0)
        name_match = re.search(r'name="([^"]+)"', joint)
        if name_match is None or not should_flip_joint_axis(name_match.group(1)):
            return joint
        return re.sub(r'<axis xyz="([^"]+)"\s*/>', _negate_axis_tag, joint)

    return re.sub(r"<joint .+?>.*?</joint>", process_joint, text, flags=re.S)


def flip_mounted_joint_axes(spec: mujoco.MjSpec, *, mount_prefix: str = "") -> None:
    """Negate hinge axes for right-side modular attachments."""
    if not mount_prefix.startswith("right_"):
        return
    for joint in spec.joints:
        if joint.type != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        if bilateral_joint_base(joint.name) is None:
            continue
        joint.axis = [-joint.axis[0], -joint.axis[1], -joint.axis[2]]


def prepare_urdf(text: str, package_prefix: str) -> str:
    """Fix ROS package paths, strip empty root links, and ensure collision geoms."""
    text = text.replace(package_prefix, "")
    text = re.sub(
        r'<link name="root">.*?</link>', '<link name="root" />', text, flags=re.S
    )
    if "<mujoco>" not in text:
        text = re.sub(
            r'(<robot\s+name="[^"]+">)',
            r'\1\n    <mujoco><compiler fusestatic="false"/></mujoco>',
            text,
            count=1,
        )

    def add_collision_from_visual(match: re.Match[str]) -> str:
        visual = match.group(0)
        collision = visual.replace("<visual>", "<collision>").replace(
            "</visual>", "</collision>"
        )
        return f"{visual}\n        {collision}"

    def process_link(match: re.Match[str]) -> str:
        link = match.group(0)
        if "<collision" in link:
            return link
        return re.sub(
            r"<visual>.*?</visual>", add_collision_from_visual, link, flags=re.S
        )

    text = re.sub(r"<link .+?>.*?</link>", process_link, text, flags=re.S)
    return flip_right_side_joint_axes(text)


def load_urdf_spec(
    urdf_path: Path,
    mesh_dir: Path,
    package_prefix: str,
) -> mujoco.MjSpec:
    """Load a URDF into an MjSpec using link inertials and collision geometry."""
    text = prepare_urdf(urdf_path.read_text(), package_prefix)
    meshdir = str(mesh_dir.resolve())

    def make_spec() -> mujoco.MjSpec:
        spec = mujoco.MjSpec.from_string(text)
        spec.compiler.meshdir = meshdir
        spec.compiler.inertiafromgeom = 0
        return spec

    spec = make_spec()
    try:
        spec.compile()
        # compile() mutates the spec; re-parse so attach/compile preserves kinematics
        spec = make_spec()
    except mujoco.FatalError:
        for mesh in spec.meshes:
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    return spec


def hinge_joint_names(spec: mujoco.MjSpec) -> list[str]:
    return [
        joint.name
        for joint in spec.joints
        if joint.type == int(mujoco.mjtJoint.mjJNT_HINGE)
    ]
