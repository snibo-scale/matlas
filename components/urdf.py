"""URDF preprocessing utilities for MuJoCo."""

from __future__ import annotations

import re
from pathlib import Path

import mujoco


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

    return re.sub(r"<link .+?>.*?</link>", process_link, text, flags=re.S)


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
