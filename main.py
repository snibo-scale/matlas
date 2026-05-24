"""Load full_assembly 2 URDF with floor and joint motors."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import mujoco.viewer

from components.urdf import hinge_joint_names, load_urdf_spec

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "full_assembly 2"
URDF_PATH = MODEL_DIR / "urdf" / "full_assembly.urdf"
MESH_DIR = MODEL_DIR / "meshes"
PACKAGE_PREFIX = "package://full_assembly/meshes/"
INITIAL_HEIGHT = 1.0
TORQUE_LIMIT = 36.0


def build_model() -> mujoco.MjModel:
    scene = mujoco.MjSpec()
    scene.option.gravity = [0, 0, -9.81]

    # floor = scene.worldbody.add_geom()
    # floor.name = "floor"
    # floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    # floor.size = [2, 2, 0.01]
    # floor.friction = [1.0, 0.005, 0.0001]
    # floor.condim = 3

    base = scene.worldbody.add_body(name="base")
    base.add_freejoint(name="base_free")
    base.pos = [0, 0, INITIAL_HEIGHT]

    robot = load_urdf_spec(URDF_PATH, MESH_DIR, PACKAGE_PREFIX)
    scene.attach(robot, frame=base.add_frame(name="robot_mount"), prefix="")

    for joint_name in hinge_joint_names(robot):
        act = scene.add_actuator(name=f"{joint_name}_motor")
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.target = joint_name
        act.ctrlrange = [-TORQUE_LIMIT, TORQUE_LIMIT]
        act.forcelimited = True
        act.forcerange = [-TORQUE_LIMIT, TORQUE_LIMIT]

    return scene.compile()


def main() -> None:
    model = build_model()
    data = mujoco.MjData(model)
    data.ctrl[:] = 0.0

    if sys.platform == "darwin":
        mujoco.viewer.launch(model, data)
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()


if __name__ == "__main__":
    main()
