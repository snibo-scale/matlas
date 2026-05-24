"""Load full_assembly MJCF with a floor and lights."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import mujoco.viewer

MJCF_PATH = Path(
    "/Users/sidney.nimako/Documents/Codex/2026-05-23/i-want-to-build-a-low/"
    "full_assembly_model/mjcf/full_assembly.xml"
)


def build_model() -> mujoco.MjModel:
    scene = mujoco.MjSpec()
    scene.option.gravity = [0, 0, -9.81]
    scene.option.timestep = 0.002

    floor = scene.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [4, 4, 0.01]
    floor.friction = [1.0, 0.005, 0.0001]
    floor.condim = 3

    key = scene.worldbody.add_light(name="key")
    key.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    key.pos = [2, -2, 3]
    key.dir = [-1, 1, -1.5]
    key.castshadow = True

    fill = scene.worldbody.add_light(name="fill")
    fill.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    fill.pos = [-2, -1, 2]
    fill.dir = [1, 0.5, -1]

    robot = mujoco.MjSpec.from_file(str(MJCF_PATH))
    scene.attach(robot, frame=scene.worldbody.add_frame(name="robot_mount"), prefix="")

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
