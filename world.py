import sys

import mujoco
import mujoco.viewer

from components import build_full_assembly, build_full_system


def build_leg_env() -> mujoco.MjModel:
    """Full robot from models/full_assembly 2."""
    return build_full_assembly()


def build_component_system() -> mujoco.MjModel:
    """Full biped assembled from leg/torso/arm components."""
    return build_full_system()


def main() -> None:
    model = build_leg_env()
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
