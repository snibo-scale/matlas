"""Visualize named standing/crouching poses in the MuJoCo viewer."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.full_assembly import build_full_assembly
from components.poses import POSES, named_pose_qpos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", choices=sorted(POSES), default="standing")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Step physics instead of showing a static forwarded pose.",
    )
    parser.add_argument(
        "--gravity",
        action="store_true",
        help="Keep gravity enabled while simulating.",
    )
    parser.add_argument(
        "--run-on-open",
        action="store_true",
        help="Start MuJoCo's physics thread immediately instead of opening paused.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="Close after this many seconds; 0 keeps the viewer open.",
    )
    args = parser.parse_args()

    model = build_full_assembly()
    if not args.gravity:
        model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)
    target = named_pose_qpos(model, args.pose)
    data.qpos[:] = target
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    def hold_pose_callback(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        for actuator_id in range(model.nu):
            joint_id = model.actuator_trnid[actuator_id, 0]
            qadr = model.jnt_qposadr[joint_id]
            dadr = model.jnt_dofadr[joint_id]
            err = (target[qadr] - data.qpos[qadr] + np.pi) % (2 * np.pi) - np.pi
            torque = 28.0 * err - 1.2 * data.qvel[dadr]
            low, high = model.actuator_ctrlrange[actuator_id]
            data.ctrl[actuator_id] = np.clip(torque, low, high)

    if sys.platform == "darwin":
        if args.simulate and args.run_on_open:
            mujoco.set_mjcb_control(hold_pose_callback)
            try:
                mujoco.viewer.launch(model, data)
            finally:
                mujoco.set_mjcb_control(None)
        else:
            mujoco.viewer._launch_internal(
                loader=lambda: (model, data),
                run_physics_thread=False,
                show_left_ui=True,
                show_right_ui=True,
            )
        return

    start = time.monotonic()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 135
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        while viewer.is_running():
            if args.simulate:
                hold_pose_callback(model, data)
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)
            viewer.sync()
            if args.seconds > 0 and time.monotonic() - start >= args.seconds:
                break
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
