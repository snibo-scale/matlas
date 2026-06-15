"""Audit the compiled MuJoCo collision setup."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matlas.robots.matlas_constants import MATLAS_XML


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MATLAS_XML))
    robot_geoms = [
        geom_id
        for geom_id in range(model.ngeom)
        if mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id]
        )
        != "world"
    ]
    collision_geoms = [
        geom_id
        for geom_id in robot_geoms
        if model.geom_contype[geom_id] != 0 and model.geom_conaffinity[geom_id] != 0
    ]
    mesh_collision_geoms = [
        geom_id
        for geom_id in collision_geoms
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
    ]
    disabled_contact_geoms = [
        geom_id
        for geom_id in robot_geoms
        if model.geom_contype[geom_id] == 0 or model.geom_conaffinity[geom_id] == 0
    ]

    print(f"robot_geoms={len(robot_geoms)}")
    print(f"active_collision_geoms={len(collision_geoms)}")
    print(f"mesh_collision_geoms={len(mesh_collision_geoms)}")
    print(f"disabled_contact_geoms={len(disabled_contact_geoms)}")
    print(f"explicit_collision_pairs={model.npair}")
    print(f"explicit_excludes={model.nexclude}")
    print(f"contact_disableflag={bool(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT)}")

    if mesh_collision_geoms:
        raise SystemExit("Some active collision geoms are still mesh geoms.")
    if len(collision_geoms) != 2:
        raise SystemExit("Expected two primitive foot collision geoms.")
    if model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT:
        raise SystemExit("Global contacts are disabled.")

    print("collision_audit=ok")


if __name__ == "__main__":
    main()
