"""Export the assembled Matlas biped to a single self-contained MJCF.

The runtime model used to be assembled from URDF parts at import time, with
torque motors added in Python (``components/actuators.py``). mjlab consumes a
single MJCF per Entity, so this script bakes the assembly down to a committed
``models/matlas/matlas.xml`` whose ``<actuator>`` block carries the motors
(36/12 N*m ranges, knee/elbow gear -1) defined by ``add_joint_actuators``.

The exported model keeps the floating-base free joint but omits the floor:
mjlab's scene supplies the terrain. Run from the repo root::

    uv run python scripts/export_mjcf.py
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.base import MODELS_ROOT, Mount
from components.full_assembly import FullAssembly
from components.system import INITIAL_HEIGHT, SystemBuilder

OUT_DIR = ROOT / "models" / "matlas"
OUT_XML = OUT_DIR / "matlas.xml"
ASSET_DIR = OUT_DIR / "assets"
NCONMAX = 4096
NJMAX = 20000


def build_robot_spec() -> mujoco.MjSpec:
    """Assemble the full robot (no floor) with torque motors in the spec."""
    builder = SystemBuilder()
    builder.add_floating_base(height=INITIAL_HEIGHT)
    robot = FullAssembly.load()
    builder.mount_component(robot, Mount(parent="base"), prefix="")
    spec = builder.scene
    spec.compile()  # validates kinematics and resolves meshes
    return spec


def _build_mesh_index() -> dict[str, Path]:
    """Map mesh basename -> source path across every models/ mesh folder."""
    index: dict[str, Path] = {}
    for ext in ("*.stl", "*.STL", "*.obj", "*.OBJ"):
        for path in MODELS_ROOT.rglob(ext):
            # Don't index our own export target.
            if ASSET_DIR in path.parents:
                continue
            index.setdefault(path.name, path)
    return index


def _copy_meshes(spec: mujoco.MjSpec, asset_dir: Path) -> None:
    """Copy every referenced mesh into ``asset_dir`` and flatten file refs."""
    asset_dir.mkdir(parents=True, exist_ok=True)
    meshdir = spec.compiler.meshdir or ""
    index = _build_mesh_index()
    for mesh in spec.meshes:
        if not mesh.file:
            continue
        src = Path(mesh.file)
        if not src.is_absolute():
            candidate = Path(meshdir) / src
            src = candidate if candidate.exists() else index.get(src.name, candidate)
        if not src.exists():
            raise FileNotFoundError(f"Mesh file not found for export: {mesh.file}")
        shutil.copyfile(src, asset_dir / src.name)
        mesh.file = src.name  # reference relative to the new meshdir
    spec.compiler.meshdir = asset_dir.name


def _clean_xml(xml: str) -> str:
    """Drop empty default classes that MjSpec.to_xml emits from the URDF import.

    The URDF loader injects ``<mujoco><compiler/></mujoco>``, which round-trips
    to an empty ``<default/>`` MuJoCo rejects on reload ("empty class name").
    """
    xml = re.sub(r"\n\s*<default/>", "", xml)
    xml = re.sub(r"\n\s*<default>\s*</default>", "", xml)
    return xml


def _ensure_size_limits(xml: str) -> str:
    """Give contact-rich rollouts enough constraint storage.

    MuJoCo's default dynamic sizing can still hit ``nefc overflow`` in batched
    high-contact training. Keep this as an export-time patch because MjSpec does
    not expose the <size> block in the Python API.
    """
    size = f'  <size nconmax="{NCONMAX}" njmax="{NJMAX}"/>'
    if re.search(r"<size\b", xml):
        xml = re.sub(r"\n\s*<size\b[^>]*/>", f"\n{size}", xml, count=1)
    else:
        xml = re.sub(r"(\n\s*<compiler\b[^>]*/>)", rf"\1\n{size}", xml, count=1)
    return xml


def export() -> Path:
    spec = build_robot_spec()
    _copy_meshes(spec, ASSET_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xml = _ensure_size_limits(_clean_xml(spec.to_xml()))
    OUT_XML.write_text(xml)

    # Round-trip to confirm the committed file is loadable on its own.
    model = mujoco.MjModel.from_xml_path(str(OUT_XML))
    motors = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
    print(f"Wrote {OUT_XML} ({len(xml)} bytes)")
    print(f"  bodies={model.nbody} dofs={model.nv} actuators={model.nu}")
    print(f"  meshes copied to {ASSET_DIR}")
    print(f"  motors: {', '.join(motors)}")
    return OUT_XML


if __name__ == "__main__":
    export()
