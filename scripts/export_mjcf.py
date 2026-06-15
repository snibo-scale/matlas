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
import trimesh

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
COLLISION_SUFFIX = "_collision"


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


def _write_simplified_collision_meshes(asset_dir: Path) -> dict[str, str]:
    """Create convex-hull STL collision meshes for every visual mesh.

    The collision meshes preserve each STL's local coordinate frame, so collision
    geoms can reuse the same pos/quat as their visual mesh counterparts.
    """
    collision_files: dict[str, str] = {}
    for src in sorted(asset_dir.glob("*.stl")):
        if src.stem.endswith(COLLISION_SUFFIX):
            continue
        mesh = trimesh.load_mesh(src, force="mesh", process=True)
        hull = mesh.convex_hull
        out_name = f"{src.stem}{COLLISION_SUFFIX}.stl"
        hull.export(asset_dir / out_name)
        collision_files[src.stem] = out_name
    return collision_files


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


def _add_collision_mesh_assets(xml: str, collision_files: dict[str, str]) -> str:
    """Add simplified collision mesh assets next to visual mesh assets."""
    for mesh_name, file_name in collision_files.items():
        collision_name = f"{mesh_name}{COLLISION_SUFFIX}"
        if f'<mesh name="{collision_name}"' in xml:
            continue
        pattern = rf'(\n\s*<mesh name="{re.escape(mesh_name)}" file="[^"]+"/>)'
        replacement = rf'\1\n    <mesh name="{collision_name}" file="{file_name}"/>'
        xml = re.sub(pattern, replacement, xml, count=1)
    return xml


def _split_visual_and_collision_mesh_geoms(xml: str) -> str:
    """Keep original STL meshes visual-only and add simplified STL collisions.

    ``contype=0, conaffinity=1`` lets these robot geoms collide with the default
    terrain (which has contype=1) but not with each other, avoiding robot
    self-collision contact storms.
    """
    lines: list[str] = []
    for line in xml.splitlines():
        if "<geom " in line and ' type="mesh"' in line:
            visual = line.replace("/>", ' contype="0" conaffinity="0" group="2"/>')
            collision = re.sub(
                r'mesh="([^"]+)"',
                rf'mesh="\1{COLLISION_SUFFIX}"',
                line,
            )
            collision = re.sub(r'\srgba="[^"]+"', "", collision)
            collision = collision.replace("/>", ' rgba="0.1 0.8 0.1 0.22" contype="0" conaffinity="1" group="3" condim="3" friction="1.0 0.005 0.0001"/>')
            lines.append(visual)
            lines.append(collision)
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def export() -> Path:
    spec = build_robot_spec()
    _copy_meshes(spec, ASSET_DIR)
    collision_files = _write_simplified_collision_meshes(ASSET_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xml = _clean_xml(spec.to_xml())
    xml = _ensure_size_limits(xml)
    xml = _add_collision_mesh_assets(xml, collision_files)
    xml = _split_visual_and_collision_mesh_geoms(xml)
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
