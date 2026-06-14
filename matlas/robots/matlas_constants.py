"""Matlas biped entity configuration for mjlab.

The robot is the single MJCF exported by ``scripts/export_mjcf.py`` from the
URDF assembly. Its ``<actuator>`` block holds 21 torque (effort) motors with
36/12 N*m limits and gear -1 on knee/elbow. ``XmlActuatorCfg`` wraps those
existing actuators, so the policy commands joint torques directly -- the mjlab
equivalent of the legacy stand env's ``control_mode="direct"``.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

MATLAS_XML: Path = Path(__file__).resolve().parents[2] / "models" / "matlas" / "matlas.xml"
assert MATLAS_XML.exists(), (
    f"Missing {MATLAS_XML}. Run `uv run python scripts/export_mjcf.py` first."
)

# Maps |action|=1 to this fraction of each actuator's torque limit, matching the
# legacy direct-control scaling (ctrl = action_scale * limit * action).
ACTION_SCALE_FRACTION = 0.15

# Nominal standing pose (physical joint coordinates), from the legacy
# ``envs/poses.py`` "standing" pose. Knee/elbow signs are folded into the
# exported actuator gear, so these joint angles transfer verbatim.
STANDING_HEIGHT = 0.61
STANDING_JOINTS: dict[str, float] = {
    "hip_pitch_1": -0.15,
    "knee_1": 0.30,
    "ankle_pitch_1": 0.15,
    "hip_pitch": -0.15,
    "knee": 0.30,
    "ankle_pitch": 0.15,
}


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MATLAS_XML))


def _build_action_scale() -> dict[str, float]:
    """Per-joint torque scale (0.15 * |forcerange|) keyed by joint name."""
    spec = get_spec()
    scale: dict[str, float] = {}
    for act in spec.actuators:
        limit = max(abs(float(act.forcerange[0])), abs(float(act.forcerange[1])))
        scale[act.target] = ACTION_SCALE_FRACTION * limit
    return scale


MATLAS_ACTION_SCALE: dict[str, float] = _build_action_scale()

MATLAS_ARTICULATION = EntityArticulationInfoCfg(
    # Wrap every torque motor already defined in the XML.
    actuators=(XmlActuatorCfg(target_names_expr=(".*",)),),
)

STANDING_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, STANDING_HEIGHT),
    # Unlisted joints default to 0.0 (see Entity init_state resolution).
    joint_pos=dict(STANDING_JOINTS),
    joint_vel={".*": 0.0},
)


def get_matlas_robot_cfg() -> EntityCfg:
    """Fresh Matlas robot config (new instance avoids cross-task mutation)."""
    return EntityCfg(
        spec_fn=get_spec,
        articulation=MATLAS_ARTICULATION,
        init_state=STANDING_KEYFRAME,
    )
