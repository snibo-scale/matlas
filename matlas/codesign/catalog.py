"""Motor + gearbox catalog and the design -> actuator-parameter map.

A *design* assigns each joint group a discrete motor and a gear ratio. From
those we derive the joint-level actuator parameters mjlab can apply per env:

  effort_limit   = peak_torque * gear           [N*m at the joint]
  velocity_limit = max_speed   / gear           [rad/s at the joint]
  armature       = reflected_inertia(rotor, gear) = rotor_inertia * gear^2
  frictionloss   = FRICTION_FRAC * peak_torque * gear   (simple gearbox model)
  mass           = motor_mass + GEARBOX_MASS_PER_RATIO * gear

The friction and gearbox-mass models are deliberately simple placeholders (the
actorob paper fits these from datasheets); swap in regressions later without
touching the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.utils.actuator import reflected_inertia

# --- Joint groups (regex-free substring match on joint names) ----------------
# Six groups keep the design vector small. Each maps to the joints whose name
# contains one of its keys; "_1" mirror joints fall in the same group.
GROUPS: tuple[str, ...] = ("hip", "knee", "ankle", "shoulder", "elbow", "torso")

_GROUP_KEYS: dict[str, tuple[str, ...]] = {
    "hip": ("hip_pitch", "hip_roll", "yaw"),
    "knee": ("knee",),
    "ankle": ("ankle_pitch", "ankle_roll"),
    "shoulder": ("shoulder_yaw", "shoulder_roll", "shoulder_pitch"),
    "elbow": ("elbow",),
    "torso": ("revolute",),
}


def group_of(joint_name: str) -> str:
    """Return the design group for a joint name (raises if unmapped)."""
    base = joint_name.rsplit("/", 1)[-1].removesuffix("_1")
    for group, keys in _GROUP_KEYS.items():
        if any(base == key or base.startswith(key) for key in keys):
            return group
    raise KeyError(f"Joint {joint_name!r} not assigned to any design group")


# --- Motor catalog -----------------------------------------------------------
@dataclass(frozen=True)
class Motor:
    """A frameless BLDC motor (pre-gearbox specs)."""

    name: str
    rotor_inertia: float  # kg*m^2, at the rotor
    peak_torque: float  # N*m, at the rotor
    max_speed: float  # rad/s, at the rotor
    mass_kg: float


# Small / medium / large representative actuators.
MOTOR_CATALOG: tuple[Motor, ...] = (
    Motor("s", rotor_inertia=5.0e-5, peak_torque=6.0, max_speed=45.0, mass_kg=0.45),
    Motor("m", rotor_inertia=1.2e-4, peak_torque=12.0, max_speed=35.0, mass_kg=0.95),
    Motor("l", rotor_inertia=2.5e-4, peak_torque=20.0, max_speed=25.0, mass_kg=1.8),
)

# Discrete gear ratios the optimizer may choose.
GEAR_OPTIONS: tuple[float, ...] = (1.0, 6.0, 9.0, 16.0, 25.0)

FRICTION_FRAC = 0.02  # Coulomb friction as a fraction of joint torque.
GEARBOX_MASS_PER_RATIO = 0.01  # kg added per unit gear ratio.

# Catalog extremes, used to normalize the design observation and to set the
# (fixed) effort-action scale. The per-env forcerange clamp then limits each
# env's torque to its own design, so the policy reads its limits from the obs.
MAX_EFFORT = max(m.peak_torque for m in MOTOR_CATALOG) * max(GEAR_OPTIONS)
MAX_ARMATURE = max(m.rotor_inertia for m in MOTOR_CATALOG) * max(GEAR_OPTIONS) ** 2
MAX_VELOCITY = max(m.max_speed for m in MOTOR_CATALOG) / min(GEAR_OPTIONS)


@dataclass(frozen=True)
class ActuatorParams:
    effort_limit: float
    velocity_limit: float
    armature: float
    frictionloss: float
    mass: float


def actuator_params(motor_idx: int, gear: float) -> ActuatorParams:
    """Per-joint actuator parameters for one (motor, gear) choice."""
    motor = MOTOR_CATALOG[motor_idx]
    return ActuatorParams(
        effort_limit=motor.peak_torque * gear,
        velocity_limit=motor.max_speed / gear,
        armature=reflected_inertia(motor.rotor_inertia, gear),
        frictionloss=FRICTION_FRAC * motor.peak_torque * gear,
        mass=motor.mass_kg + GEARBOX_MASS_PER_RATIO * gear,
    )


# --- Design encode/decode for the optimizer ----------------------------------
# A design is {group: (motor_idx, gear_idx)}. NSGA-II works on an integer genome
# of length 2*len(GROUPS): [motor_idx per group..., gear_idx per group...].
Design = dict[str, tuple[int, int]]

N_GROUPS = len(GROUPS)
GENOME_LEN = 2 * N_GROUPS
# Per-gene upper bounds (inclusive) for an integer-coded optimizer.
GENOME_UPPER: tuple[int, ...] = (
    *([len(MOTOR_CATALOG) - 1] * N_GROUPS),
    *([len(GEAR_OPTIONS) - 1] * N_GROUPS),
)


def genome_to_design(genome) -> Design:
    """Decode a length-2N integer genome into a design dict."""
    motor_idx = [int(round(g)) for g in genome[:N_GROUPS]]
    gear_idx = [int(round(g)) for g in genome[N_GROUPS:]]
    return {
        group: (motor_idx[i], gear_idx[i]) for i, group in enumerate(GROUPS)
    }


def design_to_genome(design: Design) -> list[int]:
    motors = [design[g][0] for g in GROUPS]
    gears = [design[g][1] for g in GROUPS]
    return motors + gears


def design_group_params(design: Design) -> dict[str, ActuatorParams]:
    """Resolve every group's actuator parameters for a design."""
    return {
        group: actuator_params(motor_idx, GEAR_OPTIONS[gear_idx])
        for group, (motor_idx, gear_idx) in design.items()
    }
