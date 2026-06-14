"""Design-conditioned standing task.

Extends the base stand task (``matlas/tasks/stand``) so each parallel env runs a
different sampled actuator design: a reset event applies per-env actuator
parameters, a design observation makes the policy hardware-conditioned, and
energy/over-speed costs make it energy-aware. The trained policy is what the
NSGA-II Pareto search evaluates each candidate design against.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from matlas.codesign import catalog as cat
from matlas.codesign import observations as codesign_obs
from matlas.codesign import rewards as codesign_rew
from matlas.codesign.events import set_actuator_design
from matlas.tasks.stand.stand_env_cfg import make_stand_env_cfg

_ALL_JOINTS = SceneEntityCfg("robot", joint_names=(".*",))


def make_codesign_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    # Base off the full-gravity, full-noise stand stage.
    cfg = make_stand_env_cfg(stage=3, play=play)

    # Effort authority spans the whole catalog; per-env forcerange (set by the
    # design event) clamps each env to its own design, and the design obs tells
    # the policy its actual limits.
    cfg.actions["effort"].scale = float(cat.MAX_EFFORT)

    # Sample a fresh actuator design per env on reset.
    cfg.events["set_actuator_design"] = EventTermCfg(
        func=set_actuator_design,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Make the policy hardware-conditioned: add the design block to both groups.
    design_term = ObservationTermCfg(func=codesign_obs.design_params)
    for group in ("actor", "critic"):
        cfg.observations[group].terms["design"] = design_term

    # Energy-aware shaping (small weights; standing still dominates).
    cfg.rewards["electrical_energy"] = RewardTermCfg(
        func=mdp.electrical_power_cost,
        weight=-2.0e-4,
        params={"asset_cfg": _ALL_JOINTS},
    )
    cfg.rewards["friction_energy"] = RewardTermCfg(
        func=codesign_rew.friction_power_cost, weight=-2.0e-4
    )
    cfg.rewards["over_speed"] = RewardTermCfg(
        func=codesign_rew.over_speed_cost, weight=-0.02
    )
    return cfg
