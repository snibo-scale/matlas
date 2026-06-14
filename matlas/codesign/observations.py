"""Design-conditioning observation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from matlas.codesign.events import get_state

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def design_params(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
    """Per-env normalized actuator design (effort/armature/velocity per group).

    Shape [num_envs, 3 * N_GROUPS]. This is what makes the policy
    hardware-conditioned: it sees the actuators it has been given.
    """
    return get_state(env, asset_cfg.name).design_obs
