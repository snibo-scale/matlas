"""Observation terms shared by Matlas skill tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def task_phase(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Episode phase encoded as sin/cos plus linear progress."""
    phase = (env.episode_length_buf.float() / max(env.max_episode_length, 1)).clamp(
        0.0, 1.0
    )
    angle = 2.0 * torch.pi * phase
    return torch.stack((torch.sin(angle), torch.cos(angle), phase), dim=1)

