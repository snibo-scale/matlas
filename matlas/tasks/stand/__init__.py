"""Matlas standing-balance task.

Importing this module registers the task with mjlab's task registry. The 3-stage
curriculum is exposed as separate task IDs (gravity/noise/episode/tolerance
differ per stage); ``Mjlab-Matlas-Stand`` aliases the final stage.
"""

from mjlab.tasks.registry import register_mjlab_task

from matlas.tasks.stand.agents import stand_ppo_runner_cfg
from matlas.tasks.stand.stand_env_cfg import STAND_STAGES, make_stand_env_cfg

for _stage in STAND_STAGES:
    register_mjlab_task(
        task_id=f"Mjlab-Matlas-Stand-S{_stage}",
        env_cfg=make_stand_env_cfg(stage=_stage),
        play_env_cfg=make_stand_env_cfg(stage=_stage, play=True),
        rl_cfg=stand_ppo_runner_cfg(),
    )

# Default alias points at the final curriculum stage.
register_mjlab_task(
    task_id="Mjlab-Matlas-Stand",
    env_cfg=make_stand_env_cfg(stage=3),
    play_env_cfg=make_stand_env_cfg(stage=3, play=True),
    rl_cfg=stand_ppo_runner_cfg(),
)
