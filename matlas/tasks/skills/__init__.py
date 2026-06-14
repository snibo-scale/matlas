"""Matlas actuation-characterization skill tasks.

Each task is registered as an independent mjlab environment so policies can be
trained in sequence via checkpoint warm starts.
"""

from mjlab.tasks.registry import register_mjlab_task

from matlas.tasks.skills.agents import skill_ppo_runner_cfg
from matlas.tasks.skills.skill_env_cfg import SKILL_TASKS, make_skill_env_cfg

for _name, _task in SKILL_TASKS.items():
    _task_id = f"Mjlab-Matlas-{_task.task_id_suffix}"
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=make_skill_env_cfg(_name),
        play_env_cfg=make_skill_env_cfg(_name, play=True),
        rl_cfg=skill_ppo_runner_cfg(_name),
    )

