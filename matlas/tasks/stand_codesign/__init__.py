"""Design-conditioned standing task (registers with mjlab on import)."""

from mjlab.tasks.registry import register_mjlab_task

from matlas.tasks.stand.agents import stand_ppo_runner_cfg
from matlas.tasks.stand_codesign.env_cfg import make_codesign_env_cfg

register_mjlab_task(
    task_id="Mjlab-Matlas-Stand-CoDesign",
    env_cfg=make_codesign_env_cfg(),
    play_env_cfg=make_codesign_env_cfg(play=True),
    rl_cfg=stand_ppo_runner_cfg(),
)
