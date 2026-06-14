# Legacy (retired)

This directory holds the pre-mjlab stack, kept for reference only:

- `envs/` — Gymnasium environments (`MatlasWalkEnv`, `MatlasStandEnv`) with a
  hand-written PD/direct torque control loop.
- `scripts/` — stable-baselines3 PPO/SAC training, the sweep harness, and the
  smoke/evaluation/visualization scripts that drove them.

These are no longer wired into the build (the `gymnasium` / `stable-baselines3`
optional dependency group was removed) and may not run as-is — paths and
imports assume the old top-level `envs/` layout. The standing task they
implemented now lives in `matlas/tasks/stand/` on top of mjlab. Pose data moved
to `components/poses.py`.
