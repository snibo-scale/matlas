# Matlas Skill Curriculum

These tasks are scaffolded for actuator characterization and are intended to be
trained in order, warm-starting each stage from the previous checkpoint:

1. `Mjlab-Matlas-StandBalance`
2. `Mjlab-Matlas-Squat`
3. `Mjlab-Matlas-LoadedSquat`
4. `Mjlab-Matlas-SingleStep`
5. `Mjlab-Matlas-StairStep`
6. `Mjlab-Matlas-JumpForward`
7. `Mjlab-Matlas-BackFlip`

Run a stage with:

```bash
uv run python scripts/train.py Mjlab-Matlas-Squat --env.scene.num-envs 4096
```

Warm-start the next stage with RSL-RL resume flags, for example:

```bash
uv run python scripts/train.py Mjlab-Matlas-LoadedSquat \
  --agent.resume true \
  --agent.load-run <previous-run>
```

The tasks currently use direct torque actions, phase observations, simple
height/progress/jump/rotation rewards, and per-env actuator-envelope
randomization. The policy observes `[torque_scale, velocity_scale,
continuous_torque_fraction]`, while rewards penalize torque saturation,
continuous-torque excess, over-speed, and torque-speed envelope violations.

After training, sweep fixed torque scales:

```bash
uv run python scripts/skill_torque_sweep.py Mjlab-Matlas-JumpForward \
  --checkpoint logs/rsl_rl/matlas_jump_forward/<run>/model_<n>.pt \
  --scales 0.6 0.7 0.8 0.9 1.0 1.2 1.5 2.0 \
  --out runs/skill_sweeps/jump_forward.json
```

`BackFlip` is intentionally a reward scaffold: for a real learned flip, replace
or augment the sparse rotation/landing rewards with a reference-motion imitation
term.
