# Matlas — mjlab standing task

GPU-parallel RL for the Matlas biped, built on [mjlab](https://github.com/mujocolab/mjlab)
(Isaac-Lab-style manager-based envs on MuJoCo-Warp) with RSL-RL PPO.

The robot is a single committed MJCF (`models/matlas/matlas.xml`) whose
`<actuator>` block carries 21 torque (effort) motors — 36 N·m for most joints,
12 N·m for the ankles, with gear −1 on knee/elbow so positive control means
flexion. The policy commands joint torques directly.

## Layout

- `models/matlas/matlas.xml` (+ `assets/`) — committed robot MJCF, the model mjlab loads.
- `components/` — URDF→MjSpec assembly. Now a **build-time** source for the export, not used at training time.
- `scripts/export_mjcf.py` — bakes the assembly into `models/matlas/matlas.xml` (actuators in XML).
- `matlas/robots/matlas_constants.py` — mjlab `EntityCfg` wrapping the XML's effort actuators.
- `matlas/tasks/stand/` — manager-based standing task (`stand_env_cfg.py`, custom `mdp/` reward + termination terms, RSL-RL PPO cfg).
- `scripts/train.py` / `scripts/play.py` — thin wrappers that register the task then call mjlab's CLI.
- `legacy/` — retired Gymnasium envs + stable-baselines3 scripts (kept for reference only).

## Quick start

Install [uv](https://docs.astral.sh/uv/) if needed, then from a fresh clone:

```bash
git clone https://github.com/snibo-scale/matlas.git
cd matlas
uv sync
```

The project targets Python 3.11+ and installs MuJoCo, MuJoCo-Warp/mjlab,
RSL-RL support through mjlab, NumPy, and the small optimization stack used by
the co-design search.

### CPU / Mac smoke tests

macOS and CPU-only machines are useful for authoring, XML checks, and short
wiring tests:

```bash
uv run python -m compileall matlas scripts
uv run python scripts/audit_collision_model.py
uv run python scripts/view_pose.py --pose standing
uv run python scripts/skill_torque_sweep.py Mjlab-Matlas-StandBalance \
  --scales 0.8 1.0 \
  --num-envs 2 \
  --horizon 2 \
  --device cpu
```

On macOS, MuJoCo viewer behavior can depend on Apple's main-thread rendering
rules. If a viewer launch complains, run the same command through `mjpython`
inside the uv environment.

### CUDA training machine

Real training expects an NVIDIA/CUDA machine because mjlab uses MuJoCo-Warp for
parallel simulation. A typical run is:

```bash
uv run python scripts/train.py Mjlab-Matlas-StandBalance --env.scene.num-envs 4096
```

When training on a CPU-only machine, mjlab will construct and step the
environment, but it will be too slow for serious learning.

## Regenerate the robot MJCF

Run whenever the URDF assembly or actuator definitions change:

```bash
uv run python scripts/export_mjcf.py
```

This writes `models/matlas/matlas.xml`, copies meshes into `models/matlas/assets/`,
and round-trips the file to confirm it loads with 21 motors.

## Train

Training needs a CUDA GPU (MuJoCo-Warp). Tasks: `Mjlab-Matlas-Stand` (alias of
stage 3) plus `Mjlab-Matlas-Stand-S1/S2/S3`.

```bash
uv run python scripts/train.py Mjlab-Matlas-Stand --env.scene.num-envs 4096
```

The standing curriculum (mirrors the legacy stages) varies gravity, reset noise,
episode length, and termination tolerances:

- **S1**: 0.1× gravity, deterministic reset, generous tolerances — learn to hold the pose.
- **S2**: full gravity, small reset noise, tighter tolerances.
- **S3**: full gravity, larger reset noise, tightest tolerances (default).

Warm-start a later stage from an earlier run with RSL-RL's `--agent.resume` /
`--agent.load-run` flags.

## Skill curriculum

The actuator-characterization curriculum is in `matlas/tasks/skills/`:

1. `Mjlab-Matlas-StandBalance`
2. `Mjlab-Matlas-Squat`
3. `Mjlab-Matlas-LoadedSquat`
4. `Mjlab-Matlas-SingleStep`
5. `Mjlab-Matlas-StairStep`
6. `Mjlab-Matlas-JumpForward`
7. `Mjlab-Matlas-BackFlip`

Train each stage in order and warm-start from the previous checkpoint:

```bash
uv run python scripts/train.py Mjlab-Matlas-Squat --env.scene.num-envs 4096

uv run python scripts/train.py Mjlab-Matlas-LoadedSquat \
  --agent.resume true \
  --agent.load-run <previous-run>
```

The skill tasks randomize actuator envelopes per environment. The policy sees
`[torque_scale, velocity_scale, continuous_torque_fraction]`, while reward terms
penalize torque saturation, continuous-torque excess, over-speed, and a soft
linear torque-speed envelope violation.

Default exploration torque scales:

| Task | Torque scale range |
| --- | --- |
| `StandBalance` | `0.8x-1.2x` |
| `Squat` | `1.0x-1.4x` |
| `LoadedSquat` | `1.0x-1.5x` |
| `SingleStep` | `1.0x-1.5x` |
| `StairStep` | `1.1x-1.6x` |
| `JumpForward` | `1.3x-2.0x` |
| `BackFlip` | `1.5x-2.5x` |

After training, sweep fixed limits to find the feasibility threshold:

```bash
uv run python scripts/skill_torque_sweep.py Mjlab-Matlas-JumpForward \
  --checkpoint logs/rsl_rl/matlas_jump_forward/<run>/model_<n>.pt \
  --scales 0.6 0.7 0.8 0.9 1.0 1.2 1.5 2.0 \
  --num-envs 128 \
  --out runs/skill_sweeps/jump_forward.json
```

The sweep reports survival, success, positive mechanical power, saturation
fraction, speed violation, and torque-speed violation.

## Play

```bash
uv run python scripts/play.py Mjlab-Matlas-Stand --checkpoint-file logs/rsl_rl/matlas_stand/<run>/model_<n>.pt
```

## Reward (stand task)

Ports `legacy/envs/stand_env.py` term-for-term: `alive` (+1.0), `upright`
(+1.5), `height` (+0.8), `posture` (+1.2), `stillness`/`angular_stillness`
(+0.5 each), minus `torque_cost`, `joint_speed_cost`, `action_rate_cost`, and an
`-8.0` fall penalty. Terminations: time-out, tilt past the per-stage limit, base
height out of band, any DOF speed > 40, and NaN guard.

## Actuator co-design (actorob-style)

Search the motor+gearbox catalog for the **mass / energy / performance Pareto
front**, the RL analog of [actorob](https://mkakanov.github.io/actorob/). Two
phases:

1. **Co-train a design-conditioned policy.** `Mjlab-Matlas-Stand-CoDesign` gives
   each parallel env a different sampled actuator design (per-env
   `actuator_forcerange` / `dof_armature` / `dof_frictionloss`) and feeds the
   design to the policy as extra observations, so one policy learns to stand
   with *whatever actuators it's given*. Energy and over-speed costs make it
   energy-aware.
   ```bash
   uv run python scripts/train.py Mjlab-Matlas-Stand-CoDesign --env.scene.num-envs 4096
   ```
2. **Pareto search.** NSGA-II over the catalog evaluates each candidate design by
   rolling out the frozen policy in parallel envs, minimizing
   `-survival`, mean actuator power, and total actuator mass.
   ```bash
   uv run python scripts/codesign_search.py --checkpoint logs/rsl_rl/matlas_stand/<run>/model_<n>.pt \
       --pop 100 --generations 60
   ```
   Writes the front to `runs/codesign/pareto.json` (sorted by mass). A
   `--pop 4 --generations 1` zero-policy run is a CPU wiring smoke test.

Co-design lives in `matlas/codesign/`: `catalog.py` (motors, gear ratios, joint
groups, and the design→`effort/velocity/armature/friction/mass` map via mjlab's
`reflected_inertia`), `events.py` (per-env design application), `observations.py`
(design conditioning), `rewards.py` (friction + over-speed costs). The motor
catalog, gear options, and the placeholder friction/mass models are the knobs to
edit. Motor velocity limits are enforced as a soft over-speed penalty (MuJoCo
motors have no native speed clamp).

## Inspect the model

```bash
uv run python scripts/view_pose.py --pose standing      # named poses in the MuJoCo viewer
uv run python scripts/audit_collision_model.py          # compiled collision geometry
```

## Notes

The Mac is for authoring and CPU smoke tests only; MuJoCo-Warp runs the heavy
training on CUDA. A CPU run (`CUDA_VISIBLE_DEVICES=""`) constructs and steps the
env but is too slow for real training.
