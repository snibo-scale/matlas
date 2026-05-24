## Walking Simulation Scaffold

This repo now has a minimal MuJoCo walking-task environment around the full
Matlas assembly.

### Smoke test

```bash
uv run python scripts/smoke_walk_env.py --steps 200 --zero-action
```

The zero-action policy is expected to fall. This only verifies that the model,
software PD actuator wrapper, reward, termination, and actuator metrics execute.

### Environment

```python
from envs.walk_env import MatlasWalkEnv, WalkEnvConfig

env = MatlasWalkEnv(WalkEnvConfig(command_velocity=(0.25, 0.0, 0.0)))
obs, info = env.reset(seed=1)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Actions are normalized target offsets for the 12 leg joints. The environment
uses a software PD controller and clips the resulting torques by the MuJoCo
actuator ranges currently defined in `components/actuators.py`.

The reset pose is the named `standing` pose:

- `hip_pitch[_1] = 0.15 rad`
- `knee[_1] = -0.30 rad`
- `ankle_pitch[_1] = 0.15 rad`
- pelvis/base height = `0.61 m`

Zero action means "hold this nominal pose."

Positive motor controls use the actuator gear convention defined while building
actuators from the imported URDF joints. For example, knees and elbows get a
negative actuator gear so positive motor control means flexion, matching the
hip/shoulder flexion convention without special policy-side remapping.

Visualize named poses:

```bash
uv run python scripts/view_pose.py --pose standing
uv run python scripts/view_pose.py --pose crouch
```

Add `--simulate` to step physics while a simple PD controller tries to hold the
pose.

Useful metrics in `info`:

- `mean_abs_torque`
- `mean_mechanical_power`
- `saturation_fraction`
- `leg_saturation_fraction`

### PPO training

Install optional RL dependencies:

```bash
uv sync --extra rl
```

Train:

```bash
uv run python scripts/train_ppo.py --timesteps 200000
```

Train standing stages 1-3:

```bash
uv run python scripts/train_stand_stages.py --timesteps-per-stage 50000
```

Standing curriculum stages:

- Stage 1: zero gravity, deterministic reset, standing pose hold.
- Stage 2: full gravity, tiny reset noise.
- Stage 3: full gravity, larger reset noise.

This is intentionally a first baseline, not a production locomotion setup. The
next modeling steps should be joint limits, contact tuning, and domain
randomization.

### Collision audit

The URDF loader duplicates visual mesh tags into collision tags when collision
tags are missing, so the compiled collision meshes match the visual meshes.
Check the compiled model with:

```bash
uv run python scripts/audit_collision_model.py
```

### Actuator sweep

Before a trained policy exists, this runs a fixed zero-action policy only as a
measurement harness check:

```bash
uv run python scripts/actuator_sweep.py --steps 400 --torque-scales 0.5 0.75 1.0 1.25
```

After training, replace the zero policy in `scripts/actuator_sweep.py` with the
learned policy call and compare success rate, torque saturation, power, and
distance traveled across torque scales.
