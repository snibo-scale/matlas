"""Run a short rollout to verify the walking environment is wired correctly."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.walk_env import MatlasWalkEnv, WalkEnvConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--zero-action", action="store_true")
    args = parser.parse_args()

    env = MatlasWalkEnv(WalkEnvConfig(), seed=args.seed)
    obs, info = env.reset(seed=args.seed)
    total_reward = 0.0
    terminated = False
    truncated = False

    for step in range(args.steps):
        if args.zero_action:
            action = np.zeros(env.action_space.shape if hasattr(env, "action_space") else 12)
        else:
            action = env.rng.uniform(-0.1, 0.1, size=len(env.controlled_actuator_ids))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"steps={step + 1}")
    print(f"terminated={terminated} truncated={truncated}")
    print(f"obs_dim={obs.shape[0]} action_dim={len(env.controlled_actuator_ids)}")
    print(f"total_reward={total_reward:.3f}")
    print(f"base_x={info['base_x']:.3f} base_height={info['base_height']:.3f}")
    print(f"mean_abs_torque={info['mean_abs_torque']:.3f}")
    print(f"mean_mechanical_power={info['mean_mechanical_power']:.3f}")
    print(f"leg_saturation_fraction={info['leg_saturation_fraction']:.3f}")


if __name__ == "__main__":
    main()
