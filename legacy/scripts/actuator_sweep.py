"""Evaluate fixed policies under actuator torque scaling.

This is intentionally simple: it gives you the measurement harness for actuator
constraints before a learned policy exists. Replace ``policy`` with a trained
policy call when training is added.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.walk_env import MatlasWalkEnv, WalkEnvConfig


def rollout(env: MatlasWalkEnv, steps: int, seed: int) -> dict[str, float]:
    env.reset(seed=seed)
    rewards = []
    saturation = []
    power = []
    terminated = False
    for _ in range(steps):
        action = np.zeros(len(env.controlled_actuator_ids), dtype=np.float64)
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        saturation.append(info["leg_saturation_fraction"])
        power.append(info["mean_mechanical_power"])
        if terminated or truncated:
            break
    return {
        "steps": float(len(rewards)),
        "terminated": float(terminated),
        "reward": float(np.sum(rewards)),
        "base_x": float(info["base_x"]),
        "sat": float(np.mean(saturation)),
        "power": float(np.mean(power)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--torque-scales",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0, 1.25],
    )
    args = parser.parse_args()

    print("scale,steps,terminated,reward,base_x,leg_sat,mean_power")
    for scale in args.torque_scales:
        cfg = replace(WalkEnvConfig(), command_velocity=(0.25, 0.0, 0.0))
        env = MatlasWalkEnv(cfg, seed=args.seed)
        env.ctrl_low *= scale
        env.ctrl_high *= scale
        env.torque_limit *= scale
        metrics = rollout(env, args.steps, args.seed)
        print(
            f"{scale:.3f},{metrics['steps']:.0f},{metrics['terminated']:.0f},"
            f"{metrics['reward']:.3f},{metrics['base_x']:.3f},"
            f"{metrics['sat']:.3f},{metrics['power']:.3f}"
        )


if __name__ == "__main__":
    main()
