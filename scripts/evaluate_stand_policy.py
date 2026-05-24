"""Evaluate a trained standing policy checkpoint."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.stand_env import MatlasStandEnv, STAND_STAGE_CONFIGS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/stand_stages/final")
    parser.add_argument("--stage", type=int, choices=sorted(STAND_STAGE_CONFIGS), default=3)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--reset-noise", type=float)
    parser.add_argument("--gravity-scale", type=float)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing RL dependencies. Run: uv sync --extra rl") from exc

    config = STAND_STAGE_CONFIGS[args.stage]
    config_overrides = {}
    if args.reset_noise is not None:
        config_overrides["reset_noise"] = args.reset_noise
    if args.gravity_scale is not None:
        config_overrides["gravity_scale"] = args.gravity_scale
    if config_overrides:
        config = replace(config, **config_overrides)
    env = MatlasStandEnv(config, seed=args.seed)
    model = PPO.load(args.model)
    lengths: list[int] = []
    rewards: list[float] = []
    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        total_reward = 0.0
        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        lengths.append(step + 1)
        rewards.append(total_reward)

    print(f"model={args.model}")
    print(f"stage={args.stage}")
    print(f"episodes={args.episodes}")
    print(f"deterministic={not args.stochastic}")
    print(f"reset_noise={config.reset_noise}")
    print(f"gravity_scale={config.gravity_scale}")
    print(f"mean_length={sum(lengths) / len(lengths):.2f}")
    print(f"max_length={max(lengths)}")
    print(f"mean_reward={sum(rewards) / len(rewards):.3f}")
    print(f"lengths={lengths}")


if __name__ == "__main__":
    main()
