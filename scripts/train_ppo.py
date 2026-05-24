"""Train a first PPO walking policy.

Install the optional RL dependencies first:

    uv sync --extra rl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.walk_env import MatlasWalkEnv, WalkEnvConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--out", default="runs/matlas_walk_ppo")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing RL dependencies. Run: uv sync --extra rl"
        ) from exc

    def make_env() -> Monitor:
        return Monitor(MatlasWalkEnv(WalkEnvConfig(), seed=args.seed))

    env = make_env()
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(f"saved={out}")


if __name__ == "__main__":
    main()
