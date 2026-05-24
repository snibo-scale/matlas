"""Visualize a trained standing policy in MuJoCo."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.stand_env import MatlasStandEnv, STAND_STAGE_CONFIGS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/stand_stages_long/final")
    parser.add_argument("--stage", type=int, choices=sorted(STAND_STAGE_CONFIGS), default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--zero-action", action="store_true")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing RL dependencies. Run: uv sync --extra rl") from exc

    env = MatlasStandEnv(STAND_STAGE_CONFIGS[args.stage], seed=args.seed)
    obs, _ = env.reset(seed=args.seed)
    policy = None if args.zero_action else PPO.load(args.model)
    last_policy_time = -np.inf
    action = np.zeros(len(env.controlled_actuator_ids), dtype=np.float64)

    def apply_policy(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        nonlocal obs, action, last_policy_time
        if data.time - last_policy_time >= env.config.control_dt:
            obs = env._get_obs()
            if policy is None:
                action = np.zeros(len(env.controlled_actuator_ids), dtype=np.float64)
            else:
                action, _ = policy.predict(obs, deterministic=True)
            env.previous_action[:] = action
            last_policy_time = data.time
        data.ctrl[:] = env._pd_control(action)

    if sys.platform == "darwin":
        mujoco.set_mjcb_control(apply_policy)
        try:
            mujoco.viewer.launch(env.model, env.data)
        finally:
            mujoco.set_mjcb_control(None)
        return

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 135
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        while viewer.is_running():
            apply_policy(env.model, env.data)
            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            time.sleep(env.model.opt.timestep)


if __name__ == "__main__":
    main()
