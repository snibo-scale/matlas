"""Evaluate a skill policy across fixed actuator torque scales.

This complements randomized training. During training each skill samples a
torque envelope per env; this script fixes the envelope to one scale at a time
and reports feasibility metrics. Omit --checkpoint for a zero-policy wiring
smoke test.

Examples:

    uv run python scripts/skill_torque_sweep.py Mjlab-Matlas-JumpForward \
      --checkpoint logs/rsl_rl/matlas_jump_forward/<run>/model_1000.pt

    uv run python scripts/skill_torque_sweep.py Mjlab-Matlas-BackFlip \
      --scales 0.7 0.8 0.9 1.0 1.2 1.5 2.0 --num-envs 32
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matlas.tasks  # noqa: F401
from matlas.tasks.skills.mdp.actuation import ActuatorEnvelopeCfg, get_state
from matlas.tasks.skills.skill_env_cfg import SKILL_TASKS, make_skill_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg

TASK_TO_SKILL = {
    f"Mjlab-Matlas-{task.task_id_suffix}": name
    for name, task in SKILL_TASKS.items()
}


def _build_env_and_policy(
    task_id: str,
    torque_scale: float,
    velocity_scale: float,
    num_envs: int,
    device: str,
    checkpoint: str | None,
):
    skill_name = TASK_TO_SKILL[task_id]
    env_cfg = make_skill_env_cfg(
        skill_name,
        play=False,
        actuator_envelope=ActuatorEnvelopeCfg(
            torque_scale=(torque_scale, torque_scale),
            velocity_scale=(velocity_scale, velocity_scale),
        ),
    )
    env_cfg.scene.num_envs = num_envs
    agent_cfg = load_rl_cfg(task_id)
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

    if checkpoint is None:
        action_shape = env.unwrapped.action_space.shape

        def policy(obs):
            del obs
            return torch.zeros(action_shape, device=device)

    else:
        runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
        runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)
    return env, policy


@torch.no_grad()
def _evaluate(env, policy, horizon: int) -> dict[str, float]:
    base = env.unwrapped
    robot = base.scene["robot"]
    state = get_state(base)
    obs, _ = env.reset()

    alive = torch.ones(base.num_envs, dtype=torch.bool, device=base.device)
    survived_steps = torch.zeros(base.num_envs, device=base.device)
    power_sum = torch.zeros(base.num_envs, device=base.device)
    saturation_sum = torch.zeros(base.num_envs, device=base.device)
    speed_violation_sum = torch.zeros(base.num_envs, device=base.device)
    torque_speed_violation_sum = torch.zeros(base.num_envs, device=base.device)

    for _ in range(horizon):
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)

        tau = robot.data.actuator_force[:, state.ctrl_ids]
        qd = robot.data.joint_vel[:, state.joint_cols]
        effort = state.effort_limit.clamp_min(1e-6)
        velocity = state.velocity_limit.clamp_min(1e-6)
        speed_ratio = qd.abs() / velocity
        available = effort * torch.relu(1.0 - speed_ratio)

        power = torch.relu(tau * qd).sum(1)
        saturation = (tau.abs() > 0.98 * effort).float().mean(1)
        speed_violation = torch.relu(speed_ratio - 1.0).mean(1)
        torque_speed_violation = torch.relu(tau.abs() - available).div(effort).mean(1)

        power_sum += torch.where(alive, power, torch.zeros_like(power))
        saturation_sum += torch.where(alive, saturation, torch.zeros_like(saturation))
        speed_violation_sum += torch.where(alive, speed_violation, torch.zeros_like(speed_violation))
        torque_speed_violation_sum += torch.where(
            alive,
            torque_speed_violation,
            torch.zeros_like(torque_speed_violation),
        )
        survived_steps += alive.float()
        alive = alive & ~dones.to(torch.bool)

    denom = survived_steps.clamp_min(1.0)
    return {
        "survival_frac": float((survived_steps / horizon).mean().cpu()),
        "success_frac": float(alive.float().mean().cpu()),
        "mean_positive_power": float((power_sum / denom).mean().cpu()),
        "saturation_fraction": float((saturation_sum / denom).mean().cpu()),
        "speed_violation": float((speed_violation_sum / denom).mean().cpu()),
        "torque_speed_violation": float((torque_speed_violation_sum / denom).mean().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id", choices=sorted(TASK_TO_SKILL))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0],
    )
    parser.add_argument("--velocity-scale", type=float, default=1.0)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results = []
    for scale in args.scales:
        env, policy = _build_env_and_policy(
            args.task_id,
            scale,
            args.velocity_scale,
            args.num_envs,
            args.device,
            args.checkpoint,
        )
        metrics = _evaluate(env, policy, args.horizon)
        env.close()
        row = {"task_id": args.task_id, "torque_scale": scale, **metrics}
        results.append(row)
        print(
            f"{args.task_id} scale={scale:4.2f} survival={metrics['survival_frac']:.3f} "
            f"success={metrics['success_frac']:.3f} sat={metrics['saturation_fraction']:.3f} "
            f"ts_viol={metrics['torque_speed_violation']:.3f} power={metrics['mean_positive_power']:.2f}"
        )

    if args.out is not None:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()

