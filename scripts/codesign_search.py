"""Multi-objective actuator co-design search (NSGA-II) on mjlab.

Given a frozen design-conditioned policy (trained on
``Mjlab-Matlas-Stand-CoDesign``), search the motor+gear catalog for the
mass / energy / performance Pareto front. Each candidate design is injected into
a parallel env, the policy is rolled out, and three objectives are minimized:

  f1 = -survival_fraction        (task performance: stay standing)
  f2 = mean actuator power       (electrical + gearbox-friction, while alive)
  f3 = total actuator mass

Designs are evaluated num_envs at a time (one per env). Training needs CUDA;
this search runs on CPU too (slowly) and supports a zero-action policy for
wiring dry-runs. Run from the repo root::

    uv run python scripts/codesign_search.py --checkpoint <ckpt.pt> --pop 100 --generations 60
    uv run python scripts/codesign_search.py --pop 4 --generations 1   # CPU dry-run, zero policy
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matlas.tasks  # noqa: F401  (registers tasks)
from matlas.codesign import catalog as cat
from matlas.codesign.events import get_state
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize

TASK_ID = "Mjlab-Matlas-Stand-CoDesign"


def _build_env_and_policy(num_envs: int, device: str, checkpoint: str | None):
    env_cfg = load_env_cfg(TASK_ID, play=True)  # episode_length_s huge -> done == fall
    env_cfg.scene.num_envs = num_envs
    agent_cfg = load_rl_cfg(TASK_ID)
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
def _evaluate_designs(env, policy, genomes: torch.Tensor, horizon: int) -> np.ndarray:
    """Roll out the policy under each design; return [N, 3] objectives."""
    base = env.unwrapped
    state = get_state(base)
    n = genomes.shape[0]
    device = base.device

    base._codesign_override = genomes.to(device).long()
    obs, _ = env.reset()

    alive = torch.ones(n, dtype=torch.bool, device=device)
    survived = torch.zeros(n, device=device)
    power_sum = torch.zeros(n, device=device)
    steps_alive = torch.zeros(n, device=device)

    for _ in range(horizon):
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)

        af = base.scene["robot"].data.actuator_force
        qd = base.scene["robot"].data.joint_vel[:, state.joint_cols]
        friction = base.sim.model.dof_frictionloss[:, state.v_adr]
        power = torch.relu(af * qd).sum(1) + (friction * qd.abs()).sum(1)

        power_sum += torch.where(alive, power, torch.zeros_like(power))
        steps_alive += alive.float()
        survived += alive.float()
        alive = alive & ~dones.to(torch.bool)

    survival_frac = survived / horizon
    mean_power = power_sum / steps_alive.clamp_min(1.0)
    mass = state.mass.clone()
    return torch.stack([-survival_frac, mean_power, mass], dim=1).cpu().numpy()


class CoDesignProblem(Problem):
    def __init__(self, env, policy, horizon: int):
        super().__init__(
            n_var=cat.GENOME_LEN,
            n_obj=3,
            xl=np.zeros(cat.GENOME_LEN),
            xu=np.array(cat.GENOME_UPPER, dtype=float),
            vtype=int,
        )
        self._env = env
        self._policy = policy
        self._horizon = horizon
        self._num_envs = env.unwrapped.num_envs

    def _evaluate(self, X, out, *args, **kwargs):
        # Evaluate in chunks of num_envs so the population can exceed env count.
        results = []
        for start in range(0, len(X), self._num_envs):
            chunk = X[start : start + self._num_envs]
            genomes = torch.as_tensor(np.asarray(chunk), dtype=torch.long)
            if len(genomes) < self._num_envs:
                pad = self._num_envs - len(genomes)
                genomes = torch.cat([genomes, genomes[:1].repeat(pad, 1)], dim=0)
                results.append(
                    _evaluate_designs(self._env, self._policy, genomes, self._horizon)[
                        : len(chunk)
                    ]
                )
            else:
                results.append(
                    _evaluate_designs(self._env, self._policy, genomes, self._horizon)
                )
        out["F"] = np.concatenate(results, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="trained policy .pt; omit for zero-policy dry-run")
    parser.add_argument("--pop", type=int, default=64)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=300, help="eval steps per design (~6s at 50Hz)")
    parser.add_argument("--num-envs", type=int, default=None, help="parallel envs (defaults to --pop)")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="runs/codesign/pareto.json")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    num_envs = args.num_envs or args.pop
    env, policy = _build_env_and_policy(num_envs, args.device, args.checkpoint)
    problem = CoDesignProblem(env, policy, args.horizon)

    algorithm = NSGA2(
        pop_size=args.pop,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, repair=RoundingRepair()),
        mutation=PM(eta=20, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )
    res = minimize(
        problem,
        algorithm,
        ("n_gen", args.generations),
        seed=args.seed,
        verbose=True,
    )

    front = []
    for x, f in zip(np.atleast_2d(res.X), np.atleast_2d(res.F)):
        design = cat.genome_to_design([int(round(v)) for v in x])
        readable = {
            g: {"motor": cat.MOTOR_CATALOG[mi].name, "gear": cat.GEAR_OPTIONS[gi]}
            for g, (mi, gi) in design.items()
        }
        front.append(
            {
                "design": readable,
                "survival_frac": float(-f[0]),
                "mean_power": float(f[1]),
                "mass_kg": float(f[2]),
            }
        )
    front.sort(key=lambda d: d["mass_kg"])

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(front, indent=2))
    print(f"\nPareto front: {len(front)} designs -> {out_path}")
    for d in front[:10]:
        print(
            f"  mass={d['mass_kg']:6.2f}kg  survival={d['survival_frac']:.2f}  "
            f"power={d['mean_power']:8.2f}  {d['design']}"
        )
    env.close()


if __name__ == "__main__":
    main()
