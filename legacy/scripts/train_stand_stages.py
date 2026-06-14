"""Train standing balance through curriculum stages 1, 2, and 3.

Supports two algorithms:
  --algo ppo  (default) on-policy PPO with linear LR decay
  --algo sac  off-policy SAC with automatic entropy tuning

SAC is usually much more sample-efficient on continuous control and has
built-in entropy regularization, so it doesn't need the log_std_init
gymnastics that PPO needs to explore.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.stand_env import MatlasStandEnv, STAND_STAGE_CONFIGS


def make_lr_schedule(initial_lr: float, kind: str):
    """SB3 accepts learning_rate as either a float or a callable
    f(progress_remaining: float in [0, 1]) -> float."""
    if kind == "constant":
        return initial_lr
    if kind == "linear":
        return lambda progress_remaining: initial_lr * progress_remaining
    raise ValueError(f"Unknown lr schedule '{kind}'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--timesteps-per-stage", type=int, default=50_000)
    parser.add_argument("--out-dir", default="runs/stand_stages")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--load-model")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lr-schedule", choices=["constant", "linear"], default="linear")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=256)
    # PPO-only
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--target-kl", type=float)
    parser.add_argument("--log-std-init", type=float, default=-4.0)
    # SAC-only
    parser.add_argument("--sac-buffer-size", type=int, default=200_000)
    parser.add_argument("--sac-tau", type=float, default=0.005)
    parser.add_argument("--sac-train-freq", type=int, default=64)
    parser.add_argument("--sac-gradient-steps", type=int, default=64)
    parser.add_argument("--sac-learning-starts", type=int, default=2_000)
    # Reward/env overrides
    parser.add_argument("--torque-cost", type=float)
    parser.add_argument("--action-rate-cost", type=float)
    parser.add_argument("--joint-speed-cost", type=float)
    parser.add_argument("--posture-weight", type=float)
    parser.add_argument("--stillness-weight", type=float)
    parser.add_argument("--angular-stillness-weight", type=float)
    parser.add_argument("--upright-weight", type=float)
    parser.add_argument("--height-tracking-weight", type=float)
    parser.add_argument("--fall-penalty", type=float)
    parser.add_argument("--gravity-scale", type=float)
    parser.add_argument("--reset-noise", type=float)
    parser.add_argument("--episode-seconds", type=float)
    parser.add_argument("--healthy-height-min", type=float)
    parser.add_argument("--healthy-height-max", type=float)
    parser.add_argument("--xy-drift-weight", type=float)
    parser.add_argument("--foot-height-weight", type=float)
    # Eval
    parser.add_argument("--eval-freq", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=5)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.monitor import Monitor
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing RL dependencies. Run: uv sync --extra rl") from exc

    AlgoCls = PPO if args.algo == "ppo" else SAC

    def build_algo_kwargs(env, *, lr):
        if args.algo == "ppo":
            return dict(
                policy="MlpPolicy", env=env, seed=args.seed,
                learning_rate=lr, gamma=args.gamma,
                n_steps=args.n_steps, batch_size=args.batch_size,
                n_epochs=args.n_epochs, gae_lambda=args.gae_lambda,
                clip_range=args.clip_range, ent_coef=args.ent_coef,
                vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm,
                target_kl=args.target_kl,
                policy_kwargs={"log_std_init": args.log_std_init},
                verbose=1,
            )
        return dict(
            policy="MlpPolicy", env=env, seed=args.seed,
            learning_rate=lr, gamma=args.gamma,
            buffer_size=args.sac_buffer_size, batch_size=args.batch_size,
            learning_starts=args.sac_learning_starts,
            train_freq=args.sac_train_freq, gradient_steps=args.sac_gradient_steps,
            tau=args.sac_tau, ent_coef="auto",
            verbose=1,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lr = make_lr_schedule(args.learning_rate, args.lr_schedule)
    model = None
    if args.load_model is not None:
        custom_objects = {"learning_rate": lr, "gamma": args.gamma}
        if args.algo == "ppo":
            custom_objects.update({
                "clip_range": args.clip_range,
                "ent_coef": args.ent_coef,
                "vf_coef": args.vf_coef,
                "max_grad_norm": args.max_grad_norm,
            })
        model = AlgoCls.load(args.load_model, custom_objects=custom_objects)
        if args.algo == "ppo":
            model.n_epochs = args.n_epochs
            model.target_kl = args.target_kl

    for stage in args.stages:
        if stage not in STAND_STAGE_CONFIGS:
            raise SystemExit(f"Unknown stage {stage}; choose from 1, 2, 3")
        config = STAND_STAGE_CONFIGS[stage]
        config_overrides = {
            field: value
            for field, value in {
                "torque_cost": args.torque_cost,
                "action_rate_cost": args.action_rate_cost,
                "joint_speed_cost": args.joint_speed_cost,
                "posture_weight": args.posture_weight,
                "stillness_weight": args.stillness_weight,
                "angular_stillness_weight": args.angular_stillness_weight,
                "upright_weight": args.upright_weight,
                "height_tracking_weight": args.height_tracking_weight,
                "fall_penalty": args.fall_penalty,
                "gravity_scale": args.gravity_scale,
                "reset_noise": args.reset_noise,
                "episode_seconds": args.episode_seconds,
                "xy_drift_weight": args.xy_drift_weight,
                "foot_height_weight": args.foot_height_weight,
            }.items()
            if value is not None
        }
        if args.healthy_height_min is not None or args.healthy_height_max is not None:
            config_overrides["healthy_height"] = (
                args.healthy_height_min
                if args.healthy_height_min is not None
                else config.healthy_height[0],
                args.healthy_height_max
                if args.healthy_height_max is not None
                else config.healthy_height[1],
            )
        if config_overrides:
            config = replace(config, **config_overrides)
        env = Monitor(MatlasStandEnv(config, seed=args.seed + stage))

        if model is None:
            model = AlgoCls(**build_algo_kwargs(env, lr=lr))
        else:
            model.set_env(env)

        print(f"training_stage={stage} algo={args.algo} timesteps={args.timesteps_per_stage}")
        if args.algo == "ppo":
            print(
                f"ppo lr={args.learning_rate}({args.lr_schedule}) clip={args.clip_range} "
                f"ent={args.ent_coef} vf={args.vf_coef} max_grad={args.max_grad_norm} "
                f"n_steps={args.n_steps} batch={args.batch_size} epochs={args.n_epochs} "
                f"gamma={args.gamma} gae={args.gae_lambda} target_kl={args.target_kl} "
                f"log_std_init={args.log_std_init}"
            )
        else:
            print(
                f"sac lr={args.learning_rate}({args.lr_schedule}) buffer={args.sac_buffer_size} "
                f"batch={args.batch_size} starts={args.sac_learning_starts} "
                f"train_freq={args.sac_train_freq} grad_steps={args.sac_gradient_steps} "
                f"tau={args.sac_tau} gamma={args.gamma}"
            )
        print(f"config={config}")
        callback = None
        if args.eval_freq > 0:
            eval_env = Monitor(MatlasStandEnv(config, seed=args.seed + 10_000 + stage))
            callback = EvalCallback(
                eval_env,
                best_model_save_path=str(out_dir / f"stage_{stage}_best"),
                log_path=str(out_dir / f"stage_{stage}_eval"),
                eval_freq=args.eval_freq,
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
                render=False,
            )

        model.learn(
            total_timesteps=args.timesteps_per_stage,
            reset_num_timesteps=False,
            callback=callback,
        )
        stage_out = out_dir / f"stage_{stage}"
        model.save(stage_out)
        print(f"saved={stage_out}")

    if model is not None:
        final_out = out_dir / "final"
        model.save(final_out)
        print(f"saved={final_out}")


if __name__ == "__main__":
    main()
