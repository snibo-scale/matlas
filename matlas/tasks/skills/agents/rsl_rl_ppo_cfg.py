"""RSL-RL PPO defaults for Matlas skill-curriculum tasks."""

from __future__ import annotations

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

_ITERATIONS = {
    "stand_balance": 1500,
    "squat": 1800,
    "loaded_squat": 2200,
    "single_step": 2500,
    "stair_step": 3000,
    "jump_forward": 3500,
    "back_flip": 6000,
}


def skill_ppo_runner_cfg(skill_name: str) -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.8,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.004,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=2.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name=f"matlas_{skill_name}",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=_ITERATIONS.get(skill_name, 3000),
    )

