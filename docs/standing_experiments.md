# Standing Policy Experiments

Stage 1 setup unless noted otherwise:

- Robot: full assembly with leg-only control.
- Control: direct normalized torque actions, no Python PD controller.
- Gravity: `0.1 * 9.81`.
- Episode length: 6 seconds at `control_dt=0.02`, max 300 env steps.
- Success criterion for now: deterministic evaluation should survive substantially longer than the training rollout mean, because deployment will usually use the policy mean.

## Results

| ID | Command / setting summary | Train timesteps | Final train rollout | Deterministic eval | Stochastic eval | Notes |
| --- | --- | ---: | --- | --- | --- | --- |
| `baseline_100k` | Default PPO: lr `3e-4`, clip `0.2`, ent `0.005`, log std init `-4` | 100k | `ep_len_mean=38.9`, `ep_rew_mean=135` | `mean_length=22.0`, `max=22`, `mean_reward=86.689` | not run | First leg-only direct run. Better than the earlier PD/all-joint run, but still short horizon. |
| `baseline_1m` | Default PPO: lr `3e-4`, clip `0.2`, ent `0.005`, log std init `-4` | 1M | `ep_len_mean=76.7`, `ep_rew_mean=239` | `mean_length=4.0`, `max=4`, `mean_reward=6.590` | `mean_length=23.6`, `max=87`, `lengths=[4,4,87,20,3]` | Training rollout improved, but deterministic policy collapsed. KL and clip fraction stayed high, so the updates are too aggressive and/or entropy is masking a weak mean policy. |

## Active Sweep

The next runs reduce PPO update size and entropy pressure. The goal is not just higher training reward; it is a deterministic mean policy that stands longer.

| ID | Hypothesis | Command | Status | Result |
| --- | --- | --- | --- | --- |
| `s1_lr1e4_clip01_ent0_500k` | Smaller updates and no entropy bonus should stabilize the policy mean. | `uv run python scripts/train_stand_stages.py --timesteps-per-stage 500000 --stages 1 --out-dir runs/s1_lr1e4_clip01_ent0_500k --learning-rate 0.0001 --clip-range 0.1 --ent-coef 0.0 --log-std-init -4.0` | done | Final train: `ep_len_mean=69.3`, `ep_rew_mean=235`, KL ~0.079. Deterministic eval: `mean_length=105`, `max=105`, lengths all `105`. Stochastic eval: `mean_length=88.5`, `max=168`. Best combined deterministic/stochastic result so far. |
| `s1_lr5e5_clip005_ent0_500k` | More conservative PPO updates should reduce KL spikes. | `uv run python scripts/train_stand_stages.py --timesteps-per-stage 500000 --stages 1 --out-dir runs/s1_lr5e5_clip005_ent0_500k --learning-rate 0.00005 --clip-range 0.05 --ent-coef 0.0 --log-std-init -4.0` | done | Final train: `ep_len_mean=70`, `ep_rew_mean=252`, KL ~0.008. Deterministic eval: `mean_length=103`, `max=103`, lengths all `103`. Stochastic eval: `mean_length=60.6`, `max=81`. Best deterministic policy so far. |
| `s1_lr1e4_clip01_smooth_500k` | Extra action-rate and torque penalties may produce a less brittle mean action. | `uv run python scripts/train_stand_stages.py --timesteps-per-stage 500000 --stages 1 --out-dir runs/s1_lr1e4_clip01_smooth_500k --learning-rate 0.0001 --clip-range 0.1 --ent-coef 0.0 --log-std-init -4.0 --torque-cost 0.01 --action-rate-cost 0.08` | done | Final train: `ep_len_mean=68.8`, `ep_rew_mean=237`, KL ~0.061. Deterministic eval: `mean_length=50`, `max=50`, lengths all `50`. Stochastic eval: `mean_length=50.5`, `max=102`. Smoothness penalties did not beat conservative PPO. |

## Observations

- Direct torque control with only the legs learns faster than the earlier software-PD/all-joint setup in stage 1.
- The default PPO configuration is too volatile for this problem. High KL and high clip fraction persist through training.
- Entropy pressure is likely counterproductive for the current goal because the stochastic rollout can look better while the deterministic mean policy remains unusable.
- Lower learning rate and clip range improved deterministic evaluation much more than adding larger torque/action-rate penalties.
- Periodic deterministic eval checkpoints were added after the first sweep so later runs can retain the best policy even if the final update regresses.

## Next Sweep

| ID | Hypothesis | Command | Status | Result |
| --- | --- | --- | --- | --- |
| `s1_lr1e4_clip01_ent0_1m_eval` | The best 500k setting may keep improving with more steps if best checkpoints guard against late regressions. | `uv run python scripts/train_stand_stages.py --timesteps-per-stage 1000000 --stages 1 --out-dir runs/s1_lr1e4_clip01_ent0_1m_eval --learning-rate 0.0001 --clip-range 0.1 --ent-coef 0.0 --log-std-init -4.0 --eval-freq 25000 --eval-episodes 5` | done | See 1M sweep results below. |
| `s1_lr5e5_clip005_ent0_1m_eval` | The most stable-KL run may overtake the moderate run with more training. | `uv run python scripts/train_stand_stages.py --timesteps-per-stage 1000000 --stages 1 --out-dir runs/s1_lr5e5_clip005_ent0_1m_eval --learning-rate 0.00005 --clip-range 0.05 --ent-coef 0.0 --log-std-init -4.0 --eval-freq 25000 --eval-episodes 5` | done | See 1M sweep results below. |
| `s1_lr1e4_clip005_ent0_1m_eval` | Keep the faster learning rate but restrict policy updates more tightly. | `uv run python scripts/train_stand_stages.py --timesteps-per-stage 1000000 --stages 1 --out-dir runs/s1_lr1e4_clip005_ent0_1m_eval --learning-rate 0.0001 --clip-range 0.05 --ent-coef 0.0 --log-std-init -4.0 --eval-freq 25000 --eval-episodes 5` | done | See 1M sweep results below. |

## 1M Sweep Results

| ID | Final train rollout | Final deterministic eval | Best deterministic eval | Best stochastic eval | Notes |
| --- | --- | --- | --- | --- | --- |
| `s1_lr1e4_clip01_ent0_1m_eval` | `ep_len_mean=91`, `ep_rew_mean=309`; eval callback final `88` steps | Final: `mean_length=105`, `mean_reward=349.461` | Best: `mean_length=126`, `mean_reward=457.501` | not run | Longer training improved over the 500k version only when using the best checkpoint. |
| `s1_lr5e5_clip005_ent0_1m_eval` | `ep_len_mean=94.5`, `ep_rew_mean=323`; eval callback final `189` steps | Final: `mean_length=103`, `mean_reward=392.197` | Best: `mean_length=184`, `mean_reward=695.396` | Best stochastic: `mean_length=72.2`, `max=113` | Best policy so far. Low KL remains useful even with longer training. |
| `s1_lr1e4_clip005_ent0_1m_eval` | `ep_len_mean=113`, `ep_rew_mean=408`; eval callback final `116` steps | Final: `mean_length=114`, `mean_reward=421.416` | Best: `mean_length=176`, `mean_reward=590.299` | Best stochastic: `mean_length=62.9`, `max=91` | Best rollout mean, second-best deterministic checkpoint. Higher clipping fraction than the `5e-5` run. |

## Fine-Tune Sweep

Starting checkpoint: `runs/s1_lr5e5_clip005_ent0_1m_eval/stage_1_best/best_model`, currently the best deterministic stage 1 policy.

| ID | Hypothesis | Command | Status | Result |
| --- | --- | --- | --- | --- |
| `s1_ft_best_lr2e5_clip003_500k` | Fine-tune the best checkpoint with smaller updates to improve deterministic survival without regressing. | `uv run python scripts/train_stand_stages.py --load-model runs/s1_lr5e5_clip005_ent0_1m_eval/stage_1_best/best_model --timesteps-per-stage 500000 --stages 1 --out-dir runs/s1_ft_best_lr2e5_clip003_500k --learning-rate 0.00002 --clip-range 0.03 --ent-coef 0.0 --log-std-init -4.0 --eval-freq 25000 --eval-episodes 5` | invalid | Hyperparameter override bug; see correction table. |
| `s1_ft_best_lr5e5_clip003_500k` | Keep the previous learning rate but tighten clipping during fine-tune. | `uv run python scripts/train_stand_stages.py --load-model runs/s1_lr5e5_clip005_ent0_1m_eval/stage_1_best/best_model --timesteps-per-stage 500000 --stages 1 --out-dir runs/s1_ft_best_lr5e5_clip003_500k --learning-rate 0.00005 --clip-range 0.03 --ent-coef 0.0 --log-std-init -4.0 --eval-freq 25000 --eval-episodes 5` | invalid | Hyperparameter override bug; see correction table. |
| `s1_ft_best_noise001_lr2e5_clip003_500k` | Add small reset noise to start training robustness around the nominal stand. | `uv run python scripts/train_stand_stages.py --load-model runs/s1_lr5e5_clip005_ent0_1m_eval/stage_1_best/best_model --timesteps-per-stage 500000 --stages 1 --out-dir runs/s1_ft_best_noise001_lr2e5_clip003_500k --learning-rate 0.00002 --clip-range 0.03 --ent-coef 0.0 --log-std-init -4.0 --reset-noise 0.001 --eval-freq 25000 --eval-episodes 5` | invalid | Hyperparameter override bug; see correction table. |

### Fine-Tune Correction

The first fine-tune sweep exposed a trainer issue: SB3 reloads the saved learning-rate and clip schedules from the checkpoint. Until `PPO.load(..., custom_objects=...)` was added, the requested fine-tune hyperparameters were ignored and the runs continued with the original `5e-5 / 0.05` schedule. Treat the following as continuation results, not as valid tests of the requested `2e-5 / 0.03` settings.

| ID | Actual setting | Result |
| --- | --- | --- |
| `s1_ft_best_lr2e5_clip003_500k` | Continued from best checkpoint with actual `lr=5e-5`, `clip=0.05`, no reset noise. | Best deterministic eval: `mean_length=199`, `mean_reward=702.529`. Reset-noise `0.001` eval: `mean_length=134.4`, `max=300`, lengths `[147,126,113,300,86,102,162,121,70,117]`. |
| `s1_ft_best_lr5e5_clip003_500k` | Same actual schedule as above, because checkpoint values were reused. | Best deterministic eval: `mean_length=199`, `mean_reward=702.529`. Duplicate nominal result due same seed and actual hyperparameters. |
| `s1_ft_best_noise001_lr2e5_clip003_500k` | Continued from best checkpoint with actual `lr=5e-5`, `clip=0.05`, reset noise `0.001`. | Best deterministic eval at nominal reset: `mean_length=100`, `mean_reward=346.532`. Reset-noise `0.001` eval: `mean_length=112.9`, `max=169`. More consistent but lower ceiling. |

The trainer now uses `custom_objects` during `PPO.load`, so subsequent fine-tunes can actually change the PPO schedules.

## Corrected Fine-Tune Sweep

Starting checkpoint: `runs/s1_ft_best_lr2e5_clip003_500k/stage_1_best/best_model`, the current best nominal stage 1 policy.

| ID | Hypothesis | Command | Status | Result |
| --- | --- | --- | --- | --- |
| `s1_ft199_lr2e5_clip003_300k` | Smaller true updates may improve beyond 199 deterministic steps without destabilizing. | `uv run python scripts/train_stand_stages.py --load-model runs/s1_ft_best_lr2e5_clip003_500k/stage_1_best/best_model --timesteps-per-stage 300000 --stages 1 --out-dir runs/s1_ft199_lr2e5_clip003_300k --learning-rate 0.00002 --clip-range 0.03 --ent-coef 0.0 --log-std-init -4.0 --eval-freq 25000 --eval-episodes 5` | done | Best checkpoint: deterministic `mean_length=190`, `mean_reward=676.497`. Did not beat the 199-step starting policy. |
| `s1_ft199_noise001_lr2e5_clip003_300k` | Fine-tuning the 199-step policy with reset noise may improve robustness around nominal stand. | `uv run python scripts/train_stand_stages.py --load-model runs/s1_ft_best_lr2e5_clip003_500k/stage_1_best/best_model --timesteps-per-stage 300000 --stages 1 --out-dir runs/s1_ft199_noise001_lr2e5_clip003_300k --learning-rate 0.00002 --clip-range 0.03 --ent-coef 0.0 --log-std-init -4.0 --reset-noise 0.001 --eval-freq 25000 --eval-episodes 5` | done | Best checkpoint nominal eval: `mean_length=83`, `mean_reward=307.502`. Reset-noise `0.001` eval: `mean_length=111.7`, `max=188`. Improved consistency under noise versus nominal behavior but reduced nominal performance. |

## Current Best

- Best nominal deterministic policy: `runs/s1_ft_best_lr2e5_clip003_500k/stage_1_best/best_model`, with `mean_length=199` at zero reset noise.
- Best noisy-reset check so far: the same policy evaluated with `--reset-noise 0.001`, `mean_length=134.4`, `max=300`.
- Fine-tuning directly from the 199-step policy with very small PPO updates did not improve it. The next useful branch is likely reward/curriculum work rather than smaller PPO updates: longer healthy-height tolerance, explicit foot contact/stability terms, or a gradual reset-noise curriculum.

## Shaped Stability Reward Sweep

Added optional reward terms:

- `xy_drift_weight`: rewards keeping base XY close to reset XY.
- `foot_height_weight`: rewards keeping left and right foot body heights matched.

Starting checkpoint: `runs/s1_ft_best_lr2e5_clip003_500k/stage_1_best/best_model`.

| ID | Hypothesis | Command | Status | Result |
| --- | --- | --- | --- | --- |
| `s1_shape_xy02_foot02_300k` | Penalizing drift and asymmetric foot height may extend nominal deterministic survival. | `uv run python scripts/train_stand_stages.py --load-model runs/s1_ft_best_lr2e5_clip003_500k/stage_1_best/best_model --timesteps-per-stage 300000 --stages 1 --out-dir runs/s1_shape_xy02_foot02_300k --learning-rate 0.00002 --clip-range 0.03 --ent-coef 0.0 --xy-drift-weight 0.2 --foot-height-weight 0.2 --eval-freq 25000 --eval-episodes 5` | done | Best checkpoint nominal eval: `mean_length=112`, `mean_reward=450.847`. Rejected: worse than current best 199. |
| `s1_shape_noise001_xy02_foot02_300k` | Same shaping with reset noise may improve noisy robustness without collapsing nominal behavior. | `uv run python scripts/train_stand_stages.py --load-model runs/s1_ft_best_lr2e5_clip003_500k/stage_1_best/best_model --timesteps-per-stage 300000 --stages 1 --out-dir runs/s1_shape_noise001_xy02_foot02_300k --learning-rate 0.00002 --clip-range 0.03 --ent-coef 0.0 --reset-noise 0.001 --xy-drift-weight 0.2 --foot-height-weight 0.2 --eval-freq 25000 --eval-episodes 5` | done | Best checkpoint nominal eval: `mean_length=118`, `mean_reward=466.408`. Reset-noise `0.001` eval: `mean_length=106.6`, `max=147`. Rejected: shaping lowered both nominal and noisy-reset performance. |

Observation: these dense shaping terms increased numeric reward but shortened survival. The policy likely learned to optimize posture/position terms in a way that does not preserve the existing balance strategy. Keep these terms available, but do not use them in the current best curriculum.

## Height Recovery Sweep

Failure diagnosis for current best `runs/s1_ft_best_lr2e5_clip003_500k/stage_1_best/best_model`: deterministic nominal eval terminates at step `199` because base height reaches `0.2959`, just below stage 1 lower bound `0.30`; upright is still healthy (`0.994`) and max qvel is low (`2.26`). The next branch trains with a lower termination floor while increasing height/posture reward, then evaluates against the original strict floor.

| ID | Hypothesis | Command | Status | Result |
| --- | --- | --- | --- | --- |
| `s1_recover_h025_height2_posture2_300k` | Lower termination floor lets the policy experience/recover from sagging; stronger height/posture rewards should push it back above the strict eval floor. | `uv run python scripts/train_stand_stages.py --load-model runs/s1_ft_best_lr2e5_clip003_500k/stage_1_best/best_model --timesteps-per-stage 300000 --stages 1 --out-dir runs/s1_recover_h025_height2_posture2_300k --learning-rate 0.00002 --clip-range 0.03 --ent-coef 0.0 --healthy-height-min 0.25 --height-tracking-weight 2.0 --posture-weight 2.0 --fall-penalty 12.0 --eval-freq 25000 --eval-episodes 5` | done | Best strict eval: `mean_length=116`, `mean_reward=440.431`. Reset-noise `0.001`: `mean_length=76.7`. Rejected: relaxed training floor did not transfer back to strict stage 1. |

## Conservative Continuation From 199-Step Policy

Starting checkpoint: `runs/s1_ft_best_lr2e5_clip003_500k/stage_1_best/best_model`.

| ID | Hypothesis | Command | Status | Result |
| --- | --- | --- | --- | --- |
| `s1_continue199_lr5e5_clip005_500k` | The prior improvement from 184 to 199 came from the conservative `5e-5 / 0.05` schedule; continuing it may improve nominal survival further. | `uv run python scripts/train_stand_stages.py --load-model runs/s1_ft_best_lr2e5_clip003_500k/stage_1_best/best_model --timesteps-per-stage 500000 --stages 1 --out-dir runs/s1_continue199_lr5e5_clip005_500k --learning-rate 0.00005 --clip-range 0.05 --ent-coef 0.0 --eval-freq 25000 --eval-episodes 5` | done | Best checkpoint nominal eval: `mean_length=148`, `mean_reward=533.298`. Reset-noise `0.001`: `mean_length=130.7`, `max=151`. Rejected: continued training with the same schedule regressed nominal survival. |
| `s1_continue199_lr5e5_clip003_500k` | Same learning rate with tighter clipping may preserve the 199-step behavior while still improving it. | `uv run python scripts/train_stand_stages.py --load-model runs/s1_ft_best_lr2e5_clip003_500k/stage_1_best/best_model --timesteps-per-stage 500000 --stages 1 --out-dir runs/s1_continue199_lr5e5_clip003_500k --learning-rate 0.00005 --clip-range 0.03 --ent-coef 0.0 --eval-freq 25000 --eval-episodes 5` | done | Best checkpoint nominal eval: `mean_length=196`, `mean_reward=707.141`. Close to the 199-step policy but did not improve it. |

## Low-Impact Stabilization Sweep

Starting checkpoint: `runs/s1_ft_best_lr2e5_clip003_500k/stage_1_best/best_model`.

The previous reward-shaping sweep used strong drift and foot-height terms and degraded survival. This sweep keeps changes smaller and avoids foot-height shaping.

| ID | Hypothesis | Command | Status | Result |
| --- | --- | --- | --- | --- |
| `s1_tiny_lr1e5_clip002_400k` | Very small PPO updates may preserve the 199-step behavior while finding a slight improvement. | `uv run python scripts/train_stand_stages.py --load-model runs/s1_ft_best_lr2e5_clip003_500k/stage_1_best/best_model --timesteps-per-stage 400000 --stages 1 --out-dir runs/s1_tiny_lr1e5_clip002_400k --learning-rate 0.00001 --clip-range 0.02 --ent-coef 0.0 --eval-freq 25000 --eval-episodes 5` | done | Best checkpoint nominal eval: `mean_length=149`, `mean_reward=550.664`. Reset-noise `0.001`: `mean_length=111.5`, `max=188`. Rejected. |
| `s1_light_height_xy_400k` | Light height and XY shaping may address the observed height/drop drift failure without overwhelming the existing balance strategy. | `uv run python scripts/train_stand_stages.py --load-model runs/s1_ft_best_lr2e5_clip003_500k/stage_1_best/best_model --timesteps-per-stage 400000 --stages 1 --out-dir runs/s1_light_height_xy_400k --learning-rate 0.00002 --clip-range 0.03 --ent-coef 0.0 --height-tracking-weight 1.2 --xy-drift-weight 0.05 --eval-freq 25000 --eval-episodes 5` | done | Best checkpoint nominal eval: `mean_length=165`, `mean_reward=616.832`. Reset-noise `0.001`: `mean_length=111.0`, `max=132`. Rejected, though it was the best of this low-impact sweep. |
| `s1_noise0005_lr2e5_clip003_400k` | Smaller reset noise than the failed `0.001` run may improve robustness without collapsing nominal performance. | `uv run python scripts/train_stand_stages.py --load-model runs/s1_ft_best_lr2e5_clip003_500k/stage_1_best/best_model --timesteps-per-stage 400000 --stages 1 --out-dir runs/s1_noise0005_lr2e5_clip003_400k --learning-rate 0.00002 --clip-range 0.03 --ent-coef 0.0 --reset-noise 0.0005 --eval-freq 25000 --eval-episodes 5` | done | Best checkpoint nominal eval: `mean_length=132`, `mean_reward=506.737`. Reset-noise `0.001`: `mean_length=122.8`, `max=159`. Rejected for nominal performance. |

Observation: even small learning-rate/clipping changes move the deterministic mean policy away from the 199-step behavior. The trainer now exposes `--n-epochs` and `--target-kl` so loaded-policy fine-tunes can use fewer optimizer passes and early KL stopping.

## Low-Epoch / KL-Limited Fine-Tune Sweep

Starting checkpoint: `runs/s1_ft_best_lr2e5_clip003_500k/stage_1_best/best_model`.

| ID | Hypothesis | Command | Status | Result |
| --- | --- | --- | --- | --- |
| `s1_epochs1_lr2e5_clip003_300k` | One PPO epoch per rollout may reduce policy-mean drift from the 199-step checkpoint. | `uv run python scripts/train_stand_stages.py --load-model runs/s1_ft_best_lr2e5_clip003_500k/stage_1_best/best_model --timesteps-per-stage 300000 --stages 1 --out-dir runs/s1_epochs1_lr2e5_clip003_300k --learning-rate 0.00002 --clip-range 0.03 --ent-coef 0.0 --n-epochs 1 --eval-freq 25000 --eval-episodes 5` | done | Best checkpoint nominal eval: `mean_length=300`, `mean_reward=1034.520`; reached full 6-second stage-1 horizon. Reset-noise `0.001`: `mean_length=113.45`, `max=174`; reset-noise `0.005`: `mean_length=99.05`, `max=151`. New best nominal policy. |
| `s1_targetkl002_lr2e5_clip003_300k` | Early KL stopping may keep updates from stepping out of the good basin. | `uv run python scripts/train_stand_stages.py --load-model runs/s1_ft_best_lr2e5_clip003_500k/stage_1_best/best_model --timesteps-per-stage 300000 --stages 1 --out-dir runs/s1_targetkl002_lr2e5_clip003_300k --learning-rate 0.00002 --clip-range 0.03 --ent-coef 0.0 --target-kl 0.002 --eval-freq 25000 --eval-episodes 5` | done | Best checkpoint nominal eval: `mean_length=250`, `mean_reward=724.439`. Reset-noise `0.001`: `mean_length=120.85`, `max=176`. Good but not better than the one-epoch policy. |
| `s1_epochs3_targetkl002_lr5e5_clip003_300k` | Slightly higher learning rate with fewer epochs and KL stopping may preserve the policy while allowing useful adaptation. | `uv run python scripts/train_stand_stages.py --load-model runs/s1_ft_best_lr2e5_clip003_500k/stage_1_best/best_model --timesteps-per-stage 300000 --stages 1 --out-dir runs/s1_epochs3_targetkl002_lr5e5_clip003_300k --learning-rate 0.00005 --clip-range 0.03 --ent-coef 0.0 --n-epochs 3 --target-kl 0.002 --eval-freq 25000 --eval-episodes 5` | done | Best checkpoint nominal eval: `mean_length=128`, `mean_reward=492.406`. Rejected. |

Observation: reducing PPO to one epoch per rollout was the key improvement. It kept KL/clip fraction much lower and found a full-horizon nominal stage-1 standing policy. Robustness to reset perturbations remains weak.

## Robustness And Gravity Curriculum

Starting checkpoint: `runs/s1_epochs1_lr2e5_clip003_300k/stage_1_best/best_model`.

The trainer now exposes `--gravity-scale` so we can keep the same leg-direct action space while increasing gravity gradually instead of jumping to the default stage-2 all-joint PD setup.

| ID | Hypothesis | Command | Status | Result |
| --- | --- | --- | --- | --- |
| `s1_from300_noise001_epochs1_300k` | One-epoch PPO with reset noise may improve robustness while preserving the 300-step nominal policy. | `uv run python scripts/train_stand_stages.py --load-model runs/s1_epochs1_lr2e5_clip003_300k/stage_1_best/best_model --timesteps-per-stage 300000 --stages 1 --out-dir runs/s1_from300_noise001_epochs1_300k --learning-rate 0.00002 --clip-range 0.03 --ent-coef 0.0 --n-epochs 1 --reset-noise 0.001 --eval-freq 25000 --eval-episodes 5` | done | Best noisy-eval checkpoint: nominal `mean_length=113`, reset-noise `0.001` `mean_length=116.35`, `max=176`. Rejected: robustness did not improve enough and nominal full-horizon behavior was lost. |
| `s1_from300_g02_epochs1_300k` | A 0.2g curriculum step may preserve the learned stand while beginning the path toward full gravity. | `uv run python scripts/train_stand_stages.py --load-model runs/s1_epochs1_lr2e5_clip003_300k/stage_1_best/best_model --timesteps-per-stage 300000 --stages 1 --out-dir runs/s1_from300_g02_epochs1_300k --learning-rate 0.00002 --clip-range 0.03 --ent-coef 0.0 --n-epochs 1 --gravity-scale 0.2 --eval-freq 25000 --eval-episodes 5` | done | Best checkpoint evaluated at `gravity_scale=0.2`: `mean_length=98`, `mean_reward=365.051`. Too large a gravity jump from the 0.1g policy. |
| `s1_from300_g015_epochs1_300k` | A 0.15g step may be a gentler curriculum than jumping directly to 0.2g. | `uv run python scripts/train_stand_stages.py --load-model runs/s1_epochs1_lr2e5_clip003_300k/stage_1_best/best_model --timesteps-per-stage 300000 --stages 1 --out-dir runs/s1_from300_g015_epochs1_300k --learning-rate 0.00002 --clip-range 0.03 --ent-coef 0.0 --n-epochs 1 --gravity-scale 0.15 --eval-freq 25000 --eval-episodes 5` | done | Best checkpoint evaluated at `gravity_scale=0.15`: `mean_length=143`, `mean_reward=550.934`. Same checkpoint at `0.1g`: `mean_length=96`, so the gravity adaptation does not preserve the 0.1g full-horizon stand. |
| `s1_from300_g02_lr1e5_epochs1_300k` | A smaller learning rate at 0.2g may adapt without destroying the 0.1g stand. | `uv run python scripts/train_stand_stages.py --load-model runs/s1_epochs1_lr2e5_clip003_300k/stage_1_best/best_model --timesteps-per-stage 300000 --stages 1 --out-dir runs/s1_from300_g02_lr1e5_epochs1_300k --learning-rate 0.00001 --clip-range 0.03 --ent-coef 0.0 --n-epochs 1 --gravity-scale 0.2 --eval-freq 25000 --eval-episodes 5` | done | Best checkpoint evaluated at `gravity_scale=0.2`: `mean_length=92`, `mean_reward=351.382`. Lower learning rate did not improve the 0.2g curriculum. |

Current conclusion: the best usable artifact is still `runs/s1_epochs1_lr2e5_clip003_300k/stage_1_best/best_model` for nominal 0.1g stage-1 standing. One-epoch PPO is the important trainer setting. Reset-noise and gravity adaptation need a different curriculum, likely shorter gravity increments with checkpoint selection that includes both source and target gravity, or a multi-condition eval criterion instead of optimizing only the current training config.
