# U13: gait acquisition without command-conditioned locomotion

U13 shares V24 environment/control software but starts with a fresh actor, critic, optimizer
and normalizer; no V24 restore and no frozen layers. Mean-head kernel/bias and scale-head
kernel are zeroed; scale biases use inverse softplus for the specified initial standard deviations.
Actor/critic dimensions are 64/89, with a feedforward shared network, not recurrent heads.

| Metric, 20 s horizon | Deterministic | Stochastic |
|---|---:|---:|
| Single support | 34.81% | 28.61% |
| Alternating touchdowns | 26.46 | 22.99 |
| Support transfers | 27.34 | 23.73 |
| Falls / repeats / timeouts | 0 / 0 / 0 | 0 / 0 / 0 |
| Net forward displacement | −0.032 m | +0.030 m |
| Mean body vx | −0.0021 m/s | +0.0012 m/s |

Deterministic mean command vx was +0.0486 m/s, with mean absolute vx error 0.0574 m/s.
The policy acquired a valid gait but failed forward tracking. Lateral/yaw tracking was
not adequately measured; backward/omnidirectional success is not claimed.

Reported positive reward mass: about 75.13% gait occupancy, 11.31% event completion,
6.61% quiet/standing, 5.54% task command tracking and 1.38% posture/survival.
Rounded positive contributions do not sum to net episode return, which includes penalties.
These figures support a shaping/task-balance interpretation, not a claim that valid events
were fabricated or that one architecture was causally disproven.

## Resolved PPO configuration

Actor `64 → 256 → 256 → 128 → 24`; critic `89 → 256 → 256 → 128 → 1`;
1024 environments, unroll 20, batch 128, 16 minibatches, 4 updates per batch,
learning rate 2e-4, gamma .97, GAE lambda .95, entropy coefficient .01 and
PPO clip **.20**. V24 uses clip .30; these are distinct resolved settings.
