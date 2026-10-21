# Resolved PPO configuration

The public profiles preserve the resolved training settings, rather than stale comments
from intermediate experiments. Both use asymmetric, feedforward actor–critic networks
with SiLU hidden activations. The actor's 24 outputs describe 12 Gaussian locations and
12 raw scales; they are not 24 actuators.

| Setting | V24 | U13 |
|---|---|---|
| Actor | 57 → 256 → 256 → 128 → 24 | 64 → 256 → 256 → 128 → 24 |
| Critic | 82 → 256 → 256 → 128 → 1 | 89 → 256 → 256 → 128 → 1 |
| Parallel environments | 1024 | 1024 |
| Unroll | 10 | 20 |
| Batch size | 256 | 128 |
| Minibatches | 16 | 16 |
| Updates per batch | 4 | 4 |
| Learning rate | 2e-4 | 2e-4 |
| Discount gamma | 0.97 | 0.97 |
| GAE lambda | 0.95 | 0.95 |
| Entropy coefficient | 0.01 | 0.01 |
| PPO clip epsilon | 0.30 | 0.20 |

Machine-readable profiles: [V24](../configs/v24_forward.yaml) and
[U13](../configs/u13_scratch.yaml). Config-only CLI parsing does not start PPO.
The external backend is Brax; this release does not vendor its trainer.

U13 reused environment/control software but began with a fresh actor, critic, optimizer
and normalizer, no V24 checkpoint restore and no frozen layers. Software reuse and
training initialization are separate claims.

[Observations](observation_space.md) · [Actions](action_space.md) ·
[Reproducibility](reproducibility.md) · [README](../README.md)
