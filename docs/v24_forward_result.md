# V24 canonical forward result

Historical checkpoint: `000003440640`. The canonical measurements are preserved in
`results/v24_summary.json` and were not replaced by a release-edition rollout.

| Quantity | Reported value |
|---|---:|
| Horizon | 60 s |
| Net world forward displacement | 6.9976 m |
| Mean body-frame forward velocity | 0.1151 m/s |
| Mean command velocity | 0.09997 m/s |
| Actual single support | 34.56% |
| Falls / transfer timeouts / unexpected-side events | 0 / 0 / 0 |
| Maximum same-side sequence | 1 |
| Action / control / force saturation | 0 / 0 / 0 |

Historical velocity reporting had a horizon-normalization issue. These values use the
resolved correction. Body-frame mean velocity is distinct from world displacement/time
(6.9976/60 ≈ 0.11663 m/s); they must not be substituted for one another.
The separate actuator-effort rollout has a different displacement and is not the canonical
training-evaluation result.

Architecture: actor 57→256→256→128→24; critic 82→256→256→128→1; SiLU;
1024 environments, unroll 10, batch 256, 16 minibatches, 4 updates, learning rate 2e-4,
gamma .97, GAE lambda .95, entropy .01, PPO clip **.30**.

This establishes simulated forward locomotion on the evaluated setup. It does not establish
hardware viability, seed robustness, push robustness, rough-terrain performance or
omnidirectional locomotion. Original training seed and exact launch parent path were not
persisted. The source suggests V23 warm start; exact historical initialization is not recovered.
