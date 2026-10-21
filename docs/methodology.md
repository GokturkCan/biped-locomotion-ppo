# Methodology

The research started without a validated forward gait for the custom biped.
The task is stable forward locomotion on flat ground, measured through displacement,
body-frame velocity, bilateral events and survival. Reward is an optimization signal,
not a certificate of walking.

The policy predicts normalized residual positions for twelve actuated joints. MuJoCo
converts bounded position targets into effort through affine PD. An asymmetric actor/critic
lets the value function use clean/global state and effort unavailable to the actor.
PPO uses a stochastic tanh-normal policy during collection and deterministic transformed
location (`tanh(location)`) for canonical deterministic evaluation. This is the Gaussian mean mapped through tanh, not an analytic expectation or mode
of the squashed distribution.

Acquisition progresses from weight shift and unloading to contact-qualified liftoff,
swing, actual single support, touchdown, load acceptance and support transfer.
Once initialized, the expected swing side flips on completed transfer, not on a raw
contact edge. Unknown-side bootstrap timeout recovery can initialize the side; this
is distinct from advancing an established alternating sequence.
V18 aligns the actor timing context with these physical event states.

This release separates research code from private model data. Model-dependent behavior
has not been revalidated after extraction; public checks cover software contracts only.
