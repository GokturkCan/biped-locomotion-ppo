# Reward design

Dense forward velocity reward alone did not establish valid walking: policies could
collect return without a complete, physically validated gait. This motivated explicit
event-qualified acquisition shaping rather than treating return as proof of locomotion.

The acquisition ladder is weight shift → unloading → liftoff → no-contact swing → actual
single support → touchdown → load acceptance → support transfer → opposite-side transition
→ completed step. Contact edges are debounced using consecutive air/contact ticks,
height gain, opposite support and valid ground load. Transfer completion and timeouts are
one-shot events. Timers and expected-side memory are retained from source methods.

Forward additions address command tracking, contact drag, completed placement, heading,
bilateral placement imbalance and persistent path drift. V24's growth penalty requires a
previously valid bilateral pair; the first pair must not pay growth-from-zero.

Raw term, coefficient-weighted term and dt-scaled return are different quantities.
The runtime applies the original weighting, clipping and gating order. Public accounting
utilities expose all three meanings explicitly.

Configured does not mean runtime-active. `excessive_clearance` is unreachable on the active
V24 component path; the release sets its stale coefficient to zero. Inherited configuration
fields needed by setup/metrics remain for compatibility, not as claims of active rewards.

U13 restores acquisition shaping without a gait-success-conditioned activation gate.
A sampled motion command can access precursor rewards before its first completed step.
Its outcome demonstrates a reward-balance limitation: valid gait occupancy can dominate
command-conditioned translation. The evidence does not justify a universal causal claim.

Once the expected side is known, successful support transfer flips it to the opposite
side. In the unknown-side bootstrap state, a transfer timeout can initialize recovery
side memory; a raw touchdown is not sufficient to flip a known expected side. The
canonical zero-timeout result does not depend on invoking that recovery branch.
