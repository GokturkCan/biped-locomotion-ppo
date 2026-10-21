# Action space

The policy controls 12 normalized residual joint-position channels. The actor distribution
head emits 24 values: **12 location/mean parameters and 12 scale logits**, not 24 actuators.
Brax's tanh-normal policy samples an unsquashed Gaussian and applies tanh. Scale is
`softplus(raw_scale)+0.001`. Deterministic inference uses `tanh(location)`: the unsquashed Gaussian mean is
mapped through tanh. This is not the analytic mean of the squashed distribution.

Targets are offset from a runtime reference and clipped to runtime joint bounds.
Revolute residuals use radians; prismatic residuals use metres. Channel identities,
scales, gains and limits are private runtime bindings. Public arrays are synthetic only.

V24's active source control assumes the policy already produces bounded actions; U13
additionally clips executed actions. The public standalone helper explicitly clips input
for contract use and is not substituted into the V24 source kernel.
