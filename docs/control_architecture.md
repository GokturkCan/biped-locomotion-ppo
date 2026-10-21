# Control architecture

The general residual formulation is

```text
a_t ∈ [-1,1]^12
s_eff = rho*s + m_swing*g(phi)*(s_max-rho*s)
q_target = clip(q_ref(phi) + s_eff*a_t, q_min, q_max)
effort = clip(Kp*(q_target-q) - Kd*qdot, -effort_limit, effort_limit)
```

Joint limits, reference poses, gains, mappings and scales come from the authorized runtime
model/metadata. No public constants reconstruct the private robot.

## Active canonical V24 execution path

Source dispatch establishes that V24 inherits `ResidualForwardWalkingV2._q_ref_phase`
and `_v3_control_law`. The active reference is the episode's settled
`bridge_q_ref_base`; the residual scale is fixed. Earlier phase-conditioned reference
and authority algorithms are overridden. Thus the general equation specializes to
`q_ref(phi)=q_settled` and `s_eff=s`. The phase still affects observations, reward gates and
FSM timing. Calling the active V24 nominal pose a moving scripted joint trajectory would
misrepresent the source implementation.

The public general-purpose helpers expose phase-conditioned reference/authority math;
the optional V24/U13 runtime uses the extracted active override. This distinction is
intentional. The policy action is not torque. The affine-PD model performs actuation.

The dispatch table binds both profiles' `_q_ref_phase` and `_v3_control_law` to
`ResidualForwardWalkingV2` kernels. The V18 lifecycle calls those bindings before
physics. `_q_ref_phase` computes a phase-dependent unload weight for shaping but returns
`bridge_q_ref_base` (falling back to the runtime default pose) as the reference.
`_v3_control_law` ignores phase/side inputs and multiplies the action by the fixed
runtime scale. The broader equation is conceptual/earlier-lineage methodology, and
`phase_reference.py`/`effective_scale` are standalone helpers, not optional switches in
canonical V24. No algorithm was changed to make it match the conceptual equation.
