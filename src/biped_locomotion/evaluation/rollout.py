"""Opt-in bounded private-model policy evaluation. No rendering or automatic runs."""

import numpy as np
from .metrics import summarize

EVENT_KEYS = {
    "liftoff_left": "transition/qualified_liftoff_left",
    "liftoff_right": "transition/qualified_liftoff_right",
    "touchdown_left": "transition/qualified_touchdown_left",
    "touchdown_right": "transition/qualified_touchdown_right",
    "support_transfer": "transition/support_transfer_completion_count",
    "same_side_repeat": "transition/qualified_same_side_repeat_count",
    "unexpected_side": "transition/unexpected_side_qualified_touchdown_count",
    "transfer_timeout": "transition/support_transfer_timeout_count",
}


def rollout(environment, policy, rng, num_steps, *, max_steps=100):
    """Policy(obs,key)->(action,extras). Caller chooses deterministic/stochastic policy.

    Unwrapped single episode, stops on done, no autoreset. This function is not called
    by public checks. Returned scalars are private evidence until reviewed.
    """
    if not isinstance(num_steps, int) or not 1 <= num_steps <= max_steps:
        raise ValueError("explicit bounded step count required")
    import jax

    state = environment.reset(rng)
    root = lambda s: float(environment.get_root_pos(s.data)[0])
    velocity = lambda s: float(environment.get_local_linvel(s.data)[0])
    xs = [root(state)]
    vs = [velocity(state)]
    cs = [float(state.info["command"][0])]
    ss = []
    events = {k: 0.0 for k in EVENT_KEYS}
    saturation = {
        k: 0.0 for k in ("action_saturation", "ctrl_saturation", "force_saturation")
    }
    step = jax.jit(environment.step)
    falls = 0
    for _ in range(num_steps):
        rng, key = jax.random.split(rng)
        action, _ = policy(state.obs, key)
        left, right = environment._foot_contacts(state.data)
        ss.append(float(bool(left) != bool(right)))
        state = step(state, action)
        xs.append(root(state))
        vs.append(velocity(state))
        cs.append(float(state.info["command"][0]))
        for key, name in EVENT_KEYS.items():
            events[key] += float(state.metrics[name])
        for key in saturation:
            saturation[key] += float(state.metrics["control/" + key])
        falls += int(float(state.metrics["termination/fall"]) > 0)
        if bool(state.done):
            break
    ss.append(ss[-1])
    times = np.arange(len(xs)) * environment.dt
    out = summarize(times, xs, vs, cs, single_support=ss, falls=falls, **saturation)
    out.update(events)
    out["maximum_same_side_sequence"] = float(
        state.info["v9_max_same_side_qualified_touchdown_streak"]
    )
    out["saturation_semantics"] = "sum of per-tick channel fractions"
    return out
