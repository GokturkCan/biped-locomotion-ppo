"""Explicit-duration kinematic and saturation metrics."""

import numpy as np


def summarize(
    time_s,
    world_x,
    body_vx,
    command_vx,
    *,
    single_support=None,
    falls=0,
    action_saturation=0,
    ctrl_saturation=0,
    force_saturation=0,
):
    t, x, v, c = map(
        lambda a: np.asarray(a, dtype=float), (time_s, world_x, body_vx, command_vx)
    )
    if t.ndim != 1 or len(t) < 2 or any(a.shape != t.shape for a in (x, v, c)):
        raise ValueError("matched streams with at least two timestamps required")
    if not all(np.isfinite(a).all() for a in (t, x, v, c)) or np.any(np.diff(t) <= 0):
        raise ValueError("finite streams and strictly increasing time required")
    dt = np.diff(t)
    duration = t[-1] - t[0]
    # State i is the beginning of interval i. This convention is explicit.
    mean = lambda a: float(np.sum(a[:-1] * dt) / duration)
    out = {
        "duration_s": float(duration),
        "displacement_m": float(x[-1] - x[0]),
        "world_displacement_rate_m_s": float((x[-1] - x[0]) / duration),
        "mean_body_vx_m_s": mean(v),
        "mean_command_vx_m_s": mean(c),
        "mean_abs_vx_error_m_s": mean(np.abs(v - c)),
        "falls": int(falls),
        "action_saturation": action_saturation,
        "ctrl_saturation": ctrl_saturation,
        "force_saturation": force_saturation,
    }
    if single_support is not None:
        a = np.asarray(single_support, dtype=float)
        if a.shape != t.shape or not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
            raise ValueError("invalid single-support stream")
        out["single_support_fraction"] = mean(a)
    return out
