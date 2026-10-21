"""Residual joint-position targets; all model-dependent arrays are arguments."""

import numpy as np


def effective_scale(scale, rho, swing_mask, gate, maximum):
    scale, maximum, mask = map(np.asarray, (scale, maximum, swing_mask))
    if scale.shape != maximum.shape or mask.shape != scale.shape:
        raise ValueError("scale, maximum and swing mask must have matching shapes")
    if not 0 <= rho <= 1 or not 0 <= gate <= 1 or np.any((mask < 0) | (mask > 1)):
        raise ValueError("authority coefficients must be in [0,1]")
    return rho * scale + mask * gate * (maximum - rho * scale)


def residual_target(action, reference, scale, lower, upper):
    action, reference, scale, lower, upper = map(
        np.asarray, (action, reference, scale, lower, upper)
    )
    if action.shape != reference.shape or any(
        x.shape != action.shape for x in (scale, lower, upper)
    ):
        raise ValueError("all control vectors must have matching shapes")
    if not all(
        np.isfinite(x).all() for x in (action, reference, scale, lower, upper)
    ) or np.any(lower > upper):
        raise ValueError("invalid control vectors")
    action = np.clip(action, -1, 1)
    raw = reference + scale * action
    target = np.clip(raw, lower, upper)
    return {
        "action": action,
        "target": target,
        "requested_residual": scale * action,
        "effective_residual": target - reference,
        "target_clipped": bool(np.any(raw != target)),
    }
