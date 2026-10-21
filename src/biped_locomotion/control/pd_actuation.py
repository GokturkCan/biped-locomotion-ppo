"""Affine-PD generalized effort, distinct from the normalized policy action."""

import numpy as np


def affine_pd(target, position, velocity, kp, kd, effort_limit):
    arrays = [np.asarray(x) for x in (target, position, velocity, kp, kd, effort_limit)]
    if any(a.shape != arrays[0].shape for a in arrays) or not all(
        np.isfinite(a).all() for a in arrays
    ):
        raise ValueError("finite matched PD vectors required")
    target, position, velocity, kp, kd, effort_limit = arrays
    if any(np.any(a < 0) for a in (kp, kd, effort_limit)):
        raise ValueError("negative gain/limit")
    return np.clip(
        kp * (target - position) - kd * velocity, -effort_limit, effort_limit
    )
