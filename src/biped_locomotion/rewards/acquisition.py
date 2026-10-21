"""Acquisition ladder and retained event/reward kernels."""

STAGES = (
    "weight_shift",
    "unloading",
    "liftoff",
    "no_contact_swing",
    "single_support",
    "touchdown",
    "load_acceptance",
    "support_transfer",
    "opposite_side",
    "completed_step",
)


def components(environment, data, info, phase_id, terms, motion):
    return environment._v9_components(data, info, phase_id, terms, motion)
