"""Explicit physical criteria; episode reward alone cannot qualify walking."""

REQUIRED = (
    "displacement_m",
    "mean_body_vx_m_s",
    "liftoff_left",
    "liftoff_right",
    "touchdown_left",
    "touchdown_right",
    "single_support_fraction",
    "support_transfer",
    "same_side_repeat",
    "unexpected_side",
    "transfer_timeout",
    "falls",
    "action_saturation",
    "ctrl_saturation",
    "force_saturation",
)


def qualify(metrics):
    missing = set(REQUIRED) - set(metrics)
    if missing:
        raise ValueError(f"missing physical evidence: {sorted(missing)}")
    progress = metrics["displacement_m"] > 0 and metrics["mean_body_vx_m_s"] > 0
    gait = all(
        metrics[k] > 0
        for k in (
            "liftoff_left",
            "liftoff_right",
            "touchdown_left",
            "touchdown_right",
            "single_support_fraction",
            "support_transfer",
        )
    )
    valid = all(
        metrics[k] == 0
        for k in (
            "same_side_repeat",
            "unexpected_side",
            "transfer_timeout",
            "falls",
            "action_saturation",
            "ctrl_saturation",
            "force_saturation",
        )
    )
    return bool(progress and gait and valid)
