"""Generalized actuator effort analysis. Only qfrc_actuator is accepted."""

import numpy as np


def summarize_effort(qfrc_actuator, dof_indices, joint_types, sample_rate_hz=500.0):
    a = np.asarray(qfrc_actuator, dtype=float)
    if (
        a.ndim != 2
        or a.shape[0] == 0
        or not np.isfinite(a).all()
        or sample_rate_hz <= 0
    ):
        raise ValueError("finite nonempty generalized-effort stream required")
    if len(dof_indices) != len(joint_types):
        raise ValueError("joint type/mapping mismatch")
    rows = []
    for i, (dof, kind) in enumerate(zip(dof_indices, joint_types)):
        if kind not in ("hinge", "slide"):
            raise ValueError("hinge or slide required")
        if not 0 <= dof < a.shape[1]:
            raise ValueError("DOF outside qfrc_actuator")
        v = a[:, dof]
        rows.append(
            {
                "channel": f"channel_{i:02d}",
                "joint_type": kind,
                "unit": "N" if kind == "slide" else "N*m",
                "sample_count": len(v),
                "sample_rate_hz": sample_rate_hz,
                "rms": float(np.sqrt(np.mean(v * v))),
                "p99_absolute": float(np.percentile(np.abs(v), 99)),
                "peak_absolute": float(np.max(np.abs(v))),
                "effort_source": "qfrc_actuator",
            }
        )
    return rows
