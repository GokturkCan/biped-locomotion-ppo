"""Pure public actor/critic assembly using runtime state, not robot constants."""

import numpy as np


def vector(value, n, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite shape ({n},)")
    return a


def core_actor(
    q,
    reference,
    qvel,
    up,
    velocity_body,
    angular_velocity,
    phase,
    previous_action,
    contacts,
    command,
):
    return np.concatenate(
        [
            vector(q, 12, "q") - vector(reference, 12, "reference"),
            vector(qvel, 12, "qvel"),
            vector(up, 3, "up"),
            vector(velocity_body, 3, "velocity_body"),
            vector(angular_velocity, 3, "angular_velocity"),
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
            vector(previous_action, 12, "previous_action"),
            vector(contacts, 2, "contacts"),
            vector(command, 3, "command"),
        ]
    )


def privileged_prefix(
    core, contacts, velocity_world, angular_world, effort_normalized, knees
):
    return np.concatenate(
        [
            vector(core, 52, "core"),
            vector(contacts, 2, "contacts"),
            vector(velocity_world, 3, "velocity_world"),
            vector(angular_world, 3, "angular_world"),
            vector(effort_normalized, 12, "effort_normalized"),
            vector(knees, 2, "knees"),
        ]
    )
