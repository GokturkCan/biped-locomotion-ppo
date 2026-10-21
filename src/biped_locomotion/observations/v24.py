"""V24 append-only actor and privileged observation contract."""

import numpy as np
from .common import vector

ACTOR_SIZE, CRITIC_SIZE = 57, 82
GROUPS = {
    "joint_relative_position": (0, 12),
    "joint_velocity": (12, 24),
    "up": (24, 27),
    "body_velocity": (27, 30),
    "angular_velocity": (30, 33),
    "phase": (33, 35),
    "previous_action": (35, 47),
    "contacts": (47, 49),
    "command": (49, 52),
    "path_heading_imbalance": (52, 57),
}


def assemble(
    core,
    privileged_core,
    path_error_y,
    heading_error,
    last_left,
    last_right,
    pair_valid,
    previous_abs_imbalance,
):
    b = pair_valid * (last_right - last_left) / (last_right + last_left + 1e-6)
    tail = np.array(
        [
            path_error_y / 0.10,
            np.sin(heading_error),
            np.cos(heading_error),
            b,
            pair_valid,
        ]
    )
    return {
        "state": np.concatenate([vector(core, 52, "actor prefix"), tail]),
        "privileged_state": np.concatenate(
            [
                vector(privileged_core, 74, "critic prefix"),
                tail,
                [last_left, last_right, previous_abs_imbalance],
            ]
        ),
    }
