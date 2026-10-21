"""Public U13 shape helper; live path/error/quiet features use extracted kernels."""

import numpy as np
from .common import vector

ACTOR_SIZE, CRITIC_SIZE = 64, 89
PINNED_INDICES = (49, 50, 51, 57, 58, 59, 60)


def append_unified_context(v24_actor, v24_critic, context):
    """Assemble U13 context with the runtime critic order, then append reserved zeros.

    Callers must compute U13's rewritten V24 prefix with the source observation
    kernels; simply appending to a forward-only observation does not do that.
    """
    context = vector(context, 4, "unified context")
    actor_prefix = vector(v24_actor, 57, "actor")
    critic_v24_layout = vector(v24_critic, 82, "critic")
    return {
        "state": np.concatenate([actor_prefix, context, np.zeros(3)]),
        "privileged_state": np.concatenate(
            [
                actor_prefix,
                context,
                critic_v24_layout[52:74],
                critic_v24_layout[79:82],
                np.zeros(3),
            ]
        ),
    }
