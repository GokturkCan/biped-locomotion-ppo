"""Fast CPU software-contract example; it is not a V24 reproduction."""

import numpy as np
from .training.ppo_config import PPOConfig
from .control.residual_position import residual_target
from .observations.common import core_actor, privileged_prefix
from .observations.v24 import assemble
from .gait.events import GaitEvents, qualify_foot
from types import SimpleNamespace
from .evaluation.gait_metrics import summarize_events
from .evaluation.metrics import summarize
from .rewards.common import weighted_reward


def run():
    cfg = PPOConfig()
    z = np.zeros(12)
    ctrl = residual_target(z, z, np.full(12, 0.07), np.full(12, -2.0), np.full(12, 2.0))
    core = core_actor(
        z, z, z, [0, 0, 1], [0, 0, 0], [0, 0, 0], 0, z, [1, 1], [0.1, 0, 0]
    )
    privileged = privileged_prefix(core, [1, 1], [0, 0, 0], [0, 0, 0], z, [0, 0])
    obs = assemble(core, privileged, 0, 0, 0, 0, 0, 0)
    event_config = SimpleNamespace(
        v8_liftoff_no_contact_steps=3,
        v8_liftoff_height_gain=0.017,
        v8_touchdown_contact_confirm_steps=2,
        v8_min_steps_between_qualified_touchdowns=4,
    )
    history = {
        f"v8_{k}_left": np.float32(v)
        for k, v in {
            "consecutive_no_contact_steps": 2,
            "max_height_gain_during_air": 0.02,
            "air_qualified": 0,
            "consecutive_contact_steps": 0,
            "steps_since_last_qualified_touchdown": 9,
        }.items()
    }
    qualified = qualify_foot(
        event_config,
        "left",
        np.float32(0),
        np.float32(1),
        np.float32(0.02),
        np.float32(0),
        np.float32(1),
        history,
    )
    events = summarize_events(
        [GaitEvents(liftoff_left=1), GaitEvents(touchdown_left=1, support_transfer=1)]
    )
    reward, _ = weighted_reward(
        {"alive": 1.0, "support_transfer_completion": 1.0},
        {"alive": 0.5, "support_transfer_completion": 2.0},
        0.02,
    )
    metrics = summarize(
        [0, 0.02, 0.04],
        [0, 0.001, 0.002],
        [0.03, 0.03, 0.03],
        [0.1, 0.1, 0.1],
        single_support=[0, 1, 0],
    )
    return {
        "mode": "synthetic contract only",
        "actor_shape": list(obs["state"].shape),
        "critic_shape": list(obs["privileged_state"].shape),
        "target_shape": list(ctrl["target"].shape),
        "qualified_liftoff": float(qualified["qualified_liftoff"]),
        "ppo_clip": cfg.clipping_epsilon,
        "events": events,
        "reward": reward,
        "metrics": metrics,
    }
