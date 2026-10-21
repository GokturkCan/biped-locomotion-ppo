# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0. See LICENSE.
# Adapted from the private research working tree based on MuJoCo Playground.
# Robot descriptions and model parameters are intentionally excluded.

from __future__ import annotations
from typing import Any
import jax
import jax.numpy as jp


def FootUnloadBridgeLiteV0__bridge_phase(self, step: jax.Array) -> jax.Array:
    """Step-count based finite phase schedule (not a continuous gait clock).

    Returns 0=PREP_SHIFT, 1=UNLOAD, 2=SWING_HOLD, 3=TOUCHDOWN, 4=RECENTER.
    """
    cfg = self._config
    boundaries = jp.array(
        [
            cfg.bridge_prep_shift_steps,
            cfg.bridge_prep_shift_steps + cfg.bridge_unload_steps,
            cfg.bridge_prep_shift_steps
            + cfg.bridge_unload_steps
            + cfg.bridge_swing_hold_steps,
            cfg.bridge_prep_shift_steps
            + cfg.bridge_unload_steps
            + cfg.bridge_swing_hold_steps
            + cfg.bridge_touchdown_steps,
        ]
    )
    phase_id = jp.sum(jp.asarray(step) >= boundaries)
    return jp.clip(phase_id, 0, 4).astype(jp.int32)


def ResidualForwardWalkingV18__v18_apply_phase(
    self, info: dict[str, Any], phase: dict[str, jax.Array]
) -> dict[str, Any]:
    info["target_left_swing"] = phase["target_left_swing"]
    info["target_right_swing"] = phase["target_right_swing"]
    info["support_left"] = phase["support_left"]
    info["support_right"] = phase["support_right"]
    info["desired_left_contact"] = phase["desired_left_contact"]
    info["desired_right_contact"] = phase["desired_right_contact"]
    info["gait_phase"] = phase["gait_phase"]
    return info


def ResidualForwardWalkingV18__v18_event_phase(
    self, info: dict[str, Any], legacy: dict[str, jax.Array]
) -> dict[str, jax.Array]:
    expected_side = info["v9_expected_side"]
    known = jp.float32(jp.abs(expected_side) > 0.5)
    unknown = 1.0 - known
    expected_left = jp.float32(expected_side < -0.5)
    expected_right = jp.float32(expected_side > 0.5)
    pending = jp.float32(info["v9_transfer_pending"] > 0.5)
    age = info["v9_steps_since_expected_side_set"]
    prep_steps = jp.maximum(jp.float32(self._config.v18_prep_steps), 1.0)
    swing_steps = jp.maximum(jp.float32(self._config.v18_nominal_swing_steps), 1.0)
    transfer_steps = jp.maximum(
        jp.float32(self._config.v18_nominal_transfer_steps), 1.0
    )
    total_steps = prep_steps + swing_steps + transfer_steps
    prep = known * (1.0 - pending) * jp.float32(age < prep_steps)
    swing = known * (1.0 - pending) * jp.float32(age >= prep_steps)
    transfer = known * pending
    target_left = swing * expected_left
    target_right = swing * expected_right
    desired_left_contact = 1.0 - target_left
    desired_right_contact = 1.0 - target_right
    support_weight_sum = jp.maximum(desired_left_contact + desired_right_contact, 1.0)
    support_left = desired_left_contact / support_weight_sum
    support_right = desired_right_contact / support_weight_sum
    swing_age = jp.maximum(age - prep_steps, 0.0)
    swing_progress_steps = jp.minimum(swing_age, swing_steps)
    prep_progress_steps = jp.minimum(age, prep_steps)
    prep_progress = prep_progress_steps / prep_steps
    swing_progress = swing_progress_steps / swing_steps
    phase_saturated = known * (1.0 - pending) * jp.float32(swing_age >= swing_steps)
    prep_cycle_steps = prep_progress * prep_steps
    swing_cycle_steps = prep_steps + swing_progress * swing_steps
    transfer_cycle_steps = prep_steps + swing_steps + 0.5 * transfer_steps
    event_cycle_steps = (
        prep * prep_cycle_steps
        + swing * swing_cycle_steps
        + transfer * transfer_cycle_steps
    )
    event_half_progress = jp.clip(event_cycle_steps / total_steps, 0.0, 1.0)
    event_base = 0.5 * expected_right
    event_gait_phase = event_base + 0.5 * event_half_progress
    cfg = self._config
    legacy_phase_id = legacy["phase_id"]
    legacy_local_step = legacy["local_step"]
    prep_local = jp.minimum(
        age, jp.maximum(jp.float32(cfg.bridge_prep_shift_steps) - 1.0, 0.0)
    )
    swing_local = jp.float32(
        cfg.bridge_prep_shift_steps + cfg.bridge_unload_steps
    ) + jp.minimum(
        swing_age, jp.maximum(jp.float32(cfg.bridge_swing_hold_steps) - 1.0, 0.0)
    )
    transfer_local = jp.float32(
        cfg.bridge_prep_shift_steps
        + cfg.bridge_unload_steps
        + cfg.bridge_swing_hold_steps
        + cfg.bridge_touchdown_steps
    )
    event_phase_id = jp.where(
        prep > 0.5, jp.int32(0), jp.where(swing > 0.5, jp.int32(2), jp.int32(4))
    )
    event_local_step = jp.where(
        prep > 0.5, prep_local, jp.where(swing > 0.5, swing_local, transfer_local)
    )
    legacy_match = known * (
        expected_left * legacy["target_left_swing"]
        + expected_right * legacy["target_right_swing"]
    )
    legacy_mismatch = known * (
        expected_left * legacy["target_right_swing"]
        + expected_right * legacy["target_left_swing"]
    )

    def select(name: str, event_value: jax.Array) -> jax.Array:
        return known * event_value + unknown * legacy[name]

    gait_phase = select("gait_phase", event_gait_phase)
    actor_left_phase = jp.float32(gait_phase < 0.5)
    actor_right_phase = 1.0 - actor_left_phase
    result = {
        "known": known,
        "prep": select("prep", prep),
        "swing": select("swing", swing),
        "transfer": select("transfer", transfer),
        "target_left_swing": select("target_left_swing", target_left),
        "target_right_swing": select("target_right_swing", target_right),
        "support_left": select("support_left", support_left),
        "support_right": select("support_right", support_right),
        "desired_left_contact": select("desired_left_contact", desired_left_contact),
        "desired_right_contact": select("desired_right_contact", desired_right_contact),
        "phase_id": jp.where(known > 0.5, event_phase_id, legacy_phase_id).astype(
            jp.int32
        ),
        "local_step": select("local_step", event_local_step),
        "gait_phase": gait_phase,
        "phase_sin": jp.sin(2.0 * jp.pi * gait_phase),
        "phase_cos": jp.cos(2.0 * jp.pi * gait_phase),
        "actor_left_phase": actor_left_phase,
        "actor_right_phase": actor_right_phase,
        "phase_saturated": select("phase_saturated", phase_saturated),
        "legacy_side_match": legacy_match,
        "legacy_side_mismatch": legacy_mismatch,
    }
    return result


def ResidualForwardWalkingV18__v18_legacy_phase(
    self, step: jax.Array
) -> dict[str, jax.Array]:
    target_left, target_right, phase_id, local_step, gait_phase = self._stepping_phase(
        step
    )
    return {
        "known": jp.zeros(()),
        "prep": jp.float32(phase_id == 0),
        "swing": jp.float32(phase_id == 2),
        "transfer": jp.float32((phase_id == 3) | (phase_id == 4)),
        "target_left_swing": target_left,
        "target_right_swing": target_right,
        "support_left": target_right,
        "support_right": target_left,
        "desired_left_contact": 1.0 - target_left,
        "desired_right_contact": 1.0 - target_right,
        "phase_id": phase_id,
        "local_step": local_step,
        "gait_phase": gait_phase,
        "phase_sin": jp.sin(2.0 * jp.pi * gait_phase),
        "phase_cos": jp.cos(2.0 * jp.pi * gait_phase),
        "phase_saturated": jp.zeros(()),
        "legacy_side_match": jp.zeros(()),
        "legacy_side_mismatch": jp.zeros(()),
    }


def ResidualSteppingInPlaceV1__stepping_phase(
    self, step: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Deterministic periodic phase from the global step count.

    step in [0, half_cycle) is left swing, [half_cycle, cycle) is right
    swing; the sub-phase within each half reuses
    FootUnloadBridgeLiteV0._bridge_phase unchanged via the local step count
    within that half-cycle.

    Returns (target_left, target_right, phase_id, local_step, gait_phase).
    """
    cycle_step = jp.mod(jp.asarray(step), self._cycle_steps)
    is_left_half = cycle_step < self._half_cycle_steps
    local_step = jp.where(is_left_half, cycle_step, cycle_step - self._half_cycle_steps)
    target_left = jp.float32(is_left_half)
    target_right = 1.0 - target_left
    phase_id = self._bridge_phase(local_step)
    gait_phase = jp.float32(cycle_step) / jp.float32(self._cycle_steps)
    return (target_left, target_right, phase_id, local_step, gait_phase)
