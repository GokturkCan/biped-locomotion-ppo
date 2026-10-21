# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0. See LICENSE.
# Adapted from the private research working tree based on MuJoCo Playground.
# Robot descriptions and model parameters are intentionally excluded.

from __future__ import annotations
from typing import Any
import jax
import jax.numpy as jp


def ResidualForwardWalkingV2__q_ref_phase(
    self,
    phase_id: jax.Array,
    target_left: jax.Array,
    target_right: jax.Array,
    info: dict[str, Any],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    del target_left, target_right
    cfg = self._config
    step = jp.float32(info["step"])
    prep = jp.float32(cfg.bridge_prep_shift_steps)
    unload = jp.float32(cfg.bridge_unload_steps)
    hold = jp.float32(cfg.bridge_swing_hold_steps)
    touchdown = jp.float32(cfg.bridge_touchdown_steps)
    recenter = jp.float32(cfg.bridge_recenter_steps)
    shift_end = prep + unload
    recenter_start = prep + unload + hold + touchdown
    recenter_end = recenter_start + recenter
    ramp_up = jp.clip(step / jp.maximum(shift_end, 1.0), 0.0, 1.0)
    ramp_down = jp.clip((recenter_end - step) / jp.maximum(recenter, 1.0), 0.0, 1.0)
    unload_weight = jp.where(step >= recenter_start, ramp_down, ramp_up)
    unload_weight = jp.clip(unload_weight, 0.0, 1.0)
    q_nominal = info.get("bridge_q_ref_base", self._default_pose)
    return (q_nominal, unload_weight, jp.zeros(()))


def ResidualForwardWalkingV2__v3_control_law(
    self,
    action: jax.Array,
    q_ref: jax.Array,
    target_left: jax.Array,
    target_right: jax.Array,
    phase_id: jax.Array,
    local_step: jax.Array,
) -> dict[str, jax.Array]:
    del target_left, target_right, phase_id, local_step
    action_policy = action
    q_nominal = q_ref
    requested_residual = self._action_scale_jp * action_policy
    raw_target = q_nominal + requested_residual
    motor_targets = jp.clip(raw_target, self._joint_lowers, self._joint_uppers)
    effective_residual = motor_targets - q_nominal
    target_clipped = jp.float32(
        jp.any((raw_target < self._joint_lowers) | (raw_target > self._joint_uppers))
    )
    return {
        "clipped_action": action_policy,
        "motor_targets": motor_targets,
        "effective_scale": self._action_scale_jp,
        "requested_residual": requested_residual,
        "effective_residual": effective_residual,
        "target_clipped": target_clipped,
        "swing_authority_gate": jp.ones(()),
        "relaxable_mask": jp.ones_like(action_policy),
        "requested_residual_offset_norm": jp.linalg.norm(requested_residual),
        "effective_residual_offset_norm": jp.linalg.norm(effective_residual),
        "swing_residual_action_norm": jp.linalg.norm(action_policy),
    }
