# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0. See LICENSE.
# Adapted from the private research working tree based on MuJoCo Playground.
# Robot descriptions and model parameters are intentionally excluded.

from __future__ import annotations
from typing import Any
import jax
import jax.numpy as jp
from mujoco import mjx


def ResidualForwardWalkingV13__v9_components(
    self,
    data: mjx.Data,
    info: dict[str, Any],
    phase_id: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    components = self._invoke_source(
        "ResidualForwardWalkingV9._v9_components",
        data,
        info,
        phase_id,
        terms,
        foot_motion,
    )
    vx_actual = self.get_local_linvel(data)[0]
    vx_cmd = info["command"][0]
    vx_error = vx_actual - vx_cmd
    sigma = jp.maximum(self._config.forward_velocity_tracking_sigma, 1e-06)
    left_contact = jp.float32(foot_motion["left_contact"])
    right_contact = jp.float32(foot_motion["right_contact"])
    left_stance_slip = left_contact * foot_motion["left_vel_xy_norm"]
    right_stance_slip = right_contact * foot_motion["right_vel_xy_norm"]
    components.update(
        {
            "forward_velocity_tracking": jp.exp(-jp.square(vx_error / sigma)),
            "stance_slip": jp.clip(
                left_stance_slip + right_stance_slip, 0.0, self._config.stance_slip_cap
            ),
            "v13_left_stance_slip": left_stance_slip,
            "v13_right_stance_slip": right_stance_slip,
            "v13_vx_error": vx_error,
        }
    )
    return components


def ResidualForwardWalkingV16__v9_components(
    self,
    data: mjx.Data,
    info: dict[str, Any],
    phase_id: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    components = self._invoke_source(
        "ResidualForwardWalkingV13._v9_components",
        data,
        info,
        phase_id,
        terms,
        foot_motion,
    )
    expected_gate_left, expected_gate_right = self._v9_expected_side_gates(info)
    expected_known = jp.float32(jp.abs(info["v9_expected_side"]) > 0.5)
    expected_left = expected_known * expected_gate_left
    expected_right = expected_known * expected_gate_right
    qualified_liftoff_left = components["qualified_liftoff_left"]
    qualified_liftoff_right = components["qualified_liftoff_right"]
    qualified_touchdown_left = components["qualified_touchdown_left"]
    qualified_touchdown_right = components["qualified_touchdown_right"]
    active_prev = jp.float32(info["v16_attempt_active"] > 0.5)
    inactive = 1.0 - active_prev
    start_left = inactive * expected_left * qualified_liftoff_left
    start_right = inactive * expected_right * qualified_liftoff_right
    start = jp.clip(start_left + start_right, 0.0, 1.0)
    start_side = start_right - start_left
    attempt_side = jp.where(start > 0.5, start_side, info["v16_attempt_side"])
    active = jp.clip(active_prev + start, 0.0, 1.0)
    attempt_left = active * jp.float32(attempt_side < -0.5)
    attempt_right = active * jp.float32(attempt_side > 0.5)
    left_contact = jp.float32(foot_motion["left_contact"])
    right_contact = jp.float32(foot_motion["right_contact"])
    physical_single_support = (
        attempt_left * (1.0 - left_contact) * right_contact
        + attempt_right * (1.0 - right_contact) * left_contact
    )
    single_support_steps = (
        active_prev * info["v16_single_support_steps"]
        + active * physical_single_support
    )
    single_support_target = jp.maximum(
        jp.float32(self._config.v16_single_support_target_steps), 1e-06
    )
    single_support_quality = jp.clip(
        single_support_steps / single_support_target, 0.0, 1.0
    )
    touchdown_left = attempt_left * qualified_touchdown_left
    touchdown_right = attempt_right * qualified_touchdown_right
    touchdown = jp.clip(touchdown_left + touchdown_right, 0.0, 1.0)
    d_forward = attempt_left * (
        foot_motion["left_pos"][0] - foot_motion["right_pos"][0]
    ) + attempt_right * (foot_motion["right_pos"][0] - foot_motion["left_pos"][0])
    forward_placement = jp.maximum(d_forward, 0.0)
    placement_target = jp.maximum(
        jp.float32(self._config.v16_forward_placement_target), 1e-06
    )
    forward_placement_quality = jp.clip(forward_placement / placement_target, 0.0, 1.0)
    latched_forward_placement = jp.where(
        touchdown > 0.5, forward_placement, info["v16_forward_placement"]
    )
    latched_forward_quality = jp.where(
        touchdown > 0.5,
        forward_placement_quality,
        info["v16_forward_placement_quality"],
    )
    touchdown_latched = jp.clip(
        active * info["v16_touchdown_latched"] + touchdown, 0.0, 1.0
    )
    side_consistent_completion = (
        attempt_left * components["left_transfer_completion"]
        + attempt_right * components["right_transfer_completion"]
    )
    valid_completion = active * touchdown_latched * side_consistent_completion
    v16_valid_step = valid_completion * latched_forward_quality * single_support_quality
    landed_side = components["next_support_transfer_landed_side"]
    side_consistent_timeout = components["support_transfer_timeout"] * (
        attempt_left * jp.float32(landed_side < -0.5)
        + attempt_right * jp.float32(landed_side > 0.5)
    )
    conflicting_liftoff = active * (
        attempt_left * qualified_liftoff_right + attempt_right * qualified_liftoff_left
    )
    cancel = (1.0 - valid_completion) * jp.clip(
        side_consistent_timeout + conflicting_liftoff, 0.0, 1.0
    )
    clear = jp.clip(valid_completion + cancel, 0.0, 1.0)
    next_active = active * (1.0 - clear)
    components.update(
        {
            "v16_valid_step": v16_valid_step,
            "v16_valid_attempt_start": start,
            "v16_expected_single_support_step": active * physical_single_support,
            "v16_forward_placement_sample": touchdown * forward_placement,
            "v16_forward_placement_count": touchdown,
            "v16_zero_forward_placement_count": touchdown
            * jp.float32(d_forward <= 0.0),
            "v16_valid_step_completion": valid_completion,
            "v16_attempt_cancel": cancel,
            "v16_single_support_quality": single_support_quality,
            "v16_forward_placement_quality": latched_forward_quality,
            "next_v16_attempt_active": next_active,
            "next_v16_attempt_side": next_active * attempt_side,
            "next_v16_single_support_steps": next_active * single_support_steps,
            "next_v16_touchdown_latched": next_active * touchdown_latched,
            "next_v16_forward_placement": next_active * latched_forward_placement,
            "next_v16_forward_placement_quality": next_active * latched_forward_quality,
        }
    )
    return components


def ResidualForwardWalkingV19__v9_components(
    self,
    data: mjx.Data,
    info: dict[str, Any],
    phase_id: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    components = self._invoke_source(
        "ResidualForwardWalkingV16._v9_components",
        data,
        info,
        phase_id,
        terms,
        foot_motion,
    )
    phase = self._v18_event_phase(info, self._v18_legacy_phase(info["step"]))
    expected_side = info["v9_expected_side"]
    expected_left = jp.float32(expected_side < -0.5)
    expected_right = jp.float32(expected_side > 0.5)
    left_contact = jp.float32(foot_motion["left_contact"])
    right_contact = jp.float32(foot_motion["right_contact"])
    components["expected_swing_contact_motion_drag"] = phase["swing"] * (
        expected_left * left_contact * foot_motion["left_vel_xy_norm"]
        + expected_right * right_contact * foot_motion["right_vel_xy_norm"]
    )
    return components


def ResidualForwardWalkingV20__v9_components(
    self,
    data: mjx.Data,
    info: dict[str, Any],
    phase_id: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    components = self._invoke_source(
        "ResidualForwardWalkingV19._v9_components",
        data,
        info,
        phase_id,
        terms,
        foot_motion,
    )
    completion = components["v16_valid_step_completion"]
    attempt_active = jp.float32(info["v16_attempt_active"] > 0.5)
    attempt_side = info["v16_attempt_side"]
    left_gate = completion * attempt_active * jp.float32(attempt_side < -0.5)
    right_gate = completion * attempt_active * jp.float32(attempt_side > 0.5)
    completion_gate = jp.clip(left_gate + right_gate, 0.0, 1.0)
    touchdown = components["v16_forward_placement_count"]
    completed_placement = jp.where(
        touchdown > 0.5,
        components["v16_forward_placement_sample"],
        info["v16_forward_placement"],
    )
    placement_min = jp.float32(self._config.v20_placement_min)
    placement_max = jp.float32(self._config.v20_placement_max)
    placement_sigma = jp.maximum(jp.float32(self._config.v20_placement_sigma), 1e-06)
    underplacement = jp.maximum(placement_min - completed_placement, 0.0)
    overplacement = jp.maximum(completed_placement - placement_max, 0.0)
    completed_placement_band = completion_gate * jp.clip(
        jp.square(underplacement / placement_sigma)
        + jp.square(overplacement / placement_sigma),
        0.0,
        1.0,
    )
    components.update(
        {
            "completed_placement_band": completed_placement_band,
            "v20_left_completed_placement_count": left_gate,
            "v20_right_completed_placement_count": right_gate,
            "v20_left_underplacement_count": left_gate
            * jp.float32(completed_placement < placement_min),
            "v20_right_underplacement_count": right_gate
            * jp.float32(completed_placement < placement_min),
            "v20_left_overplacement_count": left_gate
            * jp.float32(completed_placement > placement_max),
            "v20_right_overplacement_count": right_gate
            * jp.float32(completed_placement > placement_max),
            "v20_left_completed_placement_sum": left_gate * completed_placement,
            "v20_right_completed_placement_sum": right_gate * completed_placement,
        }
    )
    return components


def ResidualForwardWalkingV21__v9_components(
    self,
    data: mjx.Data,
    info: dict[str, Any],
    phase_id: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    components = self._invoke_source(
        "ResidualForwardWalkingV20._v9_components",
        data,
        info,
        phase_id,
        terms,
        foot_motion,
    )
    yaw = self._forward_yaw_from_quat(self.get_orientation(data))
    yaw_rate = self.get_gyro(data)[2]
    heading_error = jp.abs(yaw) + 0.1 * jp.abs(yaw_rate)
    components["yaw_drift_penalty"] = heading_error / jp.maximum(
        self._config.forward_yaw_drift_scale, 1e-06
    )
    return components


def ResidualForwardWalkingV23__v9_components(
    self,
    data: mjx.Data,
    info: dict[str, Any],
    phase_id: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    components = self._invoke_source(
        "ResidualForwardWalkingV21._v9_components",
        data,
        info,
        phase_id,
        terms,
        foot_motion,
    )
    yaw = self._forward_yaw_from_quat(self.get_orientation(data))
    angle = -yaw
    heading_error = jp.arctan2(jp.sin(angle), jp.cos(angle))
    speed_cmd = info["command"][0]
    target_vx_body = speed_cmd * jp.cos(heading_error)
    target_vy_body = speed_cmd * jp.sin(heading_error)
    local_linvel = self.get_local_linvel(data)
    measured_vx_body = local_linvel[0]
    measured_vy_body = local_linvel[1]
    sigma_x = jp.maximum(self._config.forward_velocity_tracking_sigma, 1e-06)
    sigma_y = sigma_x
    vector_tracking = jp.exp(
        -jp.square((measured_vx_body - target_vx_body) / sigma_x)
        - jp.square((measured_vy_body - target_vy_body) / sigma_y)
    )
    components["forward_velocity_tracking"] = vector_tracking
    epsilon = 1e-06
    target_speed = jp.sqrt(jp.square(target_vx_body) + jp.square(target_vy_body))
    safe_target_speed = jp.maximum(target_speed, epsilon)
    target_direction_x = target_vx_body / safe_target_speed
    target_direction_y = target_vy_body / safe_target_speed
    command_aligned_velocity = (
        measured_vx_body * target_direction_x + measured_vy_body * target_direction_y
    )
    locomotion_gate = components["locomotion_gate"]
    velocity_ratio = jp.clip(command_aligned_velocity / safe_target_speed, 0.0, 1.0)
    velocity_score = velocity_ratio * jp.exp(
        -jp.square((command_aligned_velocity - safe_target_speed) / 0.05)
    )
    components["forward_velocity_progress"] = locomotion_gate * velocity_score
    return components


def ResidualForwardWalkingV24__v9_components(
    self,
    data: mjx.Data,
    info: dict[str, Any],
    phase_id: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    components = self._invoke_source(
        "ResidualForwardWalkingV23._v9_components",
        data,
        info,
        phase_id,
        terms,
        foot_motion,
    )
    epsilon = 1e-06
    touchdown = components["v16_forward_placement_count"]
    attempt_side = info["v16_attempt_side"]
    is_left = touchdown * jp.float32(attempt_side < -0.5)
    is_right = touchdown * jp.float32(attempt_side > 0.5)
    placement_sample = components["v16_forward_placement_sample"]
    last_left = jp.where(
        is_left > 0.5, placement_sample, info["v24_last_left_placement"]
    )
    last_right = jp.where(
        is_right > 0.5, placement_sample, info["v24_last_right_placement"]
    )
    left_valid = jp.clip(info["v24_left_valid"] + is_left, 0.0, 1.0)
    right_valid = jp.clip(info["v24_right_valid"] + is_right, 0.0, 1.0)
    pair_valid = left_valid * right_valid
    previous_pair_valid = info["v24_pair_valid"]
    gap = jp.abs(last_right - last_left)
    signed_imbalance = pair_valid * (
        (last_right - last_left) / (last_right + last_left + epsilon)
    )
    abs_imbalance_current = jp.abs(signed_imbalance)
    abs_imbalance_previous = info["v24_abs_imbalance_previous"]
    bilateral_symmetry = pair_valid * (gap / 0.1)
    growth_gate = touchdown * pair_valid * previous_pair_valid
    imbalance_growth = growth_gate * jp.clip(
        jp.maximum(abs_imbalance_current - abs_imbalance_previous, 0.0) / 0.05, 0.0, 1.0
    )
    root_pos = self.get_root_pos(data)
    e_y = root_pos[1] - info["root_y_baseline"]
    persistent_lateral_path = jp.maximum(jp.abs(e_y) - 0.14, 0.0) / 0.1
    components.update(
        {
            "bilateral_symmetry": bilateral_symmetry,
            "imbalance_growth": imbalance_growth,
            "persistent_lateral_path": persistent_lateral_path,
            "next_v24_last_left_placement": last_left,
            "next_v24_last_right_placement": last_right,
            "next_v24_left_valid": left_valid,
            "next_v24_right_valid": right_valid,
            "next_v24_pair_valid": pair_valid,
            "next_v24_abs_imbalance_previous": abs_imbalance_current,
        }
    )
    return components
