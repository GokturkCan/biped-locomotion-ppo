# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0. See LICENSE.
# Adapted from the private research working tree based on MuJoCo Playground.
# Robot descriptions and model parameters are intentionally excluded.

from __future__ import annotations
from typing import Any, Dict
import jax
import jax.numpy as jp
from mujoco import mjx
from biped_locomotion.commands import process as ucp
from biped_locomotion.rewards import unified_contract as ucfg
from biped_locomotion.gait.unified_events import (
    U4_STRICT_YAW_DC_METRIC_KEYS,
    U4_UNIFIED_METRIC_KEYS,
)


def ResidualForwardWalkingV13__empty_metrics(self) -> dict[str, jax.Array]:
    metrics = self._invoke_source("ResidualForwardWalkingV9._empty_metrics")
    metrics.update(
        {
            name: jp.zeros(())
            for name in (
                "tracking/forward_velocity_tracking",
                "tracking/forward_velocity_tracking_scaled",
                "tracking/vx_error",
                "contact/stance_slip",
                "contact/left_stance_slip",
                "contact/right_stance_slip",
                "penalty/stance_slip",
            )
        }
    )
    return metrics


def ResidualForwardWalkingV13__update_metrics(
    self,
    metrics: dict[str, jax.Array],
    data: mjx.Data,
    clipped_action: jax.Array,
    raw_action: jax.Array,
    motor_targets: jax.Array,
    scaled_rewards: dict[str, jax.Array],
    info: dict[str, Any],
    phase_id: jax.Array,
    gait_phase: jax.Array,
    q_ref: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
    done: jax.Array,
    swing_contact_count: jax.Array,
    swing_authority_gate: jax.Array,
    swing_residual_action_norm: jax.Array,
    requested_residual_offset_norm: jax.Array,
    effective_residual_offset_norm: jax.Array,
    target_clipped: jax.Array,
    target_clipped_steps: jax.Array,
    min_swing_contact_count_left: jax.Array,
    min_swing_contact_count_right: jax.Array,
    max_clearance_left: jax.Array,
    max_clearance_right: jax.Array,
    max_abs_roll_left: jax.Array,
    max_abs_roll_right: jax.Array,
) -> dict[str, jax.Array]:
    metrics = self._invoke_source(
        "ResidualForwardWalkingV9._update_metrics",
        metrics,
        data,
        clipped_action,
        raw_action,
        motor_targets,
        scaled_rewards,
        info,
        phase_id,
        gait_phase,
        q_ref,
        terms,
        foot_motion,
        done,
        swing_contact_count,
        swing_authority_gate,
        swing_residual_action_norm,
        requested_residual_offset_norm,
        effective_residual_offset_norm,
        target_clipped,
        target_clipped_steps,
        min_swing_contact_count_left,
        min_swing_contact_count_right,
        max_clearance_left,
        max_clearance_right,
        max_abs_roll_left,
        max_abs_roll_right,
    )
    components = self._v9_components(data, info, phase_id, terms, foot_motion)
    metrics["tracking/forward_velocity_tracking"] = components[
        "forward_velocity_tracking"
    ]
    metrics["tracking/forward_velocity_tracking_scaled"] = scaled_rewards[
        "forward_velocity_tracking"
    ]
    metrics["tracking/vx_error"] = components["v13_vx_error"]
    metrics["contact/stance_slip"] = components["stance_slip"]
    metrics["contact/left_stance_slip"] = components["v13_left_stance_slip"]
    metrics["contact/right_stance_slip"] = components["v13_right_stance_slip"]
    metrics["penalty/stance_slip"] = scaled_rewards["stance_slip"]
    return metrics


def ResidualForwardWalkingV16__empty_metrics(self) -> dict[str, jax.Array]:
    metrics = self._invoke_source("ResidualForwardWalkingV13._empty_metrics")
    metrics.update(
        {
            name: jp.zeros(())
            for name in (
                "transition/v16_valid_attempt_start_count",
                "gait/v16_expected_single_support_step_sum",
                "transition/v16_forward_placement_sum",
                "transition/v16_forward_placement_count",
                "transition/v16_zero_forward_placement_count",
                "transition/v16_valid_step_completion_count",
                "transition/v16_attempt_cancel_count",
            )
        }
    )
    return metrics


def ResidualForwardWalkingV16__update_metrics(
    self,
    metrics: dict[str, jax.Array],
    data: mjx.Data,
    clipped_action: jax.Array,
    raw_action: jax.Array,
    motor_targets: jax.Array,
    scaled_rewards: dict[str, jax.Array],
    info: dict[str, Any],
    phase_id: jax.Array,
    gait_phase: jax.Array,
    q_ref: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
    done: jax.Array,
    swing_contact_count: jax.Array,
    swing_authority_gate: jax.Array,
    swing_residual_action_norm: jax.Array,
    requested_residual_offset_norm: jax.Array,
    effective_residual_offset_norm: jax.Array,
    target_clipped: jax.Array,
    target_clipped_steps: jax.Array,
    min_swing_contact_count_left: jax.Array,
    min_swing_contact_count_right: jax.Array,
    max_clearance_left: jax.Array,
    max_clearance_right: jax.Array,
    max_abs_roll_left: jax.Array,
    max_abs_roll_right: jax.Array,
) -> dict[str, jax.Array]:
    metrics = self._invoke_source(
        "ResidualForwardWalkingV13._update_metrics",
        metrics,
        data,
        clipped_action,
        raw_action,
        motor_targets,
        scaled_rewards,
        info,
        phase_id,
        gait_phase,
        q_ref,
        terms,
        foot_motion,
        done,
        swing_contact_count,
        swing_authority_gate,
        swing_residual_action_norm,
        requested_residual_offset_norm,
        effective_residual_offset_norm,
        target_clipped,
        target_clipped_steps,
        min_swing_contact_count_left,
        min_swing_contact_count_right,
        max_clearance_left,
        max_clearance_right,
        max_abs_roll_left,
        max_abs_roll_right,
    )
    components = self._v9_components(data, info, phase_id, terms, foot_motion)
    metrics["transition/v16_valid_attempt_start_count"] = components[
        "v16_valid_attempt_start"
    ]
    metrics["gait/v16_expected_single_support_step_sum"] = components[
        "v16_expected_single_support_step"
    ]
    metrics["transition/v16_forward_placement_sum"] = components[
        "v16_forward_placement_sample"
    ]
    metrics["transition/v16_forward_placement_count"] = components[
        "v16_forward_placement_count"
    ]
    metrics["transition/v16_zero_forward_placement_count"] = components[
        "v16_zero_forward_placement_count"
    ]
    metrics["transition/v16_valid_step_completion_count"] = components[
        "v16_valid_step_completion"
    ]
    metrics["transition/v16_attempt_cancel_count"] = components["v16_attempt_cancel"]
    metrics["reward/v16_valid_step"] = scaled_rewards["v16_valid_step"]
    return metrics


def ResidualForwardWalkingV18__empty_metrics(self) -> dict[str, jax.Array]:
    metrics = self._invoke_source("ResidualForwardWalkingV16._empty_metrics")
    metrics.update(
        {
            name: jp.zeros(())
            for name in (
                "transition/v18_unknown_phase_step_count",
                "gait/v18_prep_step_count",
                "gait/v18_swing_target_step_count",
                "gait/v18_transfer_step_count",
                "gait/v18_expected_left_step_count",
                "gait/v18_expected_right_step_count",
                "gait/v18_desired_single_support_step_count",
                "transition/v18_phase_saturated_step_count",
                "transition/v18_legacy_side_match_step_count",
                "transition/v18_legacy_side_mismatch_step_count",
                "transition/v18_actor_expected_side_match_step_count",
                "transition/v18_actor_expected_side_mismatch_step_count",
                "transition/v18_opposite_side_phase_entry_count",
                "transition/v18_side_change_after_completion_count",
                "transition/v18_side_change_without_completion_count",
                "transition/v18_saturated_side_match_step_count",
                "transition/v18_saturated_side_mismatch_step_count",
                "transition/v18_control_event_side_match_step_count",
                "transition/v18_control_event_side_mismatch_step_count",
                "transition/v18_transfer_completion_reset_count",
                "gait/v18_double_contact_schedule_step_count",
                "gait/v18_single_contact_schedule_step_count",
            )
        }
    )
    return metrics


def ResidualForwardWalkingV18__update_metrics(
    self,
    metrics: dict[str, jax.Array],
    data: mjx.Data,
    clipped_action: jax.Array,
    raw_action: jax.Array,
    motor_targets: jax.Array,
    scaled_rewards: dict[str, jax.Array],
    info: dict[str, Any],
    phase_id: jax.Array,
    gait_phase: jax.Array,
    q_ref: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
    done: jax.Array,
    swing_contact_count: jax.Array,
    swing_authority_gate: jax.Array,
    swing_residual_action_norm: jax.Array,
    requested_residual_offset_norm: jax.Array,
    effective_residual_offset_norm: jax.Array,
    target_clipped: jax.Array,
    target_clipped_steps: jax.Array,
    min_swing_contact_count_left: jax.Array,
    min_swing_contact_count_right: jax.Array,
    max_clearance_left: jax.Array,
    max_clearance_right: jax.Array,
    max_abs_roll_left: jax.Array,
    max_abs_roll_right: jax.Array,
) -> dict[str, jax.Array]:
    metrics = self._invoke_source(
        "ResidualForwardWalkingV16._update_metrics",
        metrics,
        data,
        clipped_action,
        raw_action,
        motor_targets,
        scaled_rewards,
        info,
        phase_id,
        gait_phase,
        q_ref,
        terms,
        foot_motion,
        done,
        swing_contact_count,
        swing_authority_gate,
        swing_residual_action_norm,
        requested_residual_offset_norm,
        effective_residual_offset_norm,
        target_clipped,
        target_clipped_steps,
        min_swing_contact_count_left,
        min_swing_contact_count_right,
        max_clearance_left,
        max_clearance_right,
        max_abs_roll_left,
        max_abs_roll_right,
    )
    phase = self._v18_event_phase(info, self._v18_legacy_phase(info["step"]))
    components = self._v9_components(data, info, phase_id, terms, foot_motion)
    known = phase["known"]
    expected_left = known * jp.float32(info["v9_expected_side"] < -0.5)
    expected_right = known * jp.float32(info["v9_expected_side"] > 0.5)
    actor_left = phase["actor_left_phase"]
    actor_right = phase["actor_right_phase"]
    actor_match = known * (expected_left * actor_left + expected_right * actor_right)
    actor_mismatch = known * (expected_left * actor_right + expected_right * actor_left)
    event_left = jp.float32(phase["target_left_swing"] > 0.5)
    event_right = jp.float32(phase["target_right_swing"] > 0.5)
    control_left = jp.float32(info["target_left_swing"] > 0.5)
    control_right = jp.float32(info["target_right_swing"] > 0.5)
    control_event_match = known * (
        event_left * control_left + event_right * control_right
    )
    control_event_mismatch = known * (
        event_left * control_right + event_right * control_left
    )
    next_expected_side = components["next_v9_expected_side"]
    next_known = jp.float32(jp.abs(next_expected_side) > 0.5)
    expected_side_changed = (
        known * next_known * jp.float32(next_expected_side != info["v9_expected_side"])
    )
    transfer_completion = components["support_transfer_completion"]
    double_contact_schedule = phase["prep"] + phase["transfer"]
    metrics["transition/v18_unknown_phase_step_count"] = 1.0 - known
    metrics["gait/v18_prep_step_count"] = phase["prep"]
    metrics["gait/v18_swing_target_step_count"] = phase["swing"]
    metrics["gait/v18_transfer_step_count"] = phase["transfer"]
    metrics["gait/v18_expected_left_step_count"] = expected_left
    metrics["gait/v18_expected_right_step_count"] = expected_right
    metrics["gait/v18_desired_single_support_step_count"] = phase["swing"]
    metrics["transition/v18_phase_saturated_step_count"] = phase["phase_saturated"]
    metrics["transition/v18_legacy_side_match_step_count"] = phase["legacy_side_match"]
    metrics["transition/v18_legacy_side_mismatch_step_count"] = phase[
        "legacy_side_mismatch"
    ]
    metrics["transition/v18_actor_expected_side_match_step_count"] = actor_match
    metrics["transition/v18_actor_expected_side_mismatch_step_count"] = actor_mismatch
    metrics["transition/v18_opposite_side_phase_entry_count"] = actor_mismatch
    metrics["transition/v18_side_change_after_completion_count"] = (
        expected_side_changed * transfer_completion
    )
    metrics["transition/v18_side_change_without_completion_count"] = (
        expected_side_changed * (1.0 - transfer_completion)
    )
    metrics["transition/v18_saturated_side_match_step_count"] = (
        phase["phase_saturated"] * actor_match
    )
    metrics["transition/v18_saturated_side_mismatch_step_count"] = (
        phase["phase_saturated"] * actor_mismatch
    )
    metrics["transition/v18_control_event_side_match_step_count"] = control_event_match
    metrics["transition/v18_control_event_side_mismatch_step_count"] = (
        control_event_mismatch
    )
    metrics["transition/v18_transfer_completion_reset_count"] = transfer_completion
    metrics["gait/v18_double_contact_schedule_step_count"] = double_contact_schedule
    metrics["gait/v18_single_contact_schedule_step_count"] = phase["swing"]
    return metrics


def ResidualForwardWalkingV2__empty_metrics(self) -> dict[str, jax.Array]:
    metrics = self._invoke_source("ResidualSteppingInPlaceV3._empty_metrics")
    for name in (
        "reward/q_ref_tracking",
        "tracking/q_ref_error",
        "tracking/swing_leg_q_ref_error",
        "tracking/stance_leg_q_ref_error",
        "control/swing_authority_gate",
        "control/swing_residual_action_norm",
        "control/target_clipped_steps",
    ):
        metrics.pop(name, None)
    metrics.update(
        {
            name: jp.zeros(())
            for name in (
                "tracking/mean_command_vx",
                "tracking/mean_forward_velocity",
                "tracking/mean_vx_abs_error",
                "tracking/net_forward_displacement",
                "gait/left_swing_forward_progress",
                "gait/right_swing_forward_progress",
                "contact/left_swing_drag",
                "contact/right_swing_drag",
                "contact/left_stance_slip",
                "contact/right_stance_slip",
                "gait/left_no_contact_step_count",
                "gait/right_no_contact_step_count",
                "gait/left_swing_no_contact_step_count",
                "gait/right_swing_no_contact_step_count",
                "gait/left_no_contact_streak_end_event",
                "gait/right_no_contact_streak_end_event",
                "touchdown/left_event_count",
                "touchdown/right_event_count",
                "touchdown/left_vertical_velocity_sum",
                "touchdown/right_vertical_velocity_sum",
                "control/target_clipped_step_count",
            )
        }
    )
    return metrics


def ResidualForwardWalkingV2__update_metrics(
    self,
    metrics: dict[str, jax.Array],
    data: mjx.Data,
    clipped_action: jax.Array,
    raw_action: jax.Array,
    motor_targets: jax.Array,
    scaled_rewards: dict[str, jax.Array],
    info: dict[str, Any],
    phase_id: jax.Array,
    gait_phase: jax.Array,
    q_ref: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
    done: jax.Array,
    swing_contact_count: jax.Array,
    swing_authority_gate: jax.Array,
    swing_residual_action_norm: jax.Array,
    requested_residual_offset_norm: jax.Array,
    effective_residual_offset_norm: jax.Array,
    target_clipped: jax.Array,
    target_clipped_steps: jax.Array,
    min_swing_contact_count_left: jax.Array,
    min_swing_contact_count_right: jax.Array,
    max_clearance_left: jax.Array,
    max_clearance_right: jax.Array,
    max_abs_roll_left: jax.Array,
    max_abs_roll_right: jax.Array,
) -> dict[str, jax.Array]:
    metrics = self._invoke_source(
        "ResidualSteppingInPlaceV3._update_metrics",
        metrics,
        data,
        clipped_action,
        raw_action,
        motor_targets,
        scaled_rewards,
        info,
        phase_id,
        gait_phase,
        q_ref,
        terms,
        foot_motion,
        done,
        swing_contact_count,
        swing_authority_gate,
        swing_residual_action_norm,
        requested_residual_offset_norm,
        effective_residual_offset_norm,
        target_clipped,
        target_clipped_steps,
        min_swing_contact_count_left,
        min_swing_contact_count_right,
        max_clearance_left,
        max_clearance_right,
        max_abs_roll_left,
        max_abs_roll_right,
    )
    for name in (
        "reward/q_ref_tracking",
        "tracking/q_ref_error",
        "tracking/swing_leg_q_ref_error",
        "tracking/stance_leg_q_ref_error",
        "control/swing_authority_gate",
        "control/swing_residual_action_norm",
        "control/target_clipped_steps",
    ):
        metrics.pop(name, None)
    forward = self._forward_terms(data, info, phase_id, terms, foot_motion)
    episode_length = jp.maximum(jp.float32(self._config.episode_length), 1.0)
    left_no_contact = 1.0 - jp.float32(foot_motion["left_contact"])
    right_no_contact = 1.0 - jp.float32(foot_motion["right_contact"])
    left_streak_end = (
        forward["left_touchdown"] * info["current_consecutive_left_no_contact"]
    )
    right_streak_end = (
        forward["right_touchdown"] * info["current_consecutive_right_no_contact"]
    )
    metrics["tracking/mean_command_vx"] = forward["vx_cmd"] / episode_length
    metrics["tracking/mean_forward_velocity"] = forward["vx_actual"] / episode_length
    metrics["tracking/mean_vx_abs_error"] = forward["vx_abs_error"] / episode_length
    metrics["tracking/net_forward_displacement"] = forward["root_dx"]
    metrics["gait/left_swing_forward_progress"] = forward["left_swing_forward_progress"]
    metrics["gait/right_swing_forward_progress"] = forward[
        "right_swing_forward_progress"
    ]
    metrics["contact/left_swing_drag"] = forward["left_swing_drag"]
    metrics["contact/right_swing_drag"] = forward["right_swing_drag"]
    metrics["contact/left_stance_slip"] = forward["left_stance_slip"]
    metrics["contact/right_stance_slip"] = forward["right_stance_slip"]
    metrics["gait/left_no_contact_step_count"] = left_no_contact
    metrics["gait/right_no_contact_step_count"] = right_no_contact
    metrics["gait/left_swing_no_contact_step_count"] = (
        info["target_left_swing"] * left_no_contact
    )
    metrics["gait/right_swing_no_contact_step_count"] = (
        info["target_right_swing"] * right_no_contact
    )
    metrics["gait/left_no_contact_streak_end_event"] = left_streak_end
    metrics["gait/right_no_contact_streak_end_event"] = right_streak_end
    metrics["touchdown/left_event_count"] = forward["left_touchdown"]
    metrics["touchdown/right_event_count"] = forward["right_touchdown"]
    metrics["touchdown/left_vertical_velocity_sum"] = forward["left_touchdown_vz"]
    metrics["touchdown/right_vertical_velocity_sum"] = forward["right_touchdown_vz"]
    metrics["control/target_clipped_step_count"] = target_clipped
    return metrics


def ResidualForwardWalkingV20__empty_metrics(self) -> dict[str, jax.Array]:
    metrics = self._invoke_source("ResidualForwardWalkingV18._empty_metrics")
    metrics.update(
        {
            name: jp.zeros(())
            for name in (
                "transition/v20_left_completed_placement_count",
                "transition/v20_right_completed_placement_count",
                "transition/v20_left_underplacement_count",
                "transition/v20_right_underplacement_count",
                "transition/v20_left_overplacement_count",
                "transition/v20_right_overplacement_count",
                "tracking/v20_left_completed_placement_sum",
                "tracking/v20_right_completed_placement_sum",
            )
        }
    )
    return metrics


def ResidualForwardWalkingV20__update_metrics(
    self,
    metrics: dict[str, jax.Array],
    data: mjx.Data,
    clipped_action: jax.Array,
    raw_action: jax.Array,
    motor_targets: jax.Array,
    scaled_rewards: dict[str, jax.Array],
    info: dict[str, Any],
    phase_id: jax.Array,
    gait_phase: jax.Array,
    q_ref: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
    done: jax.Array,
    swing_contact_count: jax.Array,
    swing_authority_gate: jax.Array,
    swing_residual_action_norm: jax.Array,
    requested_residual_offset_norm: jax.Array,
    effective_residual_offset_norm: jax.Array,
    target_clipped: jax.Array,
    target_clipped_steps: jax.Array,
    min_swing_contact_count_left: jax.Array,
    min_swing_contact_count_right: jax.Array,
    max_clearance_left: jax.Array,
    max_clearance_right: jax.Array,
    max_abs_roll_left: jax.Array,
    max_abs_roll_right: jax.Array,
) -> dict[str, jax.Array]:
    metrics = self._invoke_source(
        "ResidualForwardWalkingV18._update_metrics",
        metrics,
        data,
        clipped_action,
        raw_action,
        motor_targets,
        scaled_rewards,
        info,
        phase_id,
        gait_phase,
        q_ref,
        terms,
        foot_motion,
        done,
        swing_contact_count,
        swing_authority_gate,
        swing_residual_action_norm,
        requested_residual_offset_norm,
        effective_residual_offset_norm,
        target_clipped,
        target_clipped_steps,
        min_swing_contact_count_left,
        min_swing_contact_count_right,
        max_clearance_left,
        max_clearance_right,
        max_abs_roll_left,
        max_abs_roll_right,
    )
    components = self._v9_components(data, info, phase_id, terms, foot_motion)
    for side in ("left", "right"):
        metrics[f"transition/v20_{side}_completed_placement_count"] = components[
            f"v20_{side}_completed_placement_count"
        ]
        metrics[f"transition/v20_{side}_underplacement_count"] = components[
            f"v20_{side}_underplacement_count"
        ]
        metrics[f"transition/v20_{side}_overplacement_count"] = components[
            f"v20_{side}_overplacement_count"
        ]
        metrics[f"tracking/v20_{side}_completed_placement_sum"] = components[
            f"v20_{side}_completed_placement_sum"
        ]
    return metrics


def ResidualForwardWalkingV6__empty_metrics(self) -> dict[str, jax.Array]:
    metrics = self._invoke_source("ResidualForwardWalkingV2._empty_metrics")
    metrics.update(
        {
            name: jp.zeros(())
            for name in (
                "liftoff/continuous_swing_load",
                "liftoff/deep_unload",
                "liftoff/pre_liftoff_clearance",
                "liftoff/effective_liftoff",
                "penalty/ground_load_loss",
                "force/mean_left_load_fraction",
                "force/mean_right_load_fraction",
                "force/mean_swing_load_fraction",
                "force/mean_support_load_fraction",
                "force/swing_load_below_0p35_step_count",
                "force/swing_load_below_0p20_step_count",
                "force/swing_load_below_0p10_step_count",
                "force/swing_load_below_0p05_step_count",
                "gait/mean_left_height_gain",
                "gait/mean_right_height_gain",
                "gait/left_height_above_0p003_step_count",
                "gait/right_height_above_0p003_step_count",
                "gait/left_no_contact_step_count",
                "gait/right_no_contact_step_count",
                "gait/left_actual_single_support",
                "gait/right_actual_single_support",
                "touchdown/left_count",
                "touchdown/right_count",
                "touchdown/left_air_time_duration_sum",
                "touchdown/right_air_time_duration_sum",
                "touchdown/recent_true_touchdown_step_count",
                "gait/double_support",
                "tracking/mean_forward_velocity",
                "tracking/net_forward_displacement",
                "termination/fall",
                "force/left_fz",
                "force/right_fz",
                "force/total_load_fraction",
                "touchdown/forward_touchdown",
                "touchdown/forward_placement_score",
            )
        }
    )
    return metrics


def ResidualForwardWalkingV6__update_metrics(
    self,
    metrics: dict[str, jax.Array],
    data: mjx.Data,
    clipped_action: jax.Array,
    raw_action: jax.Array,
    motor_targets: jax.Array,
    scaled_rewards: dict[str, jax.Array],
    info: dict[str, Any],
    phase_id: jax.Array,
    gait_phase: jax.Array,
    q_ref: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
    done: jax.Array,
    swing_contact_count: jax.Array,
    swing_authority_gate: jax.Array,
    swing_residual_action_norm: jax.Array,
    requested_residual_offset_norm: jax.Array,
    effective_residual_offset_norm: jax.Array,
    target_clipped: jax.Array,
    target_clipped_steps: jax.Array,
    min_swing_contact_count_left: jax.Array,
    min_swing_contact_count_right: jax.Array,
    max_clearance_left: jax.Array,
    max_clearance_right: jax.Array,
    max_abs_roll_left: jax.Array,
    max_abs_roll_right: jax.Array,
) -> dict[str, jax.Array]:
    metrics = self._invoke_source(
        "ResidualForwardWalkingV2._update_metrics",
        metrics,
        data,
        clipped_action,
        raw_action,
        motor_targets,
        scaled_rewards,
        info,
        phase_id,
        gait_phase,
        q_ref,
        terms,
        foot_motion,
        done,
        swing_contact_count,
        swing_authority_gate,
        swing_residual_action_norm,
        requested_residual_offset_norm,
        effective_residual_offset_norm,
        target_clipped,
        target_clipped_steps,
        min_swing_contact_count_left,
        min_swing_contact_count_right,
        max_clearance_left,
        max_clearance_right,
        max_abs_roll_left,
        max_abs_roll_right,
    )
    components = self._v6_components(data, info, phase_id, terms, foot_motion)
    episode_length = jp.maximum(jp.float32(self._config.episode_length), 1.0)
    left_no_contact = 1.0 - jp.float32(foot_motion["left_contact"])
    right_no_contact = 1.0 - jp.float32(foot_motion["right_contact"])
    metrics["liftoff/continuous_swing_load"] = components["continuous_swing_load"]
    metrics["liftoff/deep_unload"] = components["deep_unload"]
    metrics["liftoff/pre_liftoff_clearance"] = components["pre_liftoff_clearance"]
    metrics["liftoff/effective_liftoff"] = components["effective_liftoff"]
    metrics["penalty/ground_load_loss"] = components["ground_load_loss_penalty"]
    metrics["force/mean_left_load_fraction"] = (
        components["lambda_left"] / episode_length
    )
    metrics["force/mean_right_load_fraction"] = (
        components["lambda_right"] / episode_length
    )
    metrics["force/mean_swing_load_fraction"] = (
        components["lambda_swing"] / episode_length
    )
    metrics["force/mean_support_load_fraction"] = (
        components["lambda_support"] / episode_length
    )
    swing_load_count_gate = (
        jp.float32((phase_id == 2) | (phase_id == 3))
        * components["valid_ground_load_gate"]
        * terms["support_contact"]
    )
    metrics["force/swing_load_below_0p35_step_count"] = jp.float32(
        (swing_load_count_gate > 0.999) & (components["lambda_swing"] < 0.35)
    )
    metrics["force/swing_load_below_0p20_step_count"] = jp.float32(
        (swing_load_count_gate > 0.999) & (components["lambda_swing"] < 0.2)
    )
    metrics["force/swing_load_below_0p10_step_count"] = jp.float32(
        (swing_load_count_gate > 0.999) & (components["lambda_swing"] < 0.1)
    )
    metrics["force/swing_load_below_0p05_step_count"] = jp.float32(
        (swing_load_count_gate > 0.999) & (components["lambda_swing"] < 0.05)
    )
    metrics["gait/mean_left_height_gain"] = (
        components["left_height_gain"] / episode_length
    )
    metrics["gait/mean_right_height_gain"] = (
        components["right_height_gain"] / episode_length
    )
    metrics["gait/left_height_above_0p003_step_count"] = jp.float32(
        components["left_height_gain"] > 0.003
    )
    metrics["gait/right_height_above_0p003_step_count"] = jp.float32(
        components["right_height_gain"] > 0.003
    )
    metrics["gait/left_no_contact_step_count"] = left_no_contact
    metrics["gait/right_no_contact_step_count"] = right_no_contact
    metrics["gait/left_actual_single_support"] = components[
        "left_actual_single_support"
    ]
    metrics["gait/right_actual_single_support"] = components[
        "right_actual_single_support"
    ]
    metrics["touchdown/left_count"] = components["left_touchdown"]
    metrics["touchdown/right_count"] = components["right_touchdown"]
    metrics["touchdown/left_air_time_duration_sum"] = (
        components["left_touchdown"]
        * info["current_consecutive_left_no_contact"]
        * self.dt
    )
    metrics["touchdown/right_air_time_duration_sum"] = (
        components["right_touchdown"]
        * info["current_consecutive_right_no_contact"]
        * self.dt
    )
    metrics["touchdown/recent_true_touchdown_step_count"] = components[
        "recent_true_touchdown"
    ]
    metrics["gait/double_support"] = jp.float32(
        foot_motion["left_contact"] & foot_motion["right_contact"]
    )
    metrics["tracking/mean_forward_velocity"] = components["vx_actual"] / episode_length
    metrics["tracking/net_forward_displacement"] = components["root_dx"]
    metrics["termination/fall"] = components["fall"]
    metrics["force/left_fz"] = components["fz_left"]
    metrics["force/right_fz"] = components["fz_right"]
    metrics["force/total_load_fraction"] = components["total_load_fraction"]
    metrics["touchdown/forward_touchdown"] = components["forward_touchdown"]
    metrics["touchdown/forward_placement_score"] = components["forward_touchdown_score"]
    return metrics


def ResidualForwardWalkingV7__empty_metrics(self) -> dict[str, jax.Array]:
    metrics = self._invoke_source("ResidualForwardWalkingV6._empty_metrics")
    metrics.update(
        {
            name: jp.zeros(())
            for name in (
                "stability/stable_single_support",
                "stability/post_touchdown_stability",
                "stability/touchdown_load_acceptance",
                "stability/left_touchdown_load_acceptance",
                "stability/right_touchdown_load_acceptance",
                "stability/locomotion_instability",
                "stability/upright_gate",
                "stability/root_height_gate",
                "stability/angular_stability_gate",
                "gait/alternating_touchdown_count",
                "gait/same_side_touchdown_repeat_count",
                "gait/last_touchdown_side",
                "termination/fall_event",
                "termination/term_height",
                "termination/term_roll",
                "termination/term_pitch",
                "termination/terminal_root_height",
                "termination/terminal_abs_roll",
                "termination/terminal_abs_pitch",
                "termination/terminal_roll_pitch_angular_velocity",
                "termination/terminal_fall_cost",
            )
        }
    )
    return metrics


def ResidualForwardWalkingV7__update_metrics(
    self,
    metrics: dict[str, jax.Array],
    data: mjx.Data,
    clipped_action: jax.Array,
    raw_action: jax.Array,
    motor_targets: jax.Array,
    scaled_rewards: dict[str, jax.Array],
    info: dict[str, Any],
    phase_id: jax.Array,
    gait_phase: jax.Array,
    q_ref: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
    done: jax.Array,
    swing_contact_count: jax.Array,
    swing_authority_gate: jax.Array,
    swing_residual_action_norm: jax.Array,
    requested_residual_offset_norm: jax.Array,
    effective_residual_offset_norm: jax.Array,
    target_clipped: jax.Array,
    target_clipped_steps: jax.Array,
    min_swing_contact_count_left: jax.Array,
    min_swing_contact_count_right: jax.Array,
    max_clearance_left: jax.Array,
    max_clearance_right: jax.Array,
    max_abs_roll_left: jax.Array,
    max_abs_roll_right: jax.Array,
) -> dict[str, jax.Array]:
    metrics = self._invoke_source(
        "ResidualForwardWalkingV6._update_metrics",
        metrics,
        data,
        clipped_action,
        raw_action,
        motor_targets,
        scaled_rewards,
        info,
        phase_id,
        gait_phase,
        q_ref,
        terms,
        foot_motion,
        done,
        swing_contact_count,
        swing_authority_gate,
        swing_residual_action_norm,
        requested_residual_offset_norm,
        effective_residual_offset_norm,
        target_clipped,
        target_clipped_steps,
        min_swing_contact_count_left,
        min_swing_contact_count_right,
        max_clearance_left,
        max_clearance_right,
        max_abs_roll_left,
        max_abs_roll_right,
    )
    components = self._v7_components(data, info, phase_id, terms, foot_motion)
    metrics["stability/stable_single_support"] = components["stable_single_support"]
    metrics["stability/post_touchdown_stability"] = components[
        "post_touchdown_stability"
    ]
    metrics["stability/touchdown_load_acceptance"] = components[
        "touchdown_load_acceptance"
    ]
    metrics["stability/left_touchdown_load_acceptance"] = components[
        "left_touchdown_load_acceptance"
    ]
    metrics["stability/right_touchdown_load_acceptance"] = components[
        "right_touchdown_load_acceptance"
    ]
    metrics["stability/locomotion_instability"] = components["locomotion_instability"]
    metrics["stability/upright_gate"] = components["v7_upright_gate"]
    metrics["stability/root_height_gate"] = components["v7_root_height_gate"]
    metrics["stability/angular_stability_gate"] = components[
        "v7_angular_stability_gate"
    ]
    metrics["gait/alternating_touchdown_count"] = components["alternating_touchdown"]
    metrics["gait/same_side_touchdown_repeat_count"] = components[
        "same_side_touchdown_repeat"
    ]
    metrics["gait/last_touchdown_side"] = components["last_touchdown_side"]
    for name in (
        "termination/fall_event",
        "termination/term_height",
        "termination/term_roll",
        "termination/term_pitch",
        "termination/terminal_root_height",
        "termination/terminal_abs_roll",
        "termination/terminal_abs_pitch",
        "termination/terminal_roll_pitch_angular_velocity",
    ):
        metrics[name] = components[name]
    metrics["termination/terminal_fall_cost"] = jp.zeros(())
    return metrics


def ResidualForwardWalkingV8__empty_metrics(self) -> dict[str, jax.Array]:
    metrics = self._invoke_source("ResidualForwardWalkingV7._empty_metrics")
    metrics.update(
        {
            name: jp.zeros(())
            for name in (
                "transition/raw_liftoff_left",
                "transition/raw_liftoff_right",
                "transition/qualified_liftoff_left",
                "transition/qualified_liftoff_right",
                "transition/raw_touchdown_left",
                "transition/raw_touchdown_right",
                "transition/qualified_touchdown_left",
                "transition/qualified_touchdown_right",
                "transition/left_no_contact_streak",
                "transition/right_no_contact_streak",
                "transition/left_contact_confirmation_streak",
                "transition/right_contact_confirmation_streak",
                "transition/left_max_air_height_gain",
                "transition/right_max_air_height_gain",
                "transition/recent_eligible_touchdown_step_count",
                "transition/qualified_alternating_touchdown_count",
                "transition/qualified_same_side_repeat_count",
                "transition/last_qualified_touchdown_side",
                "transition/support_transfer_progress",
                "transition/support_transfer_completion_count",
                "transition/support_transfer_timeout_count",
                "transition/left_transfer_completion_count",
                "transition/right_transfer_completion_count",
                "transition/landed_load_fraction",
                "transition/previous_support_load_fraction",
                "transition/left_landed_load_acceptance",
                "transition/right_landed_load_acceptance",
                "tracking/abs_root_y",
            )
        }
    )
    return metrics


def ResidualForwardWalkingV8__update_metrics(
    self,
    metrics: dict[str, jax.Array],
    data: mjx.Data,
    clipped_action: jax.Array,
    raw_action: jax.Array,
    motor_targets: jax.Array,
    scaled_rewards: dict[str, jax.Array],
    info: dict[str, Any],
    phase_id: jax.Array,
    gait_phase: jax.Array,
    q_ref: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
    done: jax.Array,
    swing_contact_count: jax.Array,
    swing_authority_gate: jax.Array,
    swing_residual_action_norm: jax.Array,
    requested_residual_offset_norm: jax.Array,
    effective_residual_offset_norm: jax.Array,
    target_clipped: jax.Array,
    target_clipped_steps: jax.Array,
    min_swing_contact_count_left: jax.Array,
    min_swing_contact_count_right: jax.Array,
    max_clearance_left: jax.Array,
    max_clearance_right: jax.Array,
    max_abs_roll_left: jax.Array,
    max_abs_roll_right: jax.Array,
) -> dict[str, jax.Array]:
    metrics = self._invoke_source(
        "ResidualForwardWalkingV7._update_metrics",
        metrics,
        data,
        clipped_action,
        raw_action,
        motor_targets,
        scaled_rewards,
        info,
        phase_id,
        gait_phase,
        q_ref,
        terms,
        foot_motion,
        done,
        swing_contact_count,
        swing_authority_gate,
        swing_residual_action_norm,
        requested_residual_offset_norm,
        effective_residual_offset_norm,
        target_clipped,
        target_clipped_steps,
        min_swing_contact_count_left,
        min_swing_contact_count_right,
        max_clearance_left,
        max_clearance_right,
        max_abs_roll_left,
        max_abs_roll_right,
    )
    components = self._v8_components(data, info, phase_id, terms, foot_motion)
    metrics["touchdown/forward_touchdown"] = components["forward_touchdown"]
    metrics["stability/post_touchdown_stability"] = components[
        "post_touchdown_stability"
    ]
    metrics["stability/touchdown_load_acceptance"] = components[
        "touchdown_load_acceptance"
    ]
    metrics["stability/left_touchdown_load_acceptance"] = components[
        "left_touchdown_load_acceptance"
    ]
    metrics["stability/right_touchdown_load_acceptance"] = components[
        "right_touchdown_load_acceptance"
    ]
    metrics["stability/locomotion_instability"] = components["locomotion_instability"]
    metrics["gait/alternating_touchdown_count"] = components[
        "qualified_alternating_touchdown"
    ]
    metrics["gait/same_side_touchdown_repeat_count"] = components[
        "qualified_same_side_repeat_count"
    ]
    metrics["gait/last_touchdown_side"] = components["last_qualified_touchdown_side"]
    metrics["transition/raw_liftoff_left"] = components["raw_liftoff_left"]
    metrics["transition/raw_liftoff_right"] = components["raw_liftoff_right"]
    metrics["transition/qualified_liftoff_left"] = components["qualified_liftoff_left"]
    metrics["transition/qualified_liftoff_right"] = components[
        "qualified_liftoff_right"
    ]
    metrics["transition/raw_touchdown_left"] = components["raw_touchdown_left"]
    metrics["transition/raw_touchdown_right"] = components["raw_touchdown_right"]
    metrics["transition/qualified_touchdown_left"] = components[
        "qualified_touchdown_left"
    ]
    metrics["transition/qualified_touchdown_right"] = components[
        "qualified_touchdown_right"
    ]
    metrics["transition/left_no_contact_streak"] = components["no_contact_streak_left"]
    metrics["transition/right_no_contact_streak"] = components[
        "no_contact_streak_right"
    ]
    metrics["transition/left_contact_confirmation_streak"] = components[
        "contact_streak_left"
    ]
    metrics["transition/right_contact_confirmation_streak"] = components[
        "contact_streak_right"
    ]
    metrics["transition/left_max_air_height_gain"] = components[
        "max_air_height_gain_left"
    ]
    metrics["transition/right_max_air_height_gain"] = components[
        "max_air_height_gain_right"
    ]
    metrics["transition/recent_eligible_touchdown_step_count"] = components[
        "recent_eligible_touchdown"
    ]
    metrics["transition/qualified_alternating_touchdown_count"] = components[
        "qualified_alternating_touchdown"
    ]
    metrics["transition/qualified_same_side_repeat_count"] = components[
        "qualified_same_side_repeat_count"
    ]
    metrics["transition/last_qualified_touchdown_side"] = components[
        "last_qualified_touchdown_side"
    ]
    metrics["transition/support_transfer_progress"] = components[
        "support_transfer_progress"
    ]
    metrics["transition/support_transfer_completion_count"] = components[
        "support_transfer_completion"
    ]
    metrics["transition/support_transfer_timeout_count"] = components[
        "support_transfer_timeout"
    ]
    metrics["transition/left_transfer_completion_count"] = components[
        "left_transfer_completion"
    ]
    metrics["transition/right_transfer_completion_count"] = components[
        "right_transfer_completion"
    ]
    metrics["transition/landed_load_fraction"] = components["landed_load_fraction"]
    metrics["transition/previous_support_load_fraction"] = components[
        "previous_support_load_fraction"
    ]
    metrics["transition/left_landed_load_acceptance"] = components[
        "left_touchdown_load_acceptance"
    ]
    metrics["transition/right_landed_load_acceptance"] = components[
        "right_touchdown_load_acceptance"
    ]
    abs_root_y = jp.abs(self.get_root_pos(data)[1] - info["root_y_baseline"])
    metrics["tracking/abs_root_y"] = abs_root_y
    return metrics


def ResidualForwardWalkingV9__empty_metrics(self) -> dict[str, jax.Array]:
    metrics = self._invoke_source("ResidualForwardWalkingV8._empty_metrics")
    metrics.update(
        {
            name: jp.zeros(())
            for name in (
                "transition/expected_side_qualified_liftoff_count",
                "transition/unexpected_side_qualified_liftoff_count",
                "transition/expected_side_qualified_touchdown_count",
                "transition/unexpected_side_qualified_touchdown_count",
                "transition/completed_lr_cycle_count",
                "transition/completed_rl_cycle_count",
                "transition/max_same_side_qualified_touchdown_streak",
                "transition/steps_to_expected_touchdown_sum",
                "transition/expected_touchdown_count_for_latency",
                "transition/rewarded_expected_swing_step_count",
                "transition/blocked_unexpected_swing_step_count",
            )
        }
    )
    return metrics


def ResidualForwardWalkingV9__update_metrics(
    self,
    metrics: dict[str, jax.Array],
    data: mjx.Data,
    clipped_action: jax.Array,
    raw_action: jax.Array,
    motor_targets: jax.Array,
    scaled_rewards: dict[str, jax.Array],
    info: dict[str, Any],
    phase_id: jax.Array,
    gait_phase: jax.Array,
    q_ref: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
    done: jax.Array,
    swing_contact_count: jax.Array,
    swing_authority_gate: jax.Array,
    swing_residual_action_norm: jax.Array,
    requested_residual_offset_norm: jax.Array,
    effective_residual_offset_norm: jax.Array,
    target_clipped: jax.Array,
    target_clipped_steps: jax.Array,
    min_swing_contact_count_left: jax.Array,
    min_swing_contact_count_right: jax.Array,
    max_clearance_left: jax.Array,
    max_clearance_right: jax.Array,
    max_abs_roll_left: jax.Array,
    max_abs_roll_right: jax.Array,
) -> dict[str, jax.Array]:
    metrics = self._invoke_source(
        "ResidualForwardWalkingV8._update_metrics",
        metrics,
        data,
        clipped_action,
        raw_action,
        motor_targets,
        scaled_rewards,
        info,
        phase_id,
        gait_phase,
        q_ref,
        terms,
        foot_motion,
        done,
        swing_contact_count,
        swing_authority_gate,
        swing_residual_action_norm,
        requested_residual_offset_norm,
        effective_residual_offset_norm,
        target_clipped,
        target_clipped_steps,
        min_swing_contact_count_left,
        min_swing_contact_count_right,
        max_clearance_left,
        max_clearance_right,
        max_abs_roll_left,
        max_abs_roll_right,
    )
    components = self._v9_components(data, info, phase_id, terms, foot_motion)
    metrics["transition/expected_side_qualified_liftoff_count"] = components[
        "v9_expected_side_qualified_liftoff_count"
    ]
    metrics["transition/unexpected_side_qualified_liftoff_count"] = components[
        "v9_unexpected_side_qualified_liftoff_count"
    ]
    metrics["transition/expected_side_qualified_touchdown_count"] = components[
        "v9_expected_side_qualified_touchdown_count"
    ]
    metrics["transition/unexpected_side_qualified_touchdown_count"] = components[
        "v9_unexpected_side_qualified_touchdown_count"
    ]
    metrics["transition/completed_lr_cycle_count"] = components[
        "v9_completed_lr_cycle_count"
    ]
    metrics["transition/completed_rl_cycle_count"] = components[
        "v9_completed_rl_cycle_count"
    ]
    next_max = components["next_v9_max_same_side_qualified_touchdown_streak"]
    previous_max = info["v9_max_same_side_qualified_touchdown_streak"]
    metrics["transition/max_same_side_qualified_touchdown_streak"] = jp.maximum(
        next_max - previous_max, 0.0
    )
    metrics["transition/steps_to_expected_touchdown_sum"] = components[
        "v9_steps_to_expected_touchdown_sum"
    ]
    metrics["transition/expected_touchdown_count_for_latency"] = components[
        "v9_expected_touchdown_count_for_latency"
    ]
    metrics["transition/rewarded_expected_swing_step_count"] = components[
        "v9_rewarded_expected_swing_step_count"
    ]
    metrics["transition/blocked_unexpected_swing_step_count"] = components[
        "v9_blocked_unexpected_swing_step_count"
    ]
    return metrics


def ResidualSteppingInPlaceV1__empty_metrics(self) -> dict[str, jax.Array]:
    names = (
        [f"reward/{name}" for name in self._REWARD_TERMS]
        + [
            f"penalty/{self._PENALTY_METRIC_NAMES.get(name, name)}"
            for name in self._PENALTY_TERMS
        ]
        + [
            "gait/phase",
            "gait/phase_sin",
            "gait/phase_cos",
            "gait/target_left_swing",
            "gait/target_right_swing",
            "gait/desired_left_contact",
            "gait/desired_right_contact",
            "gait/actual_left_contact",
            "gait/actual_right_contact",
            "gait/desired_single_support",
            "gait/actual_single_support",
            "gait/contact_pattern_match",
            "gait/swing_contact",
            "gait/double_support",
            "gait/feet_air_time_l",
            "gait/feet_air_time_r",
            "gait/left_foot_height",
            "gait/right_foot_height",
            "gait/left_foot_roll",
            "gait/left_foot_pitch",
            "gait/right_foot_roll",
            "gait/right_foot_pitch",
            "tracking/root_y",
            "tracking/root_roll",
            "tracking/root_pitch",
            "tracking/q_ref_error",
            "tracking/swing_leg_q_ref_error",
            "tracking/stance_leg_q_ref_error",
            "tracking/left_knee_extension",
            "tracking/right_knee_extension",
            "control/action_norm",
            "control/action_rate",
            "control/control_effort",
            "control/action_saturation",
            "control/ctrl_saturation",
            "control/force_saturation",
            "termination/done",
        ]
    )
    return {name: jp.zeros(()) for name in names}


def ResidualSteppingInPlaceV1__update_metrics(
    self,
    metrics: dict[str, jax.Array],
    data: mjx.Data,
    clipped_action: jax.Array,
    raw_action: jax.Array,
    motor_targets: jax.Array,
    scaled_rewards: dict[str, jax.Array],
    info: dict[str, Any],
    phase_id: jax.Array,
    gait_phase: jax.Array,
    q_ref: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
    done: jax.Array,
) -> dict[str, jax.Array]:
    for name in self._REWARD_TERMS:
        metrics[f"reward/{name}"] = scaled_rewards[name]
    for name in self._PENALTY_TERMS:
        tag = self._PENALTY_METRIC_NAMES.get(name, name)
        metrics[f"penalty/{tag}"] = scaled_rewards[name]
    roll, pitch = self._roll_pitch(data)
    root_y = self.get_root_pos(data)[1]
    joint_pos = data.qpos[self._qposadr_jp]
    left_roll, left_pitch, right_roll, right_pitch = self._foot_roll_pitch(data)
    left_contact = foot_motion["left_contact"]
    right_contact = foot_motion["right_contact"]
    actual_single_support = jp.float32(left_contact ^ right_contact)
    actual_double_support = jp.float32(left_contact & right_contact)
    left_height = foot_motion["left_pos"][2] - info["left_foot_baseline_z"]
    right_height = foot_motion["right_pos"][2] - info["right_foot_baseline_z"]
    desired_single_support = jp.float32(phase_id == 2)
    q_ref_error = jp.max(jp.abs(joint_pos - q_ref))
    target_left = info["target_left_swing"]
    target_right = info["target_right_swing"]
    desired_stance_contact = terms["support_contact"]
    desired_swing_no_contact = terms["target_no_contact"]
    contact_pattern_match = jp.where(
        desired_single_support > 0.5,
        desired_stance_contact * desired_swing_no_contact,
        actual_double_support,
    )
    raw_swing_contact = terms["target_contact"]
    gated_swing_contact = desired_single_support * raw_swing_contact
    swing_leg_mask = (
        target_left * self._left_leg_mask_jp + target_right * self._right_leg_mask_jp
    )
    stance_leg_mask = (
        target_left * self._right_leg_mask_jp + target_right * self._left_leg_mask_jp
    )
    abs_error = jp.abs(joint_pos - q_ref)
    swing_leg_q_ref_error = jp.sum(swing_leg_mask * abs_error) / 6.0
    stance_leg_q_ref_error = jp.sum(stance_leg_mask * abs_error) / 6.0
    force_abs = jp.abs(data.actuator_force)
    effort_abs = jp.maximum(jp.abs(self._effort_uppers), 1.0)
    force_norm = data.actuator_force / effort_abs
    metrics["gait/phase"] = gait_phase
    metrics["gait/phase_sin"] = jp.sin(2.0 * jp.pi * gait_phase)
    metrics["gait/phase_cos"] = jp.cos(2.0 * jp.pi * gait_phase)
    metrics["gait/target_left_swing"] = info["target_left_swing"]
    metrics["gait/target_right_swing"] = info["target_right_swing"]
    metrics["gait/desired_left_contact"] = info["desired_left_contact"]
    metrics["gait/desired_right_contact"] = info["desired_right_contact"]
    metrics["gait/actual_left_contact"] = jp.float32(left_contact)
    metrics["gait/actual_right_contact"] = jp.float32(right_contact)
    metrics["gait/desired_single_support"] = desired_single_support
    metrics["gait/actual_single_support"] = actual_single_support
    metrics["gait/contact_pattern_match"] = contact_pattern_match
    metrics["gait/swing_contact"] = gated_swing_contact
    metrics["gait/double_support"] = actual_double_support
    metrics["gait/feet_air_time_l"] = foot_motion["feet_air_time"][0]
    metrics["gait/feet_air_time_r"] = foot_motion["feet_air_time"][1]
    metrics["gait/left_foot_height"] = left_height
    metrics["gait/right_foot_height"] = right_height
    metrics["gait/left_foot_roll"] = left_roll
    metrics["gait/left_foot_pitch"] = left_pitch
    metrics["gait/right_foot_roll"] = right_roll
    metrics["gait/right_foot_pitch"] = right_pitch
    metrics["tracking/root_y"] = root_y
    metrics["tracking/root_roll"] = roll
    metrics["tracking/root_pitch"] = pitch
    metrics["tracking/q_ref_error"] = q_ref_error
    metrics["tracking/swing_leg_q_ref_error"] = swing_leg_q_ref_error
    metrics["tracking/stance_leg_q_ref_error"] = stance_leg_q_ref_error
    metrics["tracking/left_knee_extension"] = joint_pos[self._knee_l_idx]
    metrics["tracking/right_knee_extension"] = joint_pos[self._knee_r_idx]
    metrics["control/action_norm"] = jp.sqrt(jp.mean(jp.square(clipped_action)))
    metrics["control/action_rate"] = jp.sqrt(
        jp.mean(jp.square(clipped_action - info["last_act"]))
    )
    metrics["control/control_effort"] = jp.sqrt(jp.mean(jp.square(force_norm)))
    metrics["control/action_saturation"] = jp.mean(raw_action != clipped_action)
    metrics["control/ctrl_saturation"] = jp.mean(
        (motor_targets <= self._joint_lowers) | (motor_targets >= self._joint_uppers)
    )
    metrics["control/force_saturation"] = jp.mean(force_abs >= 0.99 * effort_abs)
    metrics["termination/done"] = done
    return metrics


def ResidualSteppingInPlaceV3__empty_metrics(self) -> dict[str, jax.Array]:
    metrics = self._invoke_source("ResidualSteppingInPlaceV1._empty_metrics")
    metrics.update(
        {
            name: jp.zeros(())
            for name in (
                "gait/swing_contact_count",
                "gait/swing_contact_fraction",
                "gait/contact_reduction_progress",
                "gait/flat_clearance_progress",
                "gait/actual_single_support_left",
                "gait/actual_single_support_right",
                "gait/min_swing_contact_count_left",
                "gait/min_swing_contact_count_right",
                "gait/max_clearance_left",
                "gait/max_clearance_right",
                "gait/max_abs_roll_left",
                "gait/max_abs_roll_right",
                "control/swing_authority_gate",
                "control/swing_residual_action_norm",
                "control/requested_residual_offset_norm",
                "control/effective_residual_offset_norm",
                "control/target_clipped",
                "control/target_clipped_steps",
            )
        }
    )
    return metrics


def ResidualSteppingInPlaceV3__update_metrics(
    self,
    metrics: dict[str, jax.Array],
    data: mjx.Data,
    clipped_action: jax.Array,
    raw_action: jax.Array,
    motor_targets: jax.Array,
    scaled_rewards: dict[str, jax.Array],
    info: dict[str, Any],
    phase_id: jax.Array,
    gait_phase: jax.Array,
    q_ref: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
    done: jax.Array,
    swing_contact_count: jax.Array,
    swing_authority_gate: jax.Array,
    swing_residual_action_norm: jax.Array,
    requested_residual_offset_norm: jax.Array,
    effective_residual_offset_norm: jax.Array,
    target_clipped: jax.Array,
    target_clipped_steps: jax.Array,
    min_swing_contact_count_left: jax.Array,
    min_swing_contact_count_right: jax.Array,
    max_clearance_left: jax.Array,
    max_clearance_right: jax.Array,
    max_abs_roll_left: jax.Array,
    max_abs_roll_right: jax.Array,
) -> dict[str, jax.Array]:
    metrics = self._invoke_source(
        "ResidualSteppingInPlaceV1._update_metrics",
        metrics,
        data,
        clipped_action,
        raw_action,
        motor_targets,
        scaled_rewards,
        info,
        phase_id,
        gait_phase,
        q_ref,
        terms,
        foot_motion,
        done,
    )
    left_contact = foot_motion["left_contact"]
    right_contact = foot_motion["right_contact"]
    contact_fraction = jp.clip(
        swing_contact_count / self._config.contact_reduction_nominal_count, 0.0, 1.0
    )
    progress_window = jp.float32((phase_id == 1) | (phase_id == 2))
    target_left = info["target_left_swing"]
    target_right = info["target_right_swing"]
    left_roll, left_pitch, right_roll, right_pitch = self._foot_roll_pitch(data)
    swing_roll = target_left * left_roll + target_right * right_roll
    swing_pitch = target_left * left_pitch + target_right * right_pitch
    metrics["gait/swing_contact_count"] = swing_contact_count
    metrics["gait/swing_contact_fraction"] = contact_fraction
    metrics["gait/contact_reduction_progress"] = _contact_reduction_progress_raw(
        swing_contact_count,
        progress_window,
        nominal_count=self._config.contact_reduction_nominal_count,
    )
    metrics["gait/flat_clearance_progress"] = _flat_clearance_progress_raw(
        terms["target_clearance_raw"],
        swing_roll,
        swing_pitch,
        progress_window,
        clearance_min=self._config.flat_clearance_min,
        clearance_max=self._config.flat_clearance_max,
        orientation_scale=self._config.flat_clearance_orientation_scale,
    )
    metrics["gait/actual_single_support_left"] = jp.float32(
        left_contact & ~right_contact
    )
    metrics["gait/actual_single_support_right"] = jp.float32(
        right_contact & ~left_contact
    )
    metrics["gait/min_swing_contact_count_left"] = min_swing_contact_count_left
    metrics["gait/min_swing_contact_count_right"] = min_swing_contact_count_right
    metrics["gait/max_clearance_left"] = max_clearance_left
    metrics["gait/max_clearance_right"] = max_clearance_right
    metrics["gait/max_abs_roll_left"] = max_abs_roll_left
    metrics["gait/max_abs_roll_right"] = max_abs_roll_right
    metrics["control/swing_authority_gate"] = swing_authority_gate
    metrics["control/swing_residual_action_norm"] = swing_residual_action_norm
    metrics["control/requested_residual_offset_norm"] = requested_residual_offset_norm
    metrics["control/effective_residual_offset_norm"] = effective_residual_offset_norm
    metrics["control/target_clipped"] = target_clipped
    metrics["control/target_clipped_steps"] = target_clipped_steps
    return metrics


def SchaftUnifiedJoystickV1__empty_metrics(self) -> Dict[str, jax.Array]:
    metrics = self._invoke_source("ResidualForwardWalkingV20._empty_metrics")
    metrics.update(
        {
            "unified/locomotion_request": jp.zeros(()),
            "unified/reward_always": jp.zeros(()),
            "unified/reward_motion": jp.zeros(()),
            "unified/reward_quiet": jp.zeros(()),
            "unified/e_path_x": jp.zeros(()),
            "unified/e_path_y": jp.zeros(()),
            "unified/forward_progress": jp.zeros(()),
            "unified/heading_target": jp.zeros(()),
            "unified/command_vx": jp.zeros(()),
            "unified/command_vy": jp.zeros(()),
            "unified/command_wz": jp.zeros(()),
            "unified/transition_progress": jp.zeros(()),
            "unified/start_swing_side": jp.zeros(()),
            "unified/raw_wz": jp.zeros(()),
            "unified/wz_dc": jp.zeros(()),
            "unified/wz_ac": jp.zeros(()),
            "unified/yaw_unwrapped": jp.zeros(()),
            "unified/heading_response_ratio": jp.zeros(()),
            "unified/under_response_contribution": jp.zeros(()),
            "transition/post_liftoff_touchdown_count": jp.zeros(()),
            "transition/alternating_touchdown_bridge_count": jp.zeros(()),
            "transition/u6_valid_command_step_completion_count": jp.zeros(()),
            "reward/u6_post_liftoff_touchdown_bridge": jp.zeros(()),
            "reward/u6_alternating_touchdown_bridge": jp.zeros(()),
            "reward/u6_valid_command_step_completion": jp.zeros(()),
            "transition/u6_raw_touchdown_count": jp.zeros(()),
            "transition/u6_raw_touchdown_left_count": jp.zeros(()),
            "transition/u6_raw_touchdown_right_count": jp.zeros(()),
            "transition/u6_td_rejected_min_air": jp.zeros(()),
            "transition/u6_td_rejected_stability": jp.zeros(()),
            "transition/u6_td_rejected_slip": jp.zeros(()),
            "transition/u6_td_rejected_quiet": jp.zeros(()),
            "transition/u6_side_bootstrap_count": jp.zeros(()),
            "transition/u6_same_side_rejected_count": jp.zeros(()),
            "transition/u6_simultaneous_rejected_count": jp.zeros(()),
            "transition/u6_expected_liftoff_commit_count": jp.zeros(()),
            "transition/u6_expected_liftoff_stale_refund_count": jp.zeros(()),
            "reward/u6_expected_liftoff_stale_refund": jp.zeros(()),
            "transition/u6_expected_foot_landed_count": jp.zeros(()),
            "transition/u6_side_bootstrap_bridge_count": jp.zeros(()),
            "transition/u6_credit_event_count": jp.zeros(()),
            "reward/u6_expected_liftoff_commit": jp.zeros(()),
            "reward/u6_side_bootstrap_bridge": jp.zeros(()),
            "transition/u6_expected_side_left": jp.zeros(()),
            "transition/u6_expected_side_right": jp.zeros(()),
            "transition/u6_alternation_latch": jp.zeros(()),
            "transition/u6_same_side_ll_count": jp.zeros(()),
            "transition/u6_same_side_rr_count": jp.zeros(()),
            "transition/u6_opposite_lr_count": jp.zeros(()),
            "transition/u6_opposite_rl_count": jp.zeros(()),
            "transition/u6_liftoff_rejected_already_credited": jp.zeros(()),
            "transition/u6_opp_td_rejected_min_air": jp.zeros(()),
            "transition/u6_opp_td_rejected_stability": jp.zeros(()),
            "transition/u6_opp_td_rejected_slip": jp.zeros(()),
            "transition/u6_opp_td_rejected_quiet": jp.zeros(()),
            "transition/u6_opp_td_rejected_side_memory": jp.zeros(()),
            "transition/u6_completion_rejected_no_alternation": jp.zeros(()),
            "transition/u6_expected_load_ratio": jp.zeros(()),
            "transition/u6_expected_unload_progress": jp.zeros(()),
            "transition/u6_expected_reload_regress": jp.zeros(()),
            "transition/u6_expected_foot_still_loaded": jp.zeros(()),
            "transition/u6_expected_left_unload_crossing": jp.zeros(()),
            "transition/u6_expected_right_unload_crossing": jp.zeros(()),
            "transition/u6_expected_air_ticks_1_4": jp.zeros(()),
            "transition/u6_expected_air_ticks_ge_5": jp.zeros(()),
            "transition/u6_first_liftoff_bootstrap_count": jp.zeros(()),
            "transition/u6_liftoff_armed_fraction": jp.zeros(()),
            "transition/u6_commit_outstanding_fraction": jp.zeros(()),
            "transition/u6_opposite_touchdown_count": jp.zeros(()),
            "transition/u6_gait_phase_wait_liftoff": jp.zeros(()),
            "transition/u6_gait_phase_wait_touchdown": jp.zeros(()),
            "transition/u6_gait_phase_wait_completion": jp.zeros(()),
            "transition/u6_command_reward_gate_mean": jp.zeros(()),
            "transition/u6v7_nonquiet_zero_motion_baseline": jp.zeros(()),
            "transition/u6v7_nonquiet_corrected_tracking": jp.zeros(()),
            "reward/u6_expected_unload_progress": jp.zeros(()),
            "reward/u6_expected_foot_still_loaded": jp.zeros(()),
            "unified/wz_safety_gate": jp.zeros(()),
        }
    )
    metrics.update({key: jp.zeros(()) for key in U4_UNIFIED_METRIC_KEYS})
    metrics.update({key: jp.zeros(()) for key in U4_STRICT_YAW_DC_METRIC_KEYS})
    if self._u6v10_unified_opposite_touchdown:
        metrics["reward/u6_opposite_touchdown_bridge"] = jp.zeros(())
    if self._v11_minimal_gait_core:
        for key in self._v11_metric_keys():
            metrics[key] = jp.zeros(())
    if self._u6v10_reward_accounting:
        for key in self._u6v10_accounting_metric_keys():
            metrics[key] = jp.zeros(())
    if self._u13_telemetry:
        for key in self._U13_METRIC_KEYS:
            metrics[key] = jp.zeros(())
    return metrics


def SchaftUnifiedJoystickV1__strict_yaw_dc_metrics(
    self,
    cmd,
    raw_wz,
    wz_dc,
    wz_ac,
    mvx,
    e_path_y,
    heading_ratio,
    heading_valid,
    components,
    info,
    fall_event,
) -> Dict[str, jax.Array]:
    """U4_STRICT_SIGNED_YAW_DC_ACQUISITION telemetry (§ "Direct normalized
    telemetry"): every "mean"/"fraction" tag is divided by a FIXED,
    schedule-known tick count (see the `STRICT_YAW_*_TICKS` constants in
    unified_command_process.py) before being written, so what Brax's
    `EvalWrapper` sums over the episode is already the normalized value --
    not a raw numerator a caller must remember to divide."""
    n_active = float(ucp.STRICT_YAW_ACTIVE_TICKS)
    n_heading = float(ucp.STRICT_YAW_HEADING_VALID_TICKS)
    n_post = float(ucp.STRICT_YAW_POST_TURN_WINDOW_TICKS)
    cmd_wz = cmd[2]
    delta_wz = self._dense_delta[2]
    gate_pos = (cmd_wz > delta_wz).astype(jp.float32)
    gate_neg = (cmd_wz < -delta_wz).astype(jp.float32)
    gate_active = jp.maximum(gate_pos, gate_neg)
    heading_valid_f = heading_valid.astype(jp.float32)
    wrong_sign = jp.logical_and(
        jp.sign(wz_dc) != jp.sign(cmd_wz), jp.abs(wz_dc) > 1e-06
    ).astype(jp.float32)
    signed_ratio = components["u4_wz_signed_ratio"]

    def _sided(gate, divisor):
        return {
            "mean_command_wz": gate * cmd_wz / divisor,
            "mean_wz_dc": gate * wz_dc / divisor,
            "mean_raw_wz": gate * raw_wz / divisor,
            "signed_response_ratio": gate * signed_ratio / divisor,
            "wrong_sign_fraction": gate * wrong_sign / divisor,
            "valid_count": gate,
        }

    metrics = {f"yaw_dc_pos/{k}": v for k, v in _sided(gate_pos, n_active).items()}
    metrics.update(
        {f"yaw_dc_neg/{k}": v for k, v in _sided(gate_neg, n_active).items()}
    )
    metrics["yaw_dc_pos/heading_response_ratio"] = (
        gate_pos * heading_valid_f * heading_ratio / n_heading
    )
    metrics["yaw_dc_neg/heading_response_ratio"] = (
        gate_neg * heading_valid_f * heading_ratio / n_heading
    )
    stability_gate = (
        components["v7_upright_gate"]
        * components["v7_root_height_gate"]
        * components["v7_angular_stability_gate"]
    )
    speed_ramp = jp.clip((mvx - 0.03) / (0.05 - 0.03), 0.0, 1.0)
    same_side_ok = 1.0 - components["qualified_same_side_repeat"]
    transfer_ok = 1.0 - components["support_transfer_timeout"]
    safety_gate = stability_gate * speed_ramp * same_side_ok * transfer_ok
    positive_reward_scale = jp.where(
        gate_active > 0.5, self._active_yaw_other_positive_scale, 1.0
    )
    under_response_reward = components["u4_wz_under_response_penalty"]
    metrics.update(
        {
            "yaw_dc/mean_abs_wz_ac": gate_active * jp.abs(wz_ac) / n_active,
            "yaw_dc/under_response_fraction": gate_active
            * (signed_ratio < 0.35).astype(jp.float32)
            / n_active,
            "yaw_dc/under_response_reward_mean": gate_active
            * under_response_reward
            / n_active,
            "yaw_dc/under_response_reward_sum": gate_active * under_response_reward,
            "yaw_dc/heading_progress_reward_mean": gate_active
            * components["heading_progress_reward"]
            / n_active,
            "yaw_dc/mean_forward_velocity": gate_active * mvx / n_active,
            "yaw_dc/double_support_fraction": gate_active
            * components.get("quiet_double_support", jp.zeros(()))
            / n_active,
            "yaw_dc/cross_track_abs": gate_active * jp.abs(e_path_y) / n_active,
            "yaw_dc/fall_fraction": gate_active * fall_event / n_active,
            "yaw_dc/support_timeout_fraction": gate_active
            * components.get("support_transfer_timeout", jp.zeros(()))
            / n_active,
            "yaw_dc/same_side_repeat_fraction": gate_active
            * components.get("qualified_same_side_repeat", jp.zeros(()))
            / n_active,
            "yaw_dc/positive_reward_scale_mean": gate_active
            * positive_reward_scale
            / n_active,
            "yaw_dc/safety_gate_mean": gate_active * safety_gate / n_active,
            "yaw_dc/valid_count": gate_active,
        }
    )
    post_turn_ticks = info["u4_post_turn_ticks"]
    in_window = jp.logical_and(
        post_turn_ticks > 0, post_turn_ticks <= ucp.STRICT_YAW_POST_TURN_WINDOW_TICKS
    ).astype(jp.float32)
    settled_now = (jp.abs(wz_dc) < 0.03).astype(jp.float32)
    first_settle = in_window * settled_now * (1.0 - info["u4_post_turn_settled"])
    settling_latency = first_settle * post_turn_ticks.astype(jp.float32) * self.dt
    metrics.update(
        {
            "post_turn_dc/mean_abs_wz_dc": in_window * jp.abs(wz_dc) / n_post,
            "post_turn_dc/settled_fraction": in_window * settled_now / n_post,
            "post_turn_dc/settling_latency": settling_latency,
            "post_turn_dc/valid_count": in_window,
        }
    )
    return metrics


def SchaftUnifiedJoystickV1__u13_metrics(
    self,
    *,
    raw_transfer_completion,
    gated_transfer_completion,
    prev_expected_side,
    next_expected_side,
    transfer_timeout,
    components,
    scaled,
    gate,
) -> Dict[str, jax.Array]:
    """U13 FSM-integrity + precursor-farming telemetry.

    Every FSM quantity here is read from the values the V8/V9 machine itself
    consumed, never from a reward component that a command gate has already
    scaled. That is the whole point: the historical
    ``v18_side_change_without_completion`` metric divides an ungated FSM
    transition by a gated reward value and therefore reports ``1 - gate`` on
    perfectly legal completions.
    """
    known_now = (jp.abs(prev_expected_side) > 0.5).astype(jp.float32)
    known_next = (jp.abs(next_expected_side) > 0.5).astype(jp.float32)
    changed = (
        known_now
        * known_next
        * (next_expected_side != prev_expected_side).astype(jp.float32)
    )
    completed = (raw_transfer_completion > 0.5).astype(jp.float32)
    bootstrap = (
        (1.0 - known_now) * known_next * (transfer_timeout > 0.5).astype(jp.float32)
    )
    eligible = components["eligible_touchdown"]
    timeout_ev = (transfer_timeout > 0.5).astype(jp.float32)
    precursor_actual = jp.zeros(())
    for name in self._U13_PRECURSOR_TERMS:
        precursor_actual = precursor_actual + gate * scaled[name] * self.dt
    timeout_actual = gate * scaled["support_transfer_timeout"] * self.dt
    return {
        "u13/raw_support_transfer_completion": raw_transfer_completion,
        "u13/reward_gated_support_transfer_completion": gated_transfer_completion,
        "u13/side_change_after_raw_transfer_completion": changed * completed,
        "u13/side_change_without_raw_transfer_completion": changed * (1.0 - completed),
        "u13/bootstrap_expected_side_initialization": bootstrap,
        "u13/expected_side": next_expected_side,
        "u13/eligible_touchdown_events": eligible,
        "u13/transfer_completion_events": completed,
        "u13/transfer_timeout_events": timeout_ev,
        "u13/precursor_reward_actual_in_window": precursor_actual,
        "u13/precursor_reward_actual_on_timeout": timeout_ev * precursor_actual,
        "u13/timeout_penalty_actual": timeout_actual,
        "u13/aborted_window_net_reward_actual": timeout_ev
        * (precursor_actual + timeout_actual),
        "u13/track_planar_raw": components["u13_track_planar_raw"],
        "u13/track_yaw_raw": components["u13_track_yaw_raw"],
    }


def SchaftUnifiedJoystickV1__u4_unified_metrics(
    self, cmd, gyro_z, mvx, e_path_y, heading_error, components, info, fall_event
):
    """Category-conditioned telemetry (§ 'Direct physical telemetry').

    Every group emits gated per-tick NUMERATORS plus an explicit
    ``<group>/valid_count`` DENOMINATOR (never a raw episode sum treated as
    a mean): Brax's eval aggregator sums metrics over the episode/eval
    batch (the same convention `alternating_touchdown_count` etc. already
    rely on), so ``episode_<group>/x`` divided by
    ``episode_<group>/valid_count`` downstream is the correct conditional
    mean.

    Returns ``(metrics, new_post_turn_ticks, new_post_turn_settled)`` --
    the latter two must be written back into ``info`` by the caller.
    """
    delta_wz = self._dense_delta[2]
    cmd_wz = cmd[2]
    gate_pos = (cmd_wz > delta_wz).astype(jp.float32)
    gate_neg = (cmd_wz < -delta_wz).astype(jp.float32)
    gate_active = jp.maximum(gate_pos, gate_neg)
    signed_ratio = components["u4_wz_signed_ratio"]
    wrong_sign = jp.logical_and(
        jp.sign(gyro_z) != jp.sign(cmd_wz), jp.abs(gyro_z) > 1e-06
    ).astype(jp.float32)
    tracking_error = jp.abs(gyro_z - cmd_wz)
    metrics = {
        "yaw_pos/command_wz": gate_pos * cmd_wz,
        "yaw_pos/measured_wz": gate_pos * gyro_z,
        "yaw_pos/signed_response_ratio": gate_pos * signed_ratio,
        "yaw_pos/tracking_error": gate_pos * tracking_error,
        "yaw_pos/wrong_sign_fraction": gate_pos * wrong_sign,
        "yaw_pos/valid_count": gate_pos,
        "yaw_neg/command_wz": gate_neg * cmd_wz,
        "yaw_neg/measured_wz": gate_neg * gyro_z,
        "yaw_neg/signed_response_ratio": gate_neg * signed_ratio,
        "yaw_neg/tracking_error": gate_neg * tracking_error,
        "yaw_neg/wrong_sign_fraction": gate_neg * wrong_sign,
        "yaw_neg/valid_count": gate_neg,
        "yaw/under_response_fraction": gate_active
        * (signed_ratio < 0.35).astype(jp.float32),
        "yaw/mean_forward_velocity": gate_active * mvx,
        "yaw/cross_track_abs": gate_active * jp.abs(e_path_y),
        "yaw/heading_error": gate_active * jp.abs(heading_error),
        "yaw/double_support_fraction": gate_active
        * components.get("quiet_double_support", jp.zeros(())),
        "yaw/alternating_touchdown_rate": gate_active
        * components.get("alternating_touchdown", jp.zeros(())),
        "yaw/fall_fraction": gate_active * fall_event,
        "yaw/support_timeout_fraction": gate_active
        * components.get("support_transfer_timeout", jp.zeros(())),
        "yaw/same_side_repeat_fraction": gate_active
        * components.get("qualified_same_side_repeat", jp.zeros(())),
        "yaw/valid_count": gate_active,
    }
    abs_wz = jp.abs(cmd_wz)
    is_008 = gate_active * (abs_wz < 0.11).astype(jp.float32)
    is_014 = gate_active * jp.logical_and(abs_wz >= 0.11, abs_wz < 0.17).astype(
        jp.float32
    )
    is_020 = gate_active * (abs_wz >= 0.17).astype(jp.float32)
    measured_abs = jp.abs(gyro_z)
    metrics.update(
        {
            "yaw_mag_008/measured_abs_wz": is_008 * measured_abs,
            "yaw_mag_008/tracking_error": is_008 * tracking_error,
            "yaw_mag_008/valid_count": is_008,
            "yaw_mag_014/measured_abs_wz": is_014 * measured_abs,
            "yaw_mag_014/tracking_error": is_014 * tracking_error,
            "yaw_mag_014/valid_count": is_014,
            "yaw_mag_020/measured_abs_wz": is_020 * measured_abs,
            "yaw_mag_020/tracking_error": is_020 * tracking_error,
            "yaw_mag_020/valid_count": is_020,
        }
    )
    vx_cmd = cmd[0]
    cmd_curvature = cmd_wz / jp.maximum(jp.abs(vx_cmd), 0.001)
    measured_curvature = gyro_z / jp.maximum(jp.abs(mvx), 0.001)
    curvature_gate = gate_active * (jp.abs(vx_cmd) > 0.01).astype(jp.float32)
    metrics.update(
        {
            "curvature/command_radius": curvature_gate
            * jp.abs(vx_cmd)
            / jp.maximum(abs_wz, 0.001),
            "curvature/measured_curvature": curvature_gate * measured_curvature,
            "curvature/absolute_error": curvature_gate
            * jp.abs(measured_curvature - cmd_curvature),
            "curvature/valid_count": curvature_gate,
        }
    )
    post_turn_ticks_prev = info["u4_post_turn_ticks"]
    post_turn_ticks_new = jp.where(
        gate_active > 0.5, 0, post_turn_ticks_prev + 1
    ).astype(jp.int32)
    in_release_window = jp.logical_and(
        post_turn_ticks_new > 0, post_turn_ticks_new <= 50
    ).astype(jp.float32)
    settled_now = (jp.abs(gyro_z) < 0.03).astype(jp.float32)
    settled_prev = info["u4_post_turn_settled"]
    first_settle = in_release_window * settled_now * (1.0 - settled_prev)
    settled_new = jp.where(
        gate_active > 0.5, 0.0, jp.where(first_settle > 0.5, 1.0, settled_prev)
    )
    metrics.update(
        {
            "post_turn/mean_abs_wz": in_release_window * jp.abs(gyro_z),
            "post_turn/settling_latency": first_settle
            * post_turn_ticks_new.astype(jp.float32),
            "post_turn/settle_event_count": first_settle,
            "post_turn/settled_fraction": in_release_window * settled_now,
            "post_turn/mean_forward_velocity": in_release_window * mvx,
            "post_turn/valid_count": in_release_window,
        }
    )
    return (metrics, post_turn_ticks_new, settled_new)


def SchaftUnifiedJoystickV1__u6v10_accounting_metrics(
    self,
    components: Dict[str, jax.Array],
    scaled: Dict[str, jax.Array],
    gate: jax.Array,
    group_totals: Dict[str, jax.Array],
    frozen_clipped: jax.Array,
    actual: Dict[str, jax.Array],
    reward: jax.Array,
) -> Dict[str, jax.Array]:
    """THE single actual-contribution accounting path (U6 V10).

    Three unambiguous semantics per active frozen-matrix term:

      ``reward/<name>``        legacy, untouched: coefficient x raw component,
                               BEFORE the group gate and before ``dt``;
      ``reward_raw/<name>``    the unweighted component;
      ``reward_actual/<name>`` group_gate x coefficient x raw x ``dt`` -- what
                               that term actually put into the scalar reward,
                               PRE-clip.

    ``_assemble_reward`` clips the ASSEMBLED group total, not the individual
    terms, so the per-term sum is exported pre-clip and the single clip
    adjustment is exported separately; the two together reproduce the frozen
    block's real contribution exactly.

    ``actual`` carries the direct (non-frozen-matrix) contributions, each of
    which is the very array ``step()`` added to the reward, so nothing here
    re-derives a reward formula.
    """
    out: Dict[str, jax.Array] = {}
    preclip = jp.zeros(())
    for name in self._u6v10_active_terms:
        contribution = (
            self._u6v10_group_gate(ucfg.TERM_GROUP[name], gate) * scaled[name] * self.dt
        )
        out[f"reward_raw/{name}"] = components[name]
        out[f"reward_weighted_pre_dt/{name}"] = scaled[name]
        out[f"reward_actual/{name}"] = contribution
        preclip = preclip + contribution
    for name, contribution in actual.items():
        out[f"reward_actual/{name}"] = contribution
    direct_total = jp.zeros(())
    for name, contribution in actual.items():
        if name == "terminal_fall_cost":
            continue
        direct_total = direct_total + contribution
    out["reward_actual/_frozen_preclip_total"] = preclip
    out["reward_actual/_frozen_clipped_total"] = frozen_clipped
    out["reward_actual/_frozen_clip_adjustment"] = frozen_clipped - preclip
    out["reward_actual/_group_always"] = group_totals["ALWAYS"] * self.dt
    out["reward_actual/_group_motion"] = gate * group_totals["MOTION"] * self.dt
    out["reward_actual/_group_quiet"] = (1.0 - gate) * group_totals["QUIET"] * self.dt
    out["reward_actual/_direct_total"] = direct_total
    out["reward_actual/_terminal_total"] = actual["terminal_fall_cost"]
    out["reward_actual/_step_reward"] = reward
    out["reward_actual/_identity_residual"] = reward - (
        frozen_clipped + direct_total + actual["terminal_fall_cost"]
    )
    return out


def SchaftUnifiedJoystickV1__unified_metrics(
    self,
    cmd_state,
    group_totals,
    gate,
    e_path_x,
    e_path_y,
    fwd_progress,
    raw_wz=None,
    wz_dc=None,
    wz_ac=None,
    yaw_unwrapped=None,
    heading_response_ratio_value=None,
    under_response_contribution=None,
    wz_safety_gate=None,
) -> Dict[str, jax.Array]:
    """`raw_wz`/`wz_dc`/`wz_ac`/`yaw_unwrapped`/`heading_response_ratio_value`
    are computed UNCONDITIONALLY every tick for every stage (§ "Low-
    frequency yaw state"/"Accumulated heading progress") and exposed here,
    in the existing always-on `unified/*` telemetry namespace, purely so
    the deterministic evaluator's per-tick CSV (which reads `state.metrics`
    directly, not through Brax's sum-over-episode `EvalWrapper`) can carry
    them as plain per-tick columns for U4_STRICT_SIGNED_YAW_DC_ACQUISITION's
    validation package -- optional kwargs (default 0) so every existing
    call site/test keeps working unchanged."""
    zero = jp.zeros(())
    return {
        "unified/locomotion_request": gate,
        "unified/reward_always": group_totals["ALWAYS"],
        "unified/reward_motion": group_totals["MOTION"],
        "unified/reward_quiet": group_totals["QUIET"],
        "unified/e_path_x": e_path_x,
        "unified/e_path_y": e_path_y,
        "unified/forward_progress": fwd_progress,
        "unified/heading_target": cmd_state.heading_target,
        "unified/command_vx": cmd_state.command[0],
        "unified/command_vy": cmd_state.command[1],
        "unified/command_wz": cmd_state.command[2],
        "unified/transition_progress": ucp.transition_progress(cmd_state),
        "unified/start_swing_side": cmd_state.start_swing_side,
        "unified/raw_wz": raw_wz if raw_wz is not None else zero,
        "unified/wz_dc": wz_dc if wz_dc is not None else zero,
        "unified/wz_ac": wz_ac if wz_ac is not None else zero,
        "unified/yaw_unwrapped": yaw_unwrapped if yaw_unwrapped is not None else zero,
        "unified/heading_response_ratio": heading_response_ratio_value
        if heading_response_ratio_value is not None
        else zero,
        "unified/under_response_contribution": under_response_contribution
        if under_response_contribution is not None
        else zero,
        "unified/wz_safety_gate": wz_safety_gate
        if wz_safety_gate is not None
        else zero,
    }


def _contact_reduction_progress_raw(
    swing_contact_count: jax.Array,
    progress_window: jax.Array,
    nominal_count: float = 4.0,
) -> jax.Array:
    """Dense contact-reduction progress, gated to `progress_window`.

    contact_fraction = clip(swing_contact_count / nominal_count, 0, 1)
    returns progress_window * (1 - contact_fraction)^2, so it is 0.0 at the
    nominal full-contact count and 1.0 at zero contact, matching the V3 spec's
    expected raw values (4->0.0, 3->0.0625, 2->0.25, 1->0.5625, 0->1.0 at
    nominal_count=4).
    """
    contact_fraction = jp.clip(swing_contact_count / nominal_count, 0.0, 1.0)
    return progress_window * jp.square(1.0 - contact_fraction)


def _flat_clearance_progress_raw(
    swing_clearance: jax.Array,
    foot_roll: jax.Array,
    foot_pitch: jax.Array,
    progress_window: jax.Array,
    clearance_min: float = 0.001,
    clearance_max: float = 0.02,
    orientation_scale: float = 0.08,
) -> jax.Array:
    """Dense flat-clearance progress: rewards clearance only when the swing
    foot is close to level (small roll/pitch), so it cannot be gamed by
    rolling onto an edge to fake height. Gated to `progress_window`.
    """
    normalized_clearance = jp.clip(
        (swing_clearance - clearance_min) / (clearance_max - clearance_min), 0.0, 1.0
    )
    smooth_clearance = jp.square(normalized_clearance) * (
        3.0 - 2.0 * normalized_clearance
    )
    orientation_gate = jp.exp(
        -jp.square(foot_roll / orientation_scale)
        - jp.square(foot_pitch / orientation_scale)
    )
    return progress_window * smooth_clearance * orientation_gate
