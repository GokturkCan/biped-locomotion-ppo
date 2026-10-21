# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0. See LICENSE.
# Adapted from the private research working tree based on MuJoCo Playground.
# Robot descriptions and model parameters are intentionally excluded.

from __future__ import annotations
from typing import Any, Dict, Tuple
import jax
import jax.numpy as jp
from biped_locomotion.commands import process as ucp
from biped_locomotion.rewards import unified_contract as ucfg
from biped_locomotion.gait.unified_events import (
    BOUT_LOCAL_BILATERAL_FIELDS,
    QUIET_FROZEN_INFO_FIELDS,
)


def SchaftUnifiedJoystickV1__apply_canonical_start(
    self, info: Dict[str, Any], cmd_state: ucp.CommandState, rising_edge: jax.Array
) -> Dict[str, Any]:
    """§9.2 canonical prep state on a QUIET->MOVING edge."""
    out = dict(info)
    zero = jp.zeros(())
    out["v9_expected_side"] = jp.where(
        rising_edge, cmd_state.start_swing_side, info["v9_expected_side"]
    )
    out["v9_transfer_pending"] = jp.where(
        rising_edge, zero, info["v9_transfer_pending"]
    )
    out["v9_steps_since_expected_side_set"] = jp.where(
        rising_edge, zero, info["v9_steps_since_expected_side_set"]
    )
    out["v9_pending_landed_side"] = jp.where(
        rising_edge, zero, info["v9_pending_landed_side"]
    )
    for side in ("left", "right"):
        for key in (
            f"v8_consecutive_no_contact_steps_{side}",
            f"v8_max_height_gain_during_air_{side}",
            f"v8_air_qualified_{side}",
            f"v8_consecutive_contact_steps_{side}",
        ):
            out[key] = jp.where(rising_edge, zero, info[key])
    for key in (
        "v16_attempt_active",
        "v16_attempt_side",
        "v16_single_support_steps",
        "v16_touchdown_latched",
    ):
        out[key] = jp.where(rising_edge, zero, info[key])
    return out


def SchaftUnifiedJoystickV1__apply_quiet_freeze(
    self, new_info: Dict[str, Any], prev_info: Dict[str, Any], gate: jax.Array
) -> Dict[str, Any]:
    """Hold exactly :data:`QUIET_FROZEN_INFO_FIELDS` while the gate is low.

    Exact-name matching only; nothing physical is touched.
    """
    quiet = gate < 0.5
    out = dict(new_info)
    for key in QUIET_FROZEN_INFO_FIELDS:
        out[key] = jp.where(quiet, prev_info[key], new_info[key])
    return out


def SchaftUnifiedJoystickV1__assemble_reward(
    self, components: Dict[str, jax.Array], gate: jax.Array
):
    """§12.1 group-weighted assembly over the 58 frozen terms."""
    scaled: Dict[str, jax.Array] = {}
    totals = {"ALWAYS": jp.zeros(()), "MOTION": jp.zeros(()), "QUIET": jp.zeros(())}
    for name in ucfg.ALL_REWARD_TERMS:
        value = components.get(name)
        if value is None:
            raise KeyError(
                f"reward term {name!r} is missing from the component dict; the 58-term contract (§12.7) is not satisfied"
            )
        s = value * self._reward_scales[name]
        scaled[name] = s
        totals[ucfg.TERM_GROUP[name]] = totals[ucfg.TERM_GROUP[name]] + s
    raw = totals["ALWAYS"] + gate * totals["MOTION"] + (1.0 - gate) * totals["QUIET"]
    reward = jp.clip(raw * self.dt, -ucfg.REWARD_CLIP, ucfg.REWARD_CLIP)
    return (reward, totals, scaled)


def SchaftUnifiedJoystickV1__reset_bout_local(
    self, info: Dict[str, Any], rising_edge: jax.Array
) -> Dict[str, Any]:
    """§9.2 bout-local bilateral reward memory reset.

    Because ``v24_pair_valid`` and ``v24_abs_imbalance_previous`` become zero,
    the first completed bilateral pair of the new bout has no valid previous
    pair and therefore contributes exactly zero ``imbalance_growth``.
    """
    out = dict(info)
    for key in BOUT_LOCAL_BILATERAL_FIELDS:
        out[key] = jp.where(rising_edge, jp.zeros(()), info[key])
    return out


def SchaftUnifiedJoystickV1__u6v10_accounting_metric_keys(self) -> Tuple[str, ...]:
    """Exactly the keys :meth:`_u6v10_accounting_metrics` produces."""
    keys = []
    for name in self._u6v10_active_terms:
        keys.append(f"reward_raw/{name}")
        keys.append(f"reward_weighted_pre_dt/{name}")
        keys.append(f"reward_actual/{name}")
    direct = [
        "heading_progress_reward",
        "terminal_fall_cost",
        "u4_wz_under_response_penalty",
        "lateral_vx_leakage_penalty",
        "u13_track_planar",
        "u13_track_yaw",
        "u6_under_response_penalty",
        "u6_expected_liftoff_commit",
        "u6_expected_liftoff_stale_refund",
        "u6_side_bootstrap_bridge",
        "u6_post_liftoff_touchdown_bridge",
        "u6_alternating_touchdown_bridge",
        "u6_valid_command_step_completion",
        "u6_expected_unload_progress",
        "u6_expected_foot_still_loaded",
    ]
    if self._u6v10_unified_opposite_touchdown:
        direct.append("u6_opposite_touchdown_bridge")
    if self._v11_minimal_gait_core:
        direct.extend(
            [
                "v11_motion_double_support_duration_penalty",
                "v11_load_transfer_progress",
                "v11_swing_foot_command_progress",
                "v11_swing_clearance_progress",
                "v11_touchdown_load_acceptance_progress",
                "v11_support_exchange_progress",
                "v12_coupled_role_transfer_progress",
            ]
        )
    keys.extend((f"reward_actual/{name}" for name in direct))
    keys.extend(
        [
            "reward_actual/_frozen_preclip_total",
            "reward_actual/_frozen_clipped_total",
            "reward_actual/_frozen_clip_adjustment",
            "reward_actual/_group_always",
            "reward_actual/_group_motion",
            "reward_actual/_group_quiet",
            "reward_actual/_direct_total",
            "reward_actual/_terminal_total",
            "reward_actual/_step_reward",
            "reward_actual/_identity_residual",
        ]
    )
    return tuple(keys)


def SchaftUnifiedJoystickV1__u6v10_group_gate(
    self, group: str, gate: jax.Array
) -> jax.Array:
    if group == "ALWAYS":
        return jp.ones_like(gate)
    if group == "MOTION":
        return gate
    return 1.0 - gate


def SchaftUnifiedJoystickV1__v11_metric_keys(self) -> Tuple[str, ...]:
    """Exactly the ``v11/*`` keys :meth:`step` produces (§20)."""
    terms = (
        "motion_double_support_duration_penalty",
        "load_transfer_progress",
        "swing_foot_command_progress",
        "swing_clearance_progress",
        "touchdown_load_acceptance_progress",
        "support_exchange_progress",
    )
    keys = [
        "v11/shared_mean_head_enabled",
        "v11/phase_routing_enabled",
        "v11/phase_balanced_ppo_enabled",
        "v11/phase_dense_reward_gate_enabled",
        "v11/locomotion_request_fraction",
        "v11/u6_movement_active_fraction",
        "v11/gate_disagreement_fraction",
        "v11/motion_double_support_ticks",
        "v11/motion_double_support_ticks_max",
        "v11/motion_double_support_over_grace_fraction",
        "v11/swing_candidate_left_fraction",
        "v11/swing_candidate_right_fraction",
        "v11/touchdown_candidate_count",
        "v11/lam_left",
        "v11/lam_right",
        "v11/phase2_enter_count",
        "v11/phase2_exit_count",
        "v11/phase2_same_tick_enter_exit_count",
        "v11/phase2_dwell",
        "v11/phase2_dwell_max",
        "reward_raw/v12_coupled_role_transfer_progress",
        "reward_weighted/v12_coupled_role_transfer_progress",
        "v12/coupled_role_transfer_active_fraction",
    ]
    keys.extend((f"v11/raw_{name}" for name in terms))
    keys.extend((f"v11/weighted_{name}" for name in terms))
    return tuple(keys)


def SchaftUnifiedJoystickV1__v11_potential_delta(
    self, phi_now, prev_value, prev_valid, active, context_same
):
    """ONE bounded potential difference with safe rebasing.

    Returns ``(delta, next_value, next_valid)``.

    ``delta`` is EXACTLY 0.0 whenever the stored baseline is not valid (the
    first tick of a new context), the context changed, or the term is
    inactive -- so a rebase can never emit a reward step -- and is clipped
    symmetrically so a contact discontinuity cannot emit a spike. The baseline
    always re-anchors to ``phi_now``, and validity is carried forward only
    while the term is active, which is what makes QUIET / ``done`` /
    selector-change all rebase safely through the SAME mechanism.
    """
    usable = prev_valid * context_same * active
    delta = jp.where(usable > 0.5, phi_now - prev_value, jp.zeros(()))
    delta = jp.clip(delta, -self._v11_potential_clip, self._v11_potential_clip)
    return (delta, phi_now, active)


def SchaftUnifiedJoystickV1__write_back(
    self,
    info,
    step_info,
    components,
    foot_motion,
    data,
    metrics,
    ctrl,
    clipped_action,
    motor_targets,
    step_count,
    target_clipped_steps,
    min_left,
    min_right,
    max_roll_l,
    max_roll_r,
    max_h_l,
    max_h_r,
    terms,
) -> Dict[str, Any]:
    """Persist V24 state exactly as V24 does (no formula re-implemented)."""
    new_info = dict(step_info)
    new_info["last_last_act"] = info["last_act"]
    new_info["last_act"] = clipped_action
    new_info["motor_targets"] = motor_targets
    new_info["step"] = step_count + 1
    new_info["prev_left_foot_xy"] = foot_motion["left_pos"][:2]
    new_info["prev_right_foot_xy"] = foot_motion["right_pos"][:2]
    new_info["feet_air_time"] = foot_motion["feet_air_time"]
    new_info["target_air_time"] = terms["target_air_time_next"]
    new_info["max_foot_height_l"] = max_h_l
    new_info["max_foot_height_r"] = max_h_r
    new_info["target_clipped_steps"] = target_clipped_steps
    new_info["min_swing_contact_count_left"] = min_left
    new_info["min_swing_contact_count_right"] = min_right
    new_info["max_abs_roll_left"] = max_roll_l
    new_info["max_abs_roll_right"] = max_roll_r
    left_pos, right_pos = self._foot_positions(data)
    left_contact, right_contact = self._foot_contacts(data)
    cur_l = jp.where(
        left_contact, 0.0, info["current_consecutive_left_no_contact"] + 1.0
    )
    cur_r = jp.where(
        right_contact, 0.0, info["current_consecutive_right_no_contact"] + 1.0
    )
    td_l = jp.float32((info["prev_left_contact"] < 0.5) & left_contact)
    td_r = jp.float32((info["prev_right_contact"] < 0.5) & right_contact)
    new_info["prev_root_x"] = self.get_root_pos(data)[0]
    new_info["prev_left_foot_x"] = left_pos[0]
    new_info["prev_right_foot_x"] = right_pos[0]
    new_info["prev_left_foot_z"] = left_pos[2]
    new_info["prev_right_foot_z"] = right_pos[2]
    new_info["prev_left_contact"] = jp.float32(left_contact)
    new_info["prev_right_contact"] = jp.float32(right_contact)
    new_info["current_consecutive_left_no_contact"] = cur_l
    new_info["current_consecutive_right_no_contact"] = cur_r
    new_info["max_consecutive_left_no_contact"] = jp.maximum(
        info["max_consecutive_left_no_contact"], cur_l
    )
    new_info["max_consecutive_right_no_contact"] = jp.maximum(
        info["max_consecutive_right_no_contact"], cur_r
    )
    new_info["left_touchdown_count"] = info["left_touchdown_count"] + td_l
    new_info["right_touchdown_count"] = info["right_touchdown_count"] + td_r
    td_left_m = metrics["touchdown/left_count"]
    td_right_m = metrics["touchdown/right_count"]
    td_event = jp.maximum(td_left_m, td_right_m)
    td_side = jp.where(
        td_event > 0.5, td_right_m - td_left_m, info["last_true_touchdown_side"]
    )
    new_info["last_true_touchdown_side"] = td_side
    new_info["landed_touchdown_side"] = td_side
    for side in ("left", "right"):
        new_info[f"v8_consecutive_no_contact_steps_{side}"] = components[
            f"next_no_contact_streak_{side}"
        ]
        new_info[f"v8_max_height_gain_during_air_{side}"] = components[
            f"next_max_air_height_gain_{side}"
        ]
        new_info[f"v8_air_qualified_{side}"] = components[f"next_air_qualified_{side}"]
        new_info[f"v8_consecutive_contact_steps_{side}"] = components[
            f"next_contact_streak_{side}"
        ]
        new_info[f"v8_steps_since_last_qualified_touchdown_{side}"] = components[
            f"next_steps_since_qualified_touchdown_{side}"
        ]
    for key, src in (
        ("v8_last_qualified_touchdown_side", "next_last_qualified_touchdown_side"),
        ("v8_recent_eligible_touchdown_steps", "next_recent_eligible_touchdown_steps"),
        ("v8_support_transfer_window_steps", "next_support_transfer_window_steps"),
        ("v8_support_transfer_landed_side", "next_support_transfer_landed_side"),
        (
            "v8_support_transfer_previous_support_side",
            "next_support_transfer_previous_support_side",
        ),
        ("v8_support_transfer_completed", "next_support_transfer_completed"),
        ("v8_support_transfer_timeout_fired", "next_support_transfer_timeout_fired"),
        ("v9_expected_side", "next_v9_expected_side"),
        ("v9_transfer_pending", "next_v9_transfer_pending"),
        ("v9_pending_landed_side", "next_v9_pending_landed_side"),
        (
            "v9_previous_qualified_touchdown_side",
            "next_v9_previous_qualified_touchdown_side",
        ),
        (
            "v9_current_same_side_qualified_touchdown_streak",
            "next_v9_current_same_side_qualified_touchdown_streak",
        ),
        (
            "v9_max_same_side_qualified_touchdown_streak",
            "next_v9_max_same_side_qualified_touchdown_streak",
        ),
        ("v9_steps_since_expected_side_set", "next_v9_steps_since_expected_side_set"),
        ("v16_attempt_active", "next_v16_attempt_active"),
        ("v16_attempt_side", "next_v16_attempt_side"),
        ("v16_single_support_steps", "next_v16_single_support_steps"),
        ("v16_touchdown_latched", "next_v16_touchdown_latched"),
        ("v16_forward_placement", "next_v16_forward_placement"),
        ("v16_forward_placement_quality", "next_v16_forward_placement_quality"),
        ("v24_last_left_placement", "next_v24_last_left_placement"),
        ("v24_last_right_placement", "next_v24_last_right_placement"),
        ("v24_left_valid", "next_v24_left_valid"),
        ("v24_right_valid", "next_v24_right_valid"),
        ("v24_pair_valid", "next_v24_pair_valid"),
        ("v24_abs_imbalance_previous", "next_v24_abs_imbalance_previous"),
    ):
        if src in components:
            new_info[key] = components[src]
    return new_info
