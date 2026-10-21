# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0. See LICENSE.
# Adapted from the private research working tree based on MuJoCo Playground.
# Robot descriptions and model parameters are intentionally excluded.

from __future__ import annotations
from typing import Dict, Tuple
import jax
import jax.numpy as jp

QUIET_FROZEN_INFO_FIELDS: Tuple[str, ...] = (
    "v9_expected_side",
    "v9_transfer_pending",
    "v9_steps_since_expected_side_set",
    "v9_pending_landed_side",
    "v9_previous_qualified_touchdown_side",
    "v9_current_same_side_qualified_touchdown_streak",
    "v8_consecutive_no_contact_steps_left",
    "v8_consecutive_no_contact_steps_right",
    "v8_max_height_gain_during_air_left",
    "v8_max_height_gain_during_air_right",
    "v8_air_qualified_left",
    "v8_air_qualified_right",
    "v8_consecutive_contact_steps_left",
    "v8_consecutive_contact_steps_right",
    "v8_steps_since_last_qualified_touchdown_left",
    "v8_steps_since_last_qualified_touchdown_right",
    "v8_last_qualified_touchdown_side",
    "v8_recent_eligible_touchdown_steps",
    "v8_support_transfer_window_steps",
    "v8_support_transfer_landed_side",
    "v8_support_transfer_previous_support_side",
    "v8_support_transfer_completed",
    "v8_support_transfer_timeout_fired",
    "v16_attempt_active",
    "v16_attempt_side",
    "v16_single_support_steps",
    "v16_touchdown_latched",
    "v16_forward_placement",
    "v16_forward_placement_quality",
)
BOUT_LOCAL_BILATERAL_FIELDS: Tuple[str, ...] = (
    "v24_last_left_placement",
    "v24_last_right_placement",
    "v24_left_valid",
    "v24_right_valid",
    "v24_pair_valid",
    "v24_abs_imbalance_previous",
)


def u6_event_algebra(
    *,
    lc_f: jax.Array,
    rc_f: jax.Array,
    prev_lc: jax.Array,
    prev_rc: jax.Array,
    air_l_prev: jax.Array,
    air_r_prev: jax.Array,
    prev_side: jax.Array,
    liftoff_armed: jax.Array,
    commit_outstanding: jax.Array,
    alternation_latched: jax.Array,
    stability_gate: jax.Array,
    slip_ok: jax.Array,
    movement_active: jax.Array,
    completion_candidate: jax.Array,
    min_air_ticks: jax.Array,
    liftoff_min_air_ticks: jax.Array,
    liftoff_stale_ticks: jax.Array,
    side_aware: bool,
    require_alternation: bool,
    expected_side_init: jax.Array | None = None,
    bootstrap_used: jax.Array | None = None,
    first_liftoff_bootstrap: bool = False,
) -> Dict[str, jax.Array]:
    """The whole U6 touchdown/alternation/completion event algebra, pure.

    Extracted verbatim from ``_unified_components`` so the synthetic event and
    return tests can drive the EXACT production expressions with hand-built
    contact sequences, in the exact runtime read/write order, without a mjx
    ``Data``. ``_unified_components`` calls this and merges the result; there is
    no second copy of the algebra anywhere.

    Every input is the value read from ``info``/``components`` BEFORE this
    tick's write-back (``prev_*``, ``air_*_prev``, ``prev_side``,
    ``liftoff_armed``, ``alternation_latched``), matching ``step()``'s ordering:
    components are computed from the previous tick's state, then ``_write_back``
    advances it.

    ``side_aware`` / ``require_alternation`` are STATIC python bools (stage
    config, never traced), so a pre-V5 stage traces exactly the V3/V4 graph.

    Returns:
      The ``u6_*`` component dict. ``u6_next_*`` successor state is produced by
      :func:`u6_event_state_update`, which consumes this dict.
    """
    ev: Dict[str, jax.Array] = {}
    td_left = (1.0 - prev_lc) * lc_f
    td_right = (1.0 - prev_rc) * rc_f
    genuine_l = (air_l_prev >= min_air_ticks).astype(jp.float32)
    genuine_r = (air_r_prev >= min_air_ticks).astype(jp.float32)
    land_left = td_left * genuine_l
    land_right = td_right * genuine_r
    ev["u6_movement_active"] = movement_active
    bridge_gates = stability_gate * slip_ok * movement_active
    raw_td = jp.clip(td_left + td_right, 0.0, 1.0)
    air_ok = jp.clip(land_left + land_right, 0.0, 1.0)
    ev["u6_post_liftoff_touchdown_bridge"] = air_ok * bridge_gates
    ev["u6_raw_touchdown"] = raw_td
    ev["u6_raw_touchdown_left"] = td_left
    ev["u6_raw_touchdown_right"] = td_right
    ev["u6_td_rej_min_air"] = raw_td * (1.0 - air_ok)
    ev["u6_td_rej_stability"] = air_ok * (1.0 - stability_gate)
    ev["u6_td_rej_slip"] = air_ok * (1.0 - slip_ok)
    ev["u6_td_rej_quiet"] = air_ok * (1.0 - movement_active)
    both_landed = land_left * land_right
    touchdown_side = jp.where(
        both_landed > 0.5, 0.0, -1.0 * land_left + 1.0 * land_right
    )
    ev["u6_touchdown_side"] = touchdown_side
    has_prev = (jp.abs(prev_side) > 0.5).astype(jp.float32)
    has_side = (jp.abs(touchdown_side) > 0.5).astype(jp.float32)
    alternates = (touchdown_side * prev_side < 0.0).astype(jp.float32)
    ev["u6_alternating_touchdown_bridge"] = (
        has_prev * has_side * alternates * bridge_gates
    )
    credited = air_ok * bridge_gates
    ev["u6_side_bootstrap"] = credited * has_side * (1.0 - has_prev)
    ev["u6_same_side_rejected"] = credited * has_side * has_prev * (1.0 - alternates)
    ev["u6_simultaneous_rejected"] = credited * (1.0 - has_side) * both_landed
    if first_liftoff_bootstrap:
        _side_init = (
            jp.zeros_like(prev_side)
            if expected_side_init is None
            else expected_side_init
        )
        _boot_used = (
            jp.zeros_like(prev_side) if bootstrap_used is None else bootstrap_used
        )
        expected_side = jp.where(has_prev > 0.5, -prev_side, _side_init)
    else:
        expected_side = -prev_side
    has_expected_side = (jp.abs(expected_side) > 0.5).astype(jp.float32)
    expected_air_prev = jp.where(
        expected_side < -0.5,
        air_l_prev,
        jp.where(expected_side > 0.5, air_r_prev, jp.zeros(())),
    )
    expected_swing_committed = (expected_air_prev >= liftoff_min_air_ticks).astype(
        jp.float32
    )
    if first_liftoff_bootstrap:
        first_boot = (
            (1.0 - has_prev)
            * (1.0 - _boot_used)
            * has_expected_side
            * stability_gate
            * movement_active
        )
        armed_effective = jp.clip(liftoff_armed + first_boot, 0.0, 1.0)
        commit_side_gate = has_expected_side
    else:
        first_boot = jp.zeros_like(has_prev)
        armed_effective = liftoff_armed
        commit_side_gate = has_prev
    ev["u6_first_liftoff_bootstrap"] = first_boot
    ev["u6_liftoff_armed_effective"] = armed_effective
    expected_liftoff_commit = (
        armed_effective
        * commit_side_gate
        * expected_swing_committed
        * stability_gate
        * movement_active
    )
    ev["u6_expected_liftoff_commit"] = expected_liftoff_commit
    ev["u6_side_bootstrap_bridge"] = ev["u6_side_bootstrap"]
    ev["u6_expected_air_ticks_1_4"] = (
        commit_side_gate
        * movement_active
        * jp.logical_and(
            expected_air_prev >= 1.0, expected_air_prev < liftoff_min_air_ticks
        ).astype(jp.float32)
    )
    ev["u6_expected_air_ticks_ge_5"] = (
        commit_side_gate
        * movement_active
        * (expected_air_prev >= liftoff_min_air_ticks).astype(jp.float32)
    )
    ev["u6_expected_liftoff_stale_refund"] = commit_outstanding * (
        expected_air_prev >= liftoff_stale_ticks
    ).astype(jp.float32)
    if side_aware:
        ev["u6_post_liftoff_touchdown_bridge"] = ev["u6_alternating_touchdown_bridge"]
    credit_event = jp.clip(
        ev["u6_side_bootstrap"] + ev["u6_alternating_touchdown_bridge"], 0.0, 1.0
    )
    ev["u6_credit_event"] = credit_event
    alternation_latch = jp.clip(
        alternation_latched + ev["u6_alternating_touchdown_bridge"], 0.0, 1.0
    )
    ev["u6_alternation_latch"] = alternation_latch
    if require_alternation:
        ev["u6_valid_command_step_completion"] = (
            completion_candidate * alternation_latch
        )
    else:
        ev["u6_valid_command_step_completion"] = completion_candidate
    is_left_side = (prev_side < -0.5).astype(jp.float32)
    is_right_side = (prev_side > 0.5).astype(jp.float32)
    ev["u6_expected_side_left"] = is_right_side
    ev["u6_expected_side_right"] = is_left_side
    ev["u6_same_side_ll"] = ev["u6_same_side_rejected"] * is_left_side
    ev["u6_same_side_rr"] = ev["u6_same_side_rejected"] * is_right_side
    ev["u6_opposite_lr"] = ev["u6_alternating_touchdown_bridge"] * is_left_side
    ev["u6_opposite_rl"] = ev["u6_alternating_touchdown_bridge"] * is_right_side
    ev["u6_liftoff_rej_already_credited"] = (
        (1.0 - liftoff_armed)
        * has_prev
        * expected_swing_committed
        * stability_gate
        * movement_active
    )
    raw_td_expected = jp.clip(
        td_right * is_left_side + td_left * is_right_side, 0.0, 1.0
    )
    air_ok_expected = jp.clip(
        land_right * is_left_side + land_left * is_right_side, 0.0, 1.0
    )
    ev["u6_expected_foot_landed"] = raw_td_expected
    ev["u6_opp_td_rej_min_air"] = raw_td_expected * (1.0 - air_ok)
    ev["u6_opp_td_rej_stability"] = air_ok_expected * (1.0 - stability_gate)
    ev["u6_opp_td_rej_slip"] = air_ok_expected * (1.0 - slip_ok)
    ev["u6_opp_td_rej_quiet"] = air_ok_expected * (1.0 - movement_active)
    ev["u6_opp_td_rej_side_memory"] = ev["u6_same_side_rejected"]
    ev["u6_completion_rej_no_alternation"] = completion_candidate * (
        1.0 - alternation_latch
    )
    return ev


def u6_decode_gait_phase(
    movement_active: jax.Array,
    alternation_latched: jax.Array,
    commit_outstanding: jax.Array,
    *,
    corrected: bool,
) -> jax.Array:
    """THE single authoritative U6 discrete gait-phase decoder.

    Every consumer of the discrete phase -- the actor observation channel
    (index 62), the V8/V9 phase-head selector, the PPO phase-balanced loss
    metadata, the TensorBoard phase labels, the positive track/progress phase
    gate and the phase-1 unload / still-loaded shaping gates -- reads its phase
    from THIS function, so a corrected stage cannot end up with two different
    taxonomies in the same run.

    ``corrected=False`` (V6/V7/V8 and every historical stage) reproduces the
    shipped priority bit-for-bit:

        QUIET -> 0; else latched -> 3; else outstanding -> 2; else 1

    That priority ALIASES the reachable state ``alternation_latched == 1 and
    commit_outstanding == 1`` into phase 3, even though an unresolved committed
    swing is semantically WAIT_TOUCHDOWN. Because the alternation latch is
    sticky within a movement bout (cleared only by a valid completion, exact
    QUIET or `done`) and valid completion measured EXACTLY zero across the whole
    V8 run, every post-alternation wait-liftoff and wait-touchdown tick of a bout
    was reported as phase 3 -- which is why V8 logged a phase-2 fraction of
    exactly zero on all 200 PPO updates while phase 3 absorbed those samples.

    ``corrected=True`` (V9 onward) restores the intended contract by giving the
    active unresolved swing priority over the latch:

        QUIET -> 0; else outstanding -> 2; else latched -> 3; else 1

    No event predicate, FSM threshold, reward coefficient, gain or action scale
    is touched by either branch -- this is purely how the existing latch/commit
    state pair is projected onto the four phase ids.

    Args:
      movement_active: 1.0 when any command axis is active, else 0.0.
      alternation_latched: the sticky canonical-alternation latch.
      commit_outstanding: 1.0 while a paid swing commit is unresolved.
      corrected: STATIC python bool selecting the priority order.

    Returns:
      A float phase id in {0.0, 1.0, 2.0, 3.0}.
    """
    if corrected:
        inner = jp.where(
            commit_outstanding > 0.5,
            U6_PHASE_WAIT_TOUCHDOWN,
            jp.where(
                alternation_latched > 0.5,
                U6_PHASE_WAIT_COMPLETION,
                U6_PHASE_WAIT_LIFTOFF,
            ),
        )
    else:
        inner = jp.where(
            alternation_latched > 0.5,
            U6_PHASE_WAIT_COMPLETION,
            jp.where(
                commit_outstanding > 0.5, U6_PHASE_WAIT_TOUCHDOWN, U6_PHASE_WAIT_LIFTOFF
            ),
        )
    return jp.where(movement_active < 0.5, U6_PHASE_INACTIVE, inner)


def u6_event_state_update(
    ev: Dict[str, jax.Array],
    *,
    prev_side: jax.Array,
    liftoff_armed: jax.Array,
    commit_outstanding: jax.Array,
    done: jax.Array,
    side_aware: bool,
    bootstrap_used_prev: jax.Array = 0.0,
) -> Dict[str, jax.Array]:
    """The V5 successor state for the four U6 event-chain info fields.

    Called by ``_write_back``; also driven directly by the synthetic tests so
    the multi-tick sequences they build use the production transition, not a
    re-implementation of it.
    """
    if not side_aware:
        td_side = ev["u6_touchdown_side"]
        return {
            "u6_last_credited_side": jp.where(
                jp.abs(td_side) > 0.5, td_side, prev_side
            ),
            "u6_liftoff_armed": jp.zeros_like(liftoff_armed),
            "u6_commit_outstanding": jp.zeros_like(liftoff_armed),
            "u6_alternation_latched": jp.zeros_like(liftoff_armed),
            "u6_bout_bootstrap_used": jp.zeros_like(liftoff_armed),
        }
    credit = ev["u6_credit_event"] * (1.0 - done)
    return {
        "u6_last_credited_side": jp.where(
            credit > 0.5, ev["u6_touchdown_side"], prev_side
        ),
        "u6_liftoff_armed": jp.clip(
            ev.get("u6_liftoff_armed_effective", liftoff_armed)
            * (1.0 - ev["u6_expected_liftoff_commit"])
            + credit,
            0.0,
            1.0,
        ),
        "u6_bout_bootstrap_used": jp.clip(
            bootstrap_used_prev + ev.get("u6_first_liftoff_bootstrap", jp.zeros(())),
            0.0,
            1.0,
        )
        * ev["u6_movement_active"]
        * (1.0 - done),
        "u6_commit_outstanding": jp.clip(
            (commit_outstanding + ev["u6_expected_liftoff_commit"])
            * (1.0 - ev["u6_expected_liftoff_stale_refund"])
            * (1.0 - jp.clip(ev["u6_expected_foot_landed"], 0.0, 1.0))
            * (1.0 - credit)
            * (1.0 - done),
            0.0,
            1.0,
        ),
        "u6_alternation_latched": ev["u6_alternation_latch"]
        * (1.0 - jp.clip(ev["u6_valid_command_step_completion"], 0.0, 1.0))
        * ev["u6_movement_active"]
        * (1.0 - done),
    }


ACTIVE_YAW_OTHER_POSITIVE_TERMS: Tuple[str, ...] = (
    "effective_liftoff",
    "swing_no_contact",
    "air_time",
    "airborne_clearance",
    "pre_liftoff_clearance",
    "deep_unload",
    "touchdown_load_acceptance",
    "post_touchdown_stability",
    "support_transfer_progress",
    "support_transfer_completion",
    "stable_single_support",
    "actual_single_support",
)
ACTIVE_YAW_EQUALITY_TERMS: Tuple[str, ...] = (
    "bilateral_symmetry",
    "completed_placement_band",
    "imbalance_growth",
)
U4_UNIFIED_METRIC_KEYS: Tuple[str, ...] = (
    "yaw_pos/command_wz",
    "yaw_pos/measured_wz",
    "yaw_pos/signed_response_ratio",
    "yaw_pos/tracking_error",
    "yaw_pos/wrong_sign_fraction",
    "yaw_pos/valid_count",
    "yaw_neg/command_wz",
    "yaw_neg/measured_wz",
    "yaw_neg/signed_response_ratio",
    "yaw_neg/tracking_error",
    "yaw_neg/wrong_sign_fraction",
    "yaw_neg/valid_count",
    "yaw_mag_008/measured_abs_wz",
    "yaw_mag_008/tracking_error",
    "yaw_mag_008/valid_count",
    "yaw_mag_014/measured_abs_wz",
    "yaw_mag_014/tracking_error",
    "yaw_mag_014/valid_count",
    "yaw_mag_020/measured_abs_wz",
    "yaw_mag_020/tracking_error",
    "yaw_mag_020/valid_count",
    "yaw/under_response_fraction",
    "yaw/mean_forward_velocity",
    "yaw/cross_track_abs",
    "yaw/heading_error",
    "yaw/double_support_fraction",
    "yaw/alternating_touchdown_rate",
    "yaw/fall_fraction",
    "yaw/support_timeout_fraction",
    "yaw/same_side_repeat_fraction",
    "yaw/valid_count",
    "post_turn/mean_abs_wz",
    "post_turn/settling_latency",
    "post_turn/settle_event_count",
    "post_turn/settled_fraction",
    "post_turn/mean_forward_velocity",
    "post_turn/valid_count",
    "curvature/command_radius",
    "curvature/measured_curvature",
    "curvature/absolute_error",
    "curvature/valid_count",
)
U4_STRICT_YAW_DC_METRIC_KEYS: Tuple[str, ...] = (
    "yaw_dc_pos/mean_command_wz",
    "yaw_dc_pos/mean_wz_dc",
    "yaw_dc_pos/mean_raw_wz",
    "yaw_dc_pos/signed_response_ratio",
    "yaw_dc_pos/heading_response_ratio",
    "yaw_dc_pos/wrong_sign_fraction",
    "yaw_dc_pos/valid_count",
    "yaw_dc_neg/mean_command_wz",
    "yaw_dc_neg/mean_wz_dc",
    "yaw_dc_neg/mean_raw_wz",
    "yaw_dc_neg/signed_response_ratio",
    "yaw_dc_neg/heading_response_ratio",
    "yaw_dc_neg/wrong_sign_fraction",
    "yaw_dc_neg/valid_count",
    "yaw_dc/mean_abs_wz_ac",
    "yaw_dc/under_response_fraction",
    "yaw_dc/under_response_reward_mean",
    "yaw_dc/under_response_reward_sum",
    "yaw_dc/heading_progress_reward_mean",
    "yaw_dc/mean_forward_velocity",
    "yaw_dc/double_support_fraction",
    "yaw_dc/cross_track_abs",
    "yaw_dc/fall_fraction",
    "yaw_dc/support_timeout_fraction",
    "yaw_dc/same_side_repeat_fraction",
    "yaw_dc/positive_reward_scale_mean",
    "yaw_dc/safety_gate_mean",
    "yaw_dc/valid_count",
    "post_turn_dc/mean_abs_wz_dc",
    "post_turn_dc/settled_fraction",
    "post_turn_dc/settling_latency",
    "post_turn_dc/valid_count",
)
U6_PHASE_INACTIVE = 0.0
U6_PHASE_WAIT_LIFTOFF = 1.0
U6_PHASE_WAIT_TOUCHDOWN = 2.0
U6_PHASE_WAIT_COMPLETION = 3.0
