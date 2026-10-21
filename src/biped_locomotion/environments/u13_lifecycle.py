# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0. See LICENSE.
# Adapted from the private research working tree based on MuJoCo Playground.
# Robot descriptions and model parameters are intentionally excluded.

from __future__ import annotations
from typing import Dict
import jax
import jax.numpy as jp
from biped_locomotion.environments import mjx_support as mjx_env
from biped_locomotion.commands import process as ucp
from biped_locomotion.gait.unified_events import u6_event_state_update
from biped_locomotion.environments import disabled_teacher as sst


def SchaftUnifiedJoystickV1__reset(self, rng: jax.Array) -> mjx_env.State:
    """V24 reset plus the additive unified command/gate/path fields."""
    rng, cmd_rng, u4_rng, strict_yaw_rng = jax.random.split(rng, 4)
    state = self._invoke_source("ResidualForwardWalkingV24.reset", rng)
    info = dict(state.info)
    yaw = self._forward_yaw_from_quat(self.get_orientation(state.data))
    root_xy = self.get_root_pos(state.data)[:2]
    cmd_state = ucp.init_command_state(
        cmd_rng,
        self._sched,
        compat=self._compat,
        canonical_start_side=self._canonical_side,
        reset_yaw=yaw,
        root_xy=root_xy,
    )
    info.update(ucp.to_info(cmd_state))
    info["command"] = cmd_state.command
    info["prev_root_xy"] = root_xy
    info["unified_forward_progress"] = jp.zeros(())
    info["unified_e_path_x"] = jp.zeros(())
    info["unified_e_path_y"] = jp.zeros(())
    info["eval_forced_command_active"] = jp.zeros(())
    info["eval_forced_command"] = jp.zeros(3)
    u4_mode_id, u4_mode_params = ucp.sample_u4_unified_episode(
        u4_rng, self._u4_mode_probs
    )
    info["u4_mode_id"] = u4_mode_id
    info["u4_mode_params"] = u4_mode_params
    info["u4_post_turn_ticks"] = jp.asarray(0, dtype=jp.int32)
    info["u4_post_turn_settled"] = jp.zeros(())
    strict_yaw_active, strict_yaw_sign = ucp.sample_strict_yaw_episode(
        strict_yaw_rng, self._strict_yaw_hold_probability
    )
    info["strict_yaw_active"] = strict_yaw_active
    info["strict_yaw_sign"] = strict_yaw_sign
    raw_wz0 = self.get_gyro(state.data)[2]
    info["wz_dc"] = raw_wz0
    info["yaw_unwrapped"] = jp.zeros(())
    info["prev_yaw_for_unwrap"] = yaw
    info["heading_segment_active"] = jp.zeros(())
    info["heading_segment_start_yaw_unwrapped"] = jp.zeros(())
    info["heading_segment_elapsed_ticks"] = jp.asarray(0, dtype=jp.int32)
    info["u6_air_ticks_left"] = jp.zeros(())
    info["u6_air_ticks_right"] = jp.zeros(())
    info["u6_last_credited_side"] = jp.zeros(())
    info["u6_liftoff_armed"] = jp.zeros(())
    info["u6_commit_outstanding"] = jp.zeros(())
    info["u6_alternation_latched"] = jp.zeros(())
    info["u6_bout_bootstrap_used"] = jp.zeros(())
    info["u6v6_expected_side_init"] = jp.zeros(())
    info["u6v6_load_phi_prev"] = jp.zeros(())
    info["u6v6_load_phi_prev_side"] = jp.zeros(())
    if self._u6v7_phase_balanced_ppo or self._v11_minimal_gait_core:
        info["u6v7_gait_phase"] = jp.zeros(())
    if self._v11_minimal_gait_core:
        info["v11_motion_double_support_ticks"] = jp.zeros(())
        info["v11_motion_double_support_ticks_max"] = jp.zeros(())
        info["v11_load_phi_prev"] = jp.zeros(())
        info["v11_load_phi_valid"] = jp.zeros(())
        info["v11_load_context"] = jp.zeros(())
        info["v11_swing_context"] = jp.zeros(())
        info["v11_swing_foot_phi_prev"] = jp.zeros(())
        info["v11_swing_foot_phi_valid"] = jp.zeros(())
        info["v11_clearance_phi_prev"] = jp.zeros(())
        info["v11_clearance_phi_valid"] = jp.zeros(())
        info["v11_recent_landed_side"] = jp.zeros(())
        info["v11_touchdown_phi_prev"] = jp.zeros(())
        info["v11_touchdown_phi_valid"] = jp.zeros(())
        info["v11_exchange_phi_prev"] = jp.zeros(())
        info["v11_exchange_phi_valid"] = jp.zeros(())
        info["v12_role_transfer_phi_prev"] = jp.zeros(())
        info["v12_role_transfer_phi_valid"] = jp.zeros(())
        info["v11_phase2_prev"] = jp.zeros(())
        info["v11_phase2_dwell"] = jp.zeros(())
        info["v11_phase2_dwell_max"] = jp.zeros(())
    if self._u6v9_recurrent_hidden_size > 0:
        info["u6v9_actor_hidden"] = jp.zeros(self._u6v9_recurrent_hidden_size)
    if self._canonical_side:
        info["v9_expected_side"] = jp.zeros(())
    info["yaw_teacher_gate_smooth"] = jp.zeros(sst.N_GATE_STATES)
    obs = self._get_obs(state.data, info)
    return state.replace(obs=obs, info=info)


def SchaftUnifiedJoystickV1__step(
    self, state: mjx_env.State, action: jax.Array
) -> mjx_env.State:
    info = state.info
    step_count = info["step"]
    prev_cmd = ucp.from_info(info)
    prev_gate = prev_cmd.locomotion_request
    rng, cmd_rng = jax.random.split(info["rng"])
    if self._enable_gait_phase_observation:
        rng, gait_side_rng = jax.random.split(rng)
    yaw_prev = self._forward_yaw_from_quat(self.get_orientation(state.data))
    u4_mode_id = info["u4_mode_id"]
    u4_mode_active = u4_mode_id != ucp.U4_MODE_NORMAL
    u4_scripted_command = ucp.u4_unified_command(
        u4_mode_id, info["u4_mode_params"], step_count
    )
    strict_yaw_active = info["strict_yaw_active"] > 0.5
    strict_yaw_command = ucp.strict_yaw_hold_command(
        info["strict_yaw_active"], info["strict_yaw_sign"], step_count
    )
    eval_active = info["eval_forced_command_active"] > 0.5
    force_active = jp.logical_or(
        jp.logical_or(eval_active, u4_mode_active), strict_yaw_active
    )
    forced_command = jp.where(
        strict_yaw_active,
        strict_yaw_command,
        jp.where(u4_mode_active, u4_scripted_command, info["eval_forced_command"]),
    )
    cmd_state = ucp.step_command_state(
        prev_cmd,
        cmd_rng,
        self._sched,
        compat=self._compat,
        yaw=yaw_prev,
        ctrl_dt=self.dt,
        force_active=force_active,
        forced_command=forced_command,
    )
    gate = cmd_state.locomotion_request
    rising_edge = jp.logical_and(prev_gate < 0.5, gate > 0.5)
    if self._canonical_side:
        begun = ucp.begin_locomotion_bout(cmd_state)
        cmd_state = jax.tree.map(
            lambda new, old: jp.where(rising_edge, new, old), begun, cmd_state
        )
    step_info = dict(info)
    step_info.update(ucp.to_info(cmd_state))
    step_info["command"] = cmd_state.command
    step_info["rng"] = rng
    if self._canonical_side:
        step_info = self._apply_canonical_start(step_info, cmd_state, rising_edge)
    legacy = self._v18_legacy_phase(step_count)
    phase = self._v18_event_phase(step_info, legacy)
    step_info = self._v18_apply_phase(step_info, phase)
    control_info = dict(step_info)
    control_info["step"] = phase["local_step"]
    q_ref, unload_weight, _ = self._q_ref_phase(
        phase["phase_id"],
        phase["target_left_swing"],
        phase["target_right_swing"],
        control_info,
    )
    left_contact_now, right_contact_now = self._foot_contacts(state.data)
    teacher_residual, new_gate_smooth = sst.teacher_residual(
        left_contact_now,
        right_contact_now,
        info["yaw_teacher_gate_smooth"],
        cmd_state.command[2],
        self._yaw_teacher_positive_scale,
        self._yaw_teacher_negative_scale,
        self._yaw_teacher_gate_alpha,
    )
    effective_blend = jp.where(
        self._enable_sign_specific_yaw_teacher, self._yaw_teacher_blend, jp.zeros(())
    )
    a_preclip = action + effective_blend * teacher_residual
    a_exec = jp.clip(a_preclip, -1.0, 1.0)
    ctrl = self._v3_control_law(
        a_exec,
        q_ref,
        phase["target_left_swing"],
        phase["target_right_swing"],
        phase["phase_id"],
        phase["local_step"],
    )
    clipped_action = ctrl["clipped_action"]
    motor_targets = ctrl["motor_targets"]
    data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
    terms = self._foot_unload_terms(data, step_info)
    foot_motion = self._foot_motion(data, step_info)
    done = self._get_termination(data)
    done_f = jp.float32(done)
    root_xy = self.get_root_pos(data)[:2]
    cmd_state = ucp.integrate_path_reference(cmd_state, self._compat, self.dt)
    e_path_x, e_path_y = ucp.path_errors(
        cmd_state.p_ref, root_xy, cmd_state.heading_target
    )
    fwd_progress = ucp.forward_progress(
        info["prev_root_xy"], root_xy, cmd_state.heading_target
    )
    raw_wz = self.get_gyro(data)[2]
    wz_dc = ucp.update_wz_dc(info["wz_dc"], raw_wz, self._wz_dc_alpha)
    wz_ac = raw_wz - wz_dc
    yaw_now = self._forward_yaw_from_quat(self.get_orientation(data))
    yaw_unwrapped = ucp.unwrap_heading_step(
        info["yaw_unwrapped"], info["prev_yaw_for_unwrap"], yaw_now
    )
    is_yaw_active = jp.abs(cmd_state.command[2]) > self._dense_delta[2]
    was_active = info["heading_segment_active"] > 0.5
    starting_now = jp.logical_and(is_yaw_active, jp.logical_not(was_active))
    segment_start_yaw_unwrapped = jp.where(
        starting_now, yaw_unwrapped, info["heading_segment_start_yaw_unwrapped"]
    )
    segment_elapsed_ticks = jp.where(
        is_yaw_active,
        jp.where(starting_now, 1, info["heading_segment_elapsed_ticks"] + 1),
        0,
    ).astype(jp.int32)
    segment_elapsed_time = segment_elapsed_ticks.astype(jp.float32) * self.dt
    heading_ratio = ucp.heading_response_ratio(
        cmd_state.command[2],
        yaw_unwrapped,
        segment_start_yaw_unwrapped,
        segment_elapsed_time,
    )
    heading_progress_valid = jp.logical_and(is_yaw_active, segment_elapsed_time >= 0.5)
    components = self._v9_components(
        data, step_info, phase["phase_id"], terms, foot_motion
    )
    _u13_raw_transfer_completion = components["support_transfer_completion"]
    _u13_prev_expected_side = step_info["v9_expected_side"]
    _u13_next_expected_side = components["next_v9_expected_side"]
    _u13_transfer_timeout = components["support_transfer_timeout"]
    unified = self._unified_components(
        data,
        step_info,
        cmd_state,
        components,
        e_path_x=e_path_x,
        e_path_y=e_path_y,
        forward_progress=fwd_progress,
        clipped_action=clipped_action,
        action=action,
        wz_dc=wz_dc,
        heading_response_ratio_value=heading_ratio,
        heading_progress_valid=heading_progress_valid,
    )
    components.update(unified)
    reward, group_totals, scaled = self._assemble_reward(components, gate)
    u6v10_actual: Dict[str, jax.Array] = {}
    u6v10_frozen_clipped = reward
    _c_heading = gate * components["heading_progress_reward"] * self.dt
    reward = reward + _c_heading
    u6v10_actual["heading_progress_reward"] = _c_heading
    left_swing_count = jp.float32(self._contact_count(data, self._left_foot_geom_id))
    right_swing_count = jp.float32(self._contact_count(data, self._right_foot_geom_id))
    swing_contact_count = (
        phase["target_left_swing"] * left_swing_count
        + phase["target_right_swing"] * right_swing_count
    )
    target_clipped_steps = info["target_clipped_steps"] + ctrl["target_clipped"]
    min_left = jp.where(
        phase["target_left_swing"] > 0.5,
        jp.minimum(info["min_swing_contact_count_left"], left_swing_count),
        info["min_swing_contact_count_left"],
    )
    min_right = jp.where(
        phase["target_right_swing"] > 0.5,
        jp.minimum(info["min_swing_contact_count_right"], right_swing_count),
        info["min_swing_contact_count_right"],
    )
    left_roll, _, right_roll, _ = self._foot_roll_pitch(data)
    max_roll_l = jp.maximum(info["max_abs_roll_left"], jp.abs(left_roll))
    max_roll_r = jp.maximum(info["max_abs_roll_right"], jp.abs(right_roll))
    left_h = foot_motion["left_pos"][2] - step_info["left_foot_baseline_z"]
    right_h = foot_motion["right_pos"][2] - step_info["right_foot_baseline_z"]
    max_h_l = jp.maximum(info["max_foot_height_l"], left_h)
    max_h_r = jp.maximum(info["max_foot_height_r"], right_h)
    metrics = self._update_metrics(
        state.metrics,
        data,
        clipped_action,
        action,
        motor_targets,
        scaled,
        step_info,
        phase["phase_id"],
        phase["gait_phase"],
        q_ref,
        terms,
        foot_motion,
        done_f,
        swing_contact_count,
        ctrl["swing_authority_gate"],
        ctrl["swing_residual_action_norm"],
        ctrl["requested_residual_offset_norm"],
        ctrl["effective_residual_offset_norm"],
        ctrl["target_clipped"],
        target_clipped_steps,
        min_left,
        min_right,
        max_h_l,
        max_h_r,
        max_roll_l,
        max_roll_r,
    )
    terminal_fall = metrics["termination/fall_event"]
    _c_terminal = terminal_fall * self._config.terminal_fall_cost
    reward = reward + _c_terminal
    metrics["termination/terminal_fall_cost"] = (
        terminal_fall * self._config.terminal_fall_cost
    )
    u6v10_actual["terminal_fall_cost"] = _c_terminal
    _c_u4_under = gate * components["u4_wz_under_response_penalty"] * self.dt
    reward = reward + _c_u4_under
    u6v10_actual["u4_wz_under_response_penalty"] = _c_u4_under
    _c_lat_leak = gate * components["lateral_vx_leakage_penalty"] * self.dt
    reward = reward + _c_lat_leak
    u6v10_actual["lateral_vx_leakage_penalty"] = _c_lat_leak
    _c_u13_planar = gate * components["u13_track_planar"] * self.dt
    reward = reward + _c_u13_planar
    u6v10_actual["u13_track_planar"] = _c_u13_planar
    _c_u13_yaw = gate * components["u13_track_yaw"] * self.dt
    reward = reward + _c_u13_yaw
    u6v10_actual["u13_track_yaw"] = _c_u13_yaw
    _c_u6_under = gate * components["u6_under_response_penalty"] * self.dt
    reward = reward + _c_u6_under
    u6v10_actual["u6_under_response_penalty"] = _c_u6_under
    bridge_live = gate * (1.0 - done_f)
    u6_bridge_a = (
        self._post_liftoff_touchdown_bridge_weight
        * components["u6_post_liftoff_touchdown_bridge"]
    )
    u6_bridge_b = (
        self._alternating_touchdown_bridge_weight
        * components["u6_alternating_touchdown_bridge"]
    )
    u6_completion = (
        self._valid_step_completion_weight
        * components["u6_valid_command_step_completion"]
    )
    u6_liftoff_commit = (
        self._expected_liftoff_bridge_weight * components["u6_expected_liftoff_commit"]
    )
    u6_bootstrap = (
        self._side_bootstrap_bridge_weight * components["u6_side_bootstrap_bridge"]
    )
    u6_liftoff_refund = (
        self._expected_liftoff_bridge_weight
        * components["u6_expected_liftoff_stale_refund"]
    )
    _c_commit = bridge_live * u6_liftoff_commit * self.dt
    reward = reward + _c_commit
    _c_refund = -(1.0 - done_f) * u6_liftoff_refund * self.dt
    reward = reward - (1.0 - done_f) * u6_liftoff_refund * self.dt
    _c_boot = bridge_live * u6_bootstrap * self.dt
    reward = reward + _c_boot
    _c_bridge_a = bridge_live * u6_bridge_a * self.dt
    reward = reward + _c_bridge_a
    _c_bridge_b = bridge_live * u6_bridge_b * self.dt
    reward = reward + _c_bridge_b
    if self._u6v10_unified_opposite_touchdown:
        u6_opposite_touchdown = (
            self._opposite_touchdown_bridge_weight
            * components["u6_alternating_touchdown_bridge"]
        )
        _c_opposite = bridge_live * u6_opposite_touchdown * self.dt
        reward = reward + _c_opposite
    _c_completion = gate * (1.0 - done_f) * u6_completion * self.dt
    reward = reward + gate * (1.0 - done_f) * u6_completion * self.dt
    u6_unload_progress = self._expected_unload_progress_weight * (
        components["u6_expected_unload_progress"]
        - components["u6_expected_reload_regress"]
    )
    u6_still_loaded = (
        self._expected_foot_still_loaded_weight
        * components["u6_expected_foot_still_loaded"]
    )
    _c_unload = bridge_live * u6_unload_progress * self.dt
    reward = reward + _c_unload
    _c_still_loaded = bridge_live * u6_still_loaded * self.dt
    reward = reward + _c_still_loaded
    if self._v11_minimal_gait_core:
        v11_live = gate * (1.0 - done_f)
        _c_v11_ds = -(
            v11_live
            * self._v11_motion_ds_penalty_weight
            * components["v11_motion_double_support_duration_penalty"]
            * self.dt
        )
        reward = reward + _c_v11_ds
        _c_v11_load = (
            v11_live
            * self._v11_load_transfer_weight
            * components["v11_load_transfer_progress"]
            * self.dt
        )
        reward = reward + _c_v11_load
        _c_v11_swing = (
            v11_live
            * self._v11_swing_foot_weight
            * components["v11_swing_foot_command_progress"]
            * self.dt
        )
        reward = reward + _c_v11_swing
        _c_v11_clear = (
            v11_live
            * self._v11_swing_clearance_weight
            * components["v11_swing_clearance_progress"]
            * self.dt
        )
        reward = reward + _c_v11_clear
        _c_v11_td = (
            v11_live
            * self._v11_touchdown_weight
            * components["v11_touchdown_load_acceptance_progress"]
            * self.dt
        )
        reward = reward + _c_v11_td
        _c_v11_ex = (
            v11_live
            * self._v11_support_exchange_weight
            * components["v11_support_exchange_progress"]
            * self.dt
        )
        reward = reward + _c_v11_ex
        _c_v12_role = (
            v11_live
            * self._v12_role_transfer_weight
            * components["v12_coupled_role_transfer_progress"]
            * self.dt
        )
        reward = reward + _c_v12_role
        u6v10_actual.update(
            {
                "v11_motion_double_support_duration_penalty": _c_v11_ds,
                "v11_load_transfer_progress": _c_v11_load,
                "v11_swing_foot_command_progress": _c_v11_swing,
                "v11_swing_clearance_progress": _c_v11_clear,
                "v11_touchdown_load_acceptance_progress": _c_v11_td,
                "v11_support_exchange_progress": _c_v11_ex,
                "v12_coupled_role_transfer_progress": _c_v12_role,
            }
        )
    u6v10_actual.update(
        {
            "u6_expected_liftoff_commit": _c_commit,
            "u6_expected_liftoff_stale_refund": _c_refund,
            "u6_side_bootstrap_bridge": _c_boot,
            "u6_post_liftoff_touchdown_bridge": _c_bridge_a,
            "u6_alternating_touchdown_bridge": _c_bridge_b,
            "u6_valid_command_step_completion": _c_completion,
            "u6_expected_unload_progress": _c_unload,
            "u6_expected_foot_still_loaded": _c_still_loaded,
        }
    )
    if self._u6v10_unified_opposite_touchdown:
        u6v10_actual["u6_opposite_touchdown_bridge"] = _c_opposite
    metrics.update(
        {
            "transition/post_liftoff_touchdown_count": bridge_live
            * components["u6_post_liftoff_touchdown_bridge"],
            "transition/alternating_touchdown_bridge_count": bridge_live
            * components["u6_alternating_touchdown_bridge"],
            "transition/u6_valid_command_step_completion_count": bridge_live
            * components["u6_valid_command_step_completion"],
            "reward/u6_post_liftoff_touchdown_bridge": bridge_live
            * u6_bridge_a
            * self.dt,
            "reward/u6_alternating_touchdown_bridge": bridge_live
            * u6_bridge_b
            * self.dt,
            "reward/u6_valid_command_step_completion": bridge_live
            * u6_completion
            * self.dt,
            "transition/u6_raw_touchdown_count": components["u6_raw_touchdown"],
            "transition/u6_raw_touchdown_left_count": components[
                "u6_raw_touchdown_left"
            ],
            "transition/u6_raw_touchdown_right_count": components[
                "u6_raw_touchdown_right"
            ],
            "transition/u6_td_rejected_min_air": components["u6_td_rej_min_air"],
            "transition/u6_td_rejected_stability": components["u6_td_rej_stability"],
            "transition/u6_td_rejected_slip": components["u6_td_rej_slip"],
            "transition/u6_td_rejected_quiet": components["u6_td_rej_quiet"],
            "transition/u6_side_bootstrap_count": bridge_live
            * components["u6_side_bootstrap"],
            "transition/u6_same_side_rejected_count": bridge_live
            * components["u6_same_side_rejected"],
            "transition/u6_simultaneous_rejected_count": bridge_live
            * components["u6_simultaneous_rejected"],
            "transition/u6_expected_liftoff_commit_count": bridge_live
            * components["u6_expected_liftoff_commit"],
            "transition/u6_side_bootstrap_bridge_count": bridge_live
            * components["u6_side_bootstrap_bridge"],
            "transition/u6_credit_event_count": bridge_live
            * components["u6_credit_event"],
            "reward/u6_expected_liftoff_commit": bridge_live
            * u6_liftoff_commit
            * self.dt,
            "transition/u6_expected_liftoff_stale_refund_count": (1.0 - done_f)
            * components["u6_expected_liftoff_stale_refund"],
            "reward/u6_expected_liftoff_stale_refund": -(1.0 - done_f)
            * u6_liftoff_refund
            * self.dt,
            "transition/u6_expected_foot_landed_count": components[
                "u6_expected_foot_landed"
            ],
            "reward/u6_side_bootstrap_bridge": bridge_live * u6_bootstrap * self.dt,
            "transition/u6_expected_side_left": components["u6_expected_side_left"],
            "transition/u6_expected_side_right": components["u6_expected_side_right"],
            "transition/u6_alternation_latch": components["u6_alternation_latch"],
            "transition/u6_same_side_ll_count": bridge_live
            * components["u6_same_side_ll"],
            "transition/u6_same_side_rr_count": bridge_live
            * components["u6_same_side_rr"],
            "transition/u6_opposite_lr_count": bridge_live
            * components["u6_opposite_lr"],
            "transition/u6_opposite_rl_count": bridge_live
            * components["u6_opposite_rl"],
            "transition/u6_liftoff_rejected_already_credited": bridge_live
            * components["u6_liftoff_rej_already_credited"],
            "transition/u6_opp_td_rejected_min_air": components[
                "u6_opp_td_rej_min_air"
            ],
            "transition/u6_opp_td_rejected_stability": components[
                "u6_opp_td_rej_stability"
            ],
            "transition/u6_opp_td_rejected_slip": components["u6_opp_td_rej_slip"],
            "transition/u6_opp_td_rejected_quiet": components["u6_opp_td_rej_quiet"],
            "transition/u6_opp_td_rejected_side_memory": bridge_live
            * components["u6_opp_td_rej_side_memory"],
            "transition/u6_completion_rejected_no_alternation": gate
            * (1.0 - done_f)
            * components["u6_completion_rej_no_alternation"],
            "transition/u6_expected_load_ratio": components["u6_expected_load_ratio"],
            "transition/u6_expected_unload_progress": bridge_live
            * components["u6_expected_unload_progress"],
            "transition/u6_expected_reload_regress": bridge_live
            * components["u6_expected_reload_regress"],
            "transition/u6_expected_foot_still_loaded": bridge_live
            * components["u6_expected_foot_still_loaded"],
            "transition/u6_expected_left_unload_crossing": bridge_live
            * components["u6_expected_left_unload_crossing"],
            "transition/u6_expected_right_unload_crossing": bridge_live
            * components["u6_expected_right_unload_crossing"],
            "transition/u6_expected_air_ticks_1_4": components[
                "u6_expected_air_ticks_1_4"
            ],
            "transition/u6_expected_air_ticks_ge_5": components[
                "u6_expected_air_ticks_ge_5"
            ],
            "transition/u6_first_liftoff_bootstrap_count": components[
                "u6_first_liftoff_bootstrap"
            ],
            "transition/u6_liftoff_armed_fraction": components[
                "u6_liftoff_armed_effective"
            ],
            "transition/u6_commit_outstanding_fraction": jp.float32(
                components["u6v6_gait_phase_pre"] == 2.0
            ),
            "transition/u6_opposite_touchdown_count": bridge_live
            * components["u6_alternating_touchdown_bridge"],
            "transition/u6_gait_phase_wait_liftoff": jp.float32(
                components["u6v6_gait_phase_pre"] == 1.0
            ),
            "transition/u6_gait_phase_wait_touchdown": jp.float32(
                components["u6v6_gait_phase_pre"] == 2.0
            ),
            "transition/u6_gait_phase_wait_completion": jp.float32(
                components["u6v6_gait_phase_pre"] == 3.0
            ),
            "transition/u6_command_reward_gate_mean": components[
                "u6_command_reward_gate_mean"
            ],
            "transition/u6v7_nonquiet_zero_motion_baseline": components[
                "u6v7_nonquiet_zero_motion_baseline"
            ],
            "transition/u6v7_nonquiet_corrected_tracking": components[
                "u6v7_nonquiet_corrected_tracking"
            ],
            "reward/u6_expected_unload_progress": bridge_live
            * u6_unload_progress
            * self.dt,
            "reward/u6_expected_foot_still_loaded": bridge_live
            * u6_still_loaded
            * self.dt,
        }
    )
    if self._u6v10_unified_opposite_touchdown:
        metrics["reward/u6_opposite_touchdown_bridge"] = _c_opposite
    if self._v11_minimal_gait_core:
        metrics.update(
            {
                "v11/shared_mean_head_enabled": jp.ones(()),
                "v11/phase_routing_enabled": jp.zeros(()),
                "v11/phase_balanced_ppo_enabled": jp.float32(
                    1.0 if self._u6v7_phase_balanced_ppo else 0.0
                ),
                "v11/phase_dense_reward_gate_enabled": jp.float32(
                    0.0 if self._v11_disable_phase_dense_reward_gate else 1.0
                ),
                "v11/locomotion_request_fraction": components["v11_motion_gate"],
                "v11/u6_movement_active_fraction": components["v11_u6_movement_active"],
                "v11/gate_disagreement_fraction": components["v11_gate_disagreement"],
                "v11/motion_double_support_ticks": components[
                    "v11_motion_double_support_ticks"
                ],
                "v11/motion_double_support_over_grace_fraction": components[
                    "v11_motion_ds_over_grace"
                ],
                "v11/raw_motion_double_support_duration_penalty": components[
                    "v11_motion_double_support_duration_penalty"
                ],
                "v11/raw_load_transfer_progress": components[
                    "v11_load_transfer_progress"
                ],
                "v11/raw_swing_foot_command_progress": components[
                    "v11_swing_foot_command_progress"
                ],
                "v11/raw_swing_clearance_progress": components[
                    "v11_swing_clearance_progress"
                ],
                "v11/raw_touchdown_load_acceptance_progress": components[
                    "v11_touchdown_load_acceptance_progress"
                ],
                "v11/raw_support_exchange_progress": components[
                    "v11_support_exchange_progress"
                ],
                "v11/weighted_motion_double_support_duration_penalty": -(
                    self._v11_motion_ds_penalty_weight
                    * components["v11_motion_double_support_duration_penalty"]
                ),
                "v11/weighted_load_transfer_progress": self._v11_load_transfer_weight
                * components["v11_load_transfer_progress"],
                "v11/weighted_swing_foot_command_progress": self._v11_swing_foot_weight
                * components["v11_swing_foot_command_progress"],
                "v11/weighted_swing_clearance_progress": self._v11_swing_clearance_weight
                * components["v11_swing_clearance_progress"],
                "v11/weighted_touchdown_load_acceptance_progress": self._v11_touchdown_weight
                * components["v11_touchdown_load_acceptance_progress"],
                "v11/weighted_support_exchange_progress": self._v11_support_exchange_weight
                * components["v11_support_exchange_progress"],
                "v11/swing_candidate_left_fraction": components[
                    "v11_swing_candidate_left"
                ],
                "v11/swing_candidate_right_fraction": components[
                    "v11_swing_candidate_right"
                ],
                "v11/touchdown_candidate_count": components["v11_touchdown_candidate"],
                "v11/lam_left": components["v11_lam_left"],
                "v11/lam_right": components["v11_lam_right"],
                "reward_raw/v12_coupled_role_transfer_progress": components[
                    "v12_coupled_role_transfer_progress"
                ],
                "reward_weighted/v12_coupled_role_transfer_progress": self._v12_role_transfer_weight
                * components["v12_coupled_role_transfer_progress"],
                "v12/coupled_role_transfer_active_fraction": components[
                    "v12_role_transfer_active"
                ],
            }
        )
    if self._u6v10_reward_accounting:
        metrics.update(
            self._u6v10_accounting_metrics(
                components,
                scaled,
                gate,
                group_totals,
                u6v10_frozen_clipped,
                u6v10_actual,
                reward,
            )
        )
    if self._u13_telemetry:
        metrics.update(
            self._u13_metrics(
                raw_transfer_completion=_u13_raw_transfer_completion,
                gated_transfer_completion=components["support_transfer_completion"],
                prev_expected_side=_u13_prev_expected_side,
                next_expected_side=_u13_next_expected_side,
                transfer_timeout=_u13_transfer_timeout,
                components=components,
                scaled=scaled,
                gate=gate,
            )
        )
    metrics.update(
        self._unified_metrics(
            cmd_state,
            group_totals,
            gate,
            e_path_x,
            e_path_y,
            fwd_progress,
            raw_wz=raw_wz,
            wz_dc=wz_dc,
            wz_ac=wz_ac,
            yaw_unwrapped=yaw_unwrapped,
            heading_response_ratio_value=heading_ratio,
            under_response_contribution=components["u4_wz_under_response_penalty"]
            + components["u6_under_response_penalty"],
            wz_safety_gate=components["u4_wz_safety_gate"],
        )
    )
    mvx_now = self.get_local_linvel(data)[0]
    heading_error_now = ucp.wrap_to_pi(cmd_state.heading_target - yaw_now)
    u4_metrics, u4_post_turn_ticks, u4_post_turn_settled = self._u4_unified_metrics(
        cmd_state.command,
        raw_wz,
        mvx_now,
        e_path_y,
        heading_error_now,
        components,
        info,
        terminal_fall,
    )
    metrics.update(u4_metrics)
    strict_yaw_dc_metrics = self._strict_yaw_dc_metrics(
        cmd_state.command,
        raw_wz,
        wz_dc,
        wz_ac,
        mvx_now,
        e_path_y,
        heading_ratio,
        heading_progress_valid,
        components,
        info,
        terminal_fall,
    )
    metrics.update(strict_yaw_dc_metrics)
    new_info = self._write_back(
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
    )
    new_info = self._apply_quiet_freeze(new_info, step_info, gate)
    if self._canonical_side:
        new_info = self._reset_bout_local(new_info, rising_edge)
    new_info.update(ucp.to_info(cmd_state))
    new_info["command"] = cmd_state.command
    new_info["rng"] = rng
    new_info["prev_root_xy"] = root_xy
    new_info["unified_forward_progress"] = fwd_progress
    new_info["unified_e_path_x"] = e_path_x
    new_info["unified_e_path_y"] = e_path_y
    new_info["u4_post_turn_ticks"] = u4_post_turn_ticks
    new_info["u4_post_turn_settled"] = u4_post_turn_settled
    lc_post, rc_post = self._foot_contacts(data)
    new_info["u6_air_ticks_left"] = jp.where(
        jp.float32(lc_post) > 0.5, 0.0, info["u6_air_ticks_left"] + 1.0
    )
    new_info["u6_air_ticks_right"] = jp.where(
        jp.float32(rc_post) > 0.5, 0.0, info["u6_air_ticks_right"] + 1.0
    )
    new_info.update(
        u6_event_state_update(
            components,
            prev_side=info["u6_last_credited_side"],
            liftoff_armed=info.get("u6_liftoff_armed", jp.zeros(())),
            commit_outstanding=info.get("u6_commit_outstanding", jp.zeros(())),
            done=done_f,
            bootstrap_used_prev=info.get("u6_bout_bootstrap_used", jp.zeros(())),
            side_aware=self._u6v5_side_aware,
        )
    )
    if self._v11_minimal_gait_core:
        _v11_alive = 1.0 - done_f
        _v11_ds_ticks = components["v11_next_motion_double_support_ticks"] * _v11_alive
        new_info["v11_motion_double_support_ticks"] = _v11_ds_ticks
        _v11_ds_max = jp.maximum(
            info["v11_motion_double_support_ticks_max"], _v11_ds_ticks
        )
        new_info["v11_motion_double_support_ticks_max"] = _v11_ds_max * _v11_alive
        new_info["v11_load_phi_prev"] = components["v11_next_load_phi"]
        new_info["v11_load_phi_valid"] = (
            components["v11_next_load_phi_valid"] * _v11_alive
        )
        new_info["v11_load_context"] = components["v11_next_load_context"]
        new_info["v11_swing_context"] = components["v11_next_swing_context"]
        new_info["v11_swing_foot_phi_prev"] = components["v11_next_swing_foot_phi"]
        new_info["v11_swing_foot_phi_valid"] = (
            components["v11_next_swing_foot_phi_valid"] * _v11_alive
        )
        new_info["v11_clearance_phi_prev"] = components["v11_next_clearance_phi"]
        new_info["v11_clearance_phi_valid"] = (
            components["v11_next_clearance_phi_valid"] * _v11_alive
        )
        new_info["v11_recent_landed_side"] = (
            components["v11_next_recent_landed_side"] * _v11_alive
        )
        new_info["v11_touchdown_phi_prev"] = components["v11_next_touchdown_phi"]
        new_info["v11_touchdown_phi_valid"] = (
            components["v11_next_touchdown_phi_valid"] * _v11_alive
        )
        new_info["v11_exchange_phi_prev"] = components["v11_next_exchange_phi"]
        new_info["v11_exchange_phi_valid"] = (
            components["v11_next_exchange_phi_valid"] * _v11_alive
        )
        new_info["v12_role_transfer_phi_prev"] = components[
            "v12_next_role_transfer_phi"
        ]
        new_info["v12_role_transfer_phi_valid"] = (
            components["v12_next_role_transfer_phi_valid"] * _v11_alive
        )
        _p2_now = jp.float32(new_info["u6_commit_outstanding"] > 0.5)
        _p2_prev = info["v11_phase2_prev"]
        _p2_dwell = (info["v11_phase2_dwell"] + 1.0) * _p2_now
        new_info["v11_phase2_prev"] = _p2_now * _v11_alive
        new_info["v11_phase2_dwell"] = _p2_dwell * _v11_alive
        _p2_dwell_max = jp.maximum(info["v11_phase2_dwell_max"], _p2_dwell)
        new_info["v11_phase2_dwell_max"] = _p2_dwell_max * _v11_alive
        metrics.update(
            {
                "v11/phase2_enter_count": (1.0 - _p2_prev) * _p2_now,
                "v11/phase2_exit_count": _p2_prev * (1.0 - _p2_now),
                "v11/phase2_same_tick_enter_exit_count": jp.clip(
                    components["u6_expected_liftoff_commit"], 0.0, 1.0
                )
                * (1.0 - _p2_now),
                "v11/phase2_dwell": _p2_dwell,
                "v11/phase2_dwell_max": _p2_dwell_max,
                "v11/motion_double_support_ticks_max": _v11_ds_max,
            }
        )
    new_info["u6v6_load_phi_prev"] = components["u6v6_load_phi_now"]
    new_info["u6v6_load_phi_prev_side"] = components["u6v6_expected_side_pre"]
    if self._enable_gait_phase_observation:
        _coin = jp.where(jax.random.bernoulli(gait_side_rng, 0.5), 1.0, -1.0)
        _movement_now = components["u6_movement_active"]
        _candidate = jp.where(rising_edge, _coin, info["u6v6_expected_side_init"])
        new_info["u6v6_expected_side_init"] = jp.where(
            _movement_now < 0.5, 0.0, _candidate
        )
    else:
        new_info["u6v6_expected_side_init"] = info.get(
            "u6v6_expected_side_init", jp.zeros(())
        )
    if self._u6v7_phase_balanced_ppo or self._v11_minimal_gait_core:
        new_info["u6v7_gait_phase"] = components["u6v7_gait_phase"]
    new_info["wz_dc"] = wz_dc
    new_info["yaw_unwrapped"] = yaw_unwrapped
    new_info["prev_yaw_for_unwrap"] = yaw_now
    new_info["heading_segment_active"] = is_yaw_active.astype(jp.float32)
    new_info["heading_segment_start_yaw_unwrapped"] = segment_start_yaw_unwrapped
    new_info["heading_segment_elapsed_ticks"] = segment_elapsed_ticks
    new_info["yaw_teacher_gate_smooth"] = new_gate_smooth
    next_step = step_count + 1
    next_legacy = self._v18_legacy_phase(next_step)
    next_phase = self._v18_event_phase(new_info, next_legacy)
    new_info = self._v18_apply_phase(new_info, next_phase)
    next_phase_info = dict(new_info)
    next_phase_info["step"] = next_phase["local_step"]
    next_q_ref, _, _ = self._q_ref_phase(
        next_phase["phase_id"],
        next_phase["target_left_swing"],
        next_phase["target_right_swing"],
        next_phase_info,
    )
    new_info["periodic_q_ref"] = next_q_ref
    obs = self._get_obs(data, new_info)
    return state.replace(
        data=data, obs=obs, reward=reward, done=done_f, metrics=metrics, info=new_info
    )
