# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0. See LICENSE.
# Adapted from the private research working tree based on MuJoCo Playground.
# Robot descriptions and model parameters are intentionally excluded.

from __future__ import annotations
from typing import Dict
import jax
import jax.numpy as jp
from biped_locomotion.commands import process as ucp
from biped_locomotion.rewards import unified_contract as ucfg
from biped_locomotion.gait.unified_events import (
    ACTIVE_YAW_EQUALITY_TERMS,
    ACTIVE_YAW_OTHER_POSITIVE_TERMS,
    u6_decode_gait_phase,
    u6_event_algebra,
)


def SchaftUnifiedJoystickV1__unified_components(
    self,
    data,
    info,
    cmd_state,
    components,
    *,
    e_path_x,
    e_path_y,
    forward_progress,
    clipped_action,
    action,
    wz_dc,
    heading_response_ratio_value,
    heading_progress_valid,
) -> Dict[str, jax.Array]:
    """Overwrite the six V24 velocity/path terms and add dense + quiet terms.

    Every formula here is document 10 §12.2-§12.5 verbatim, modulated only by
    the two U4R-repair calibration hooks (``self._wz_validity_gate_weight``,
    ``self._forward_turn_discount``): both default to 0.0, at which every
    formula below reduces exactly to the frozen §12.2-§12.5 form (see
    ``ucfg.default_config``). Only a stage that explicitly overrides one of
    these two config keys sees different numerics.
    """
    del e_path_x
    cmd = cmd_state.command
    mvx, mvy = (self.get_local_linvel(data)[0], self.get_local_linvel(data)[1])
    gyro_z = self.get_gyro(data)[2]
    psi_ref = cmd_state.heading_target
    yaw = self._forward_yaw_from_quat(self.get_orientation(data))
    heading_error = ucp.wrap_to_pi(psi_ref - yaw)
    v_ref = ucp.reference_velocity_world(cmd_state, self._compat)
    w = ucp.command_region_weights(v_ref, psi_ref, cmd[2])
    fwd_str = w.w_fwd * w.w_str
    mode_discount = jp.take(self._forward_turn_discount_by_mode, info["u4_mode_id"])
    forward_discount = (1.0 - self._forward_turn_discount * w.w_turn) * (
        1.0 - mode_discount * w.w_turn
    )
    is_lateral = (cmd_state.category == ucfg.CAT_LATERAL).astype(jp.float32)
    lateral_forward_progress_factor = (
        1.0 - self._lateral_forward_progress_gate_weight * is_lateral
    )
    out: Dict[str, jax.Array] = {}
    tgt_vx, tgt_vy = (cmd[0], cmd[1])
    sig = ucfg.V24_TRACKING_SIGMA
    out["forward_velocity_tracking"] = (
        forward_discount
        * lateral_forward_progress_factor
        * jp.exp(-(((mvx - tgt_vx) / sig) ** 2) - ((mvy - tgt_vy) / sig) ** 2)
    )
    tgt_speed = jp.sqrt(tgt_vx**2 + tgt_vy**2)
    d_hat_v = jp.array([tgt_vx, tgt_vy]) / jp.maximum(tgt_speed, 1e-06)
    v_aligned = jp.dot(jp.array([mvx, mvy]), d_hat_v)
    gate = components.get("locomotion_gate", jp.ones(()))
    out["forward_velocity_progress"] = (
        forward_discount
        * lateral_forward_progress_factor
        * gate
        * jp.clip(v_aligned / jp.maximum(tgt_speed, 1e-06), 0.0, 1.0)
        * jp.exp(-(((v_aligned - tgt_speed) / ucfg.V24_PROGRESS_SIGMA) ** 2))
    )
    out["yaw_drift_penalty"] = (
        jp.abs(heading_error) + 0.1 * jp.abs(gyro_z - cmd[2])
    ) / ucfg.YAW_DRIFT_NORM
    out["lateral_drift_penalty"] = jp.clip(
        (jp.abs(e_path_y) - ucfg.LATERAL_DRIFT_DEADBAND) / ucfg.LATERAL_PATH_SCALE,
        0.0,
        1.0,
    )
    out["persistent_lateral_path"] = (
        jp.maximum(jp.abs(e_path_y) - ucfg.PERSISTENT_LATERAL_DEADBAND, 0.0)
        / ucfg.LATERAL_PATH_SCALE
    )
    out["forward_displacement_progress"] = (
        jp.clip(forward_progress, 0.0, ucfg.FORWARD_DISPLACEMENT_CLIP)
        * fwd_str
        * lateral_forward_progress_factor
    )
    out["lateral_vx_leakage_penalty"] = (
        self._lateral_vx_leakage_weight * is_lateral * jp.abs(mvx)
    )
    out["u13_track_planar"] = self._u13_track_planar_weight * jp.exp(
        -((mvx - cmd[0]) ** 2 + (mvy - cmd[1]) ** 2) / ucfg.U13_TRACK_PLANAR_SIGMA**2
    )
    out["u13_track_yaw"] = self._u13_track_yaw_weight * jp.exp(
        -((gyro_z - cmd[2]) ** 2) / ucfg.U13_TRACK_YAW_SIGMA**2
    )
    out["u13_track_planar_raw"] = jp.exp(
        -((mvx - cmd[0]) ** 2 + (mvy - cmd[1]) ** 2) / ucfg.U13_TRACK_PLANAR_SIGMA**2
    )
    out["u13_track_yaw_raw"] = jp.exp(
        -((gyro_z - cmd[2]) ** 2) / ucfg.U13_TRACK_YAW_SIGMA**2
    )
    for name in ucfg.PLACEMENT_TERMS:
        if name in components:
            out[name] = components[name] * fwd_str
    stability_gate = (
        components["v7_upright_gate"]
        * components["v7_root_height_gate"]
        * components["v7_angular_stability_gate"]
    )
    forward_validity = jp.clip(
        mvx / jp.maximum(jp.abs(cmd[0]), self._dense_delta[0]), 0.0, 1.0
    )
    wz_validity = stability_gate * forward_validity
    wz_validity_factor = 1.0 - self._wz_validity_gate_weight * (1.0 - wz_validity)
    speed_ramp = jp.clip((mvx - 0.03) / (0.05 - 0.03), 0.0, 1.0)
    speed_gate_factor = 1.0 - self._wz_forward_speed_gate_weight * (1.0 - speed_ramp)
    same_side_ok = 1.0 - components["qualified_same_side_repeat"]
    transfer_ok = 1.0 - components["support_transfer_timeout"]
    gait_validity = same_side_ok * transfer_ok
    extended_validity_factor = 1.0 - self._wz_extended_validity_gate_weight * (
        1.0 - gait_validity
    )
    wz_positive_credit_factor = (
        wz_validity_factor * speed_gate_factor * extended_validity_factor
    )
    wz_for_dense = jp.where(self._wz_dense_uses_dc > 0.5, wz_dc, gyro_z)
    wz_active_gate = (jp.abs(cmd[2]) > self._dense_delta[2]).astype(jp.float32)
    signed_ratio = jp.sign(cmd[2]) * wz_for_dense / jp.maximum(jp.abs(cmd[2]), 0.05)
    under_response_error = jp.maximum(0.35 - signed_ratio, 0.0)
    out["u4_wz_signed_ratio"] = signed_ratio
    out["u4_wz_under_response_gate"] = wz_active_gate
    out["u4_wz_under_response_penalty"] = -(
        wz_active_gate
        * self._wz_under_response_weight
        * jp.square(under_response_error)
    )
    measured = jp.array([mvx, mvy, wz_for_dense])
    eps = 1e-06
    axis_active = [
        (jp.abs(cmd[i]) > self._dense_delta[i]).astype(jp.float32) for i in range(3)
    ]
    movement_active_pre = jp.clip(
        axis_active[0] + axis_active[1] + axis_active[2], 0.0, 1.0
    )
    u6v7_active = self._u6v7_zero_motion_baseline * movement_active_pre
    u6v7_baseline_total = jp.zeros(())
    u6v7_corrected_total = jp.zeros(())
    _fvt_prefactor = forward_discount * lateral_forward_progress_factor
    _fvt_zero = _fvt_prefactor * jp.exp(-((tgt_vx / sig) ** 2) - (tgt_vy / sig) ** 2)
    _fvt_corrected = jp.maximum(out["forward_velocity_tracking"] - _fvt_zero, 0.0)
    u6v7_baseline_total = u6v7_baseline_total + u6v7_active * _fvt_zero
    u6v7_corrected_total = u6v7_corrected_total + u6v7_active * _fvt_corrected
    out["forward_velocity_tracking"] = jp.where(
        u6v7_active > 0.5, _fvt_corrected, out["forward_velocity_tracking"]
    )
    _outstanding_pre = info.get("u6_commit_outstanding", jp.zeros(()))
    _latched_pre = info.get("u6_alternation_latched", jp.zeros(()))
    _last_credited_pre = info.get("u6_last_credited_side", jp.zeros(()))
    _has_prev_pre = (jp.abs(_last_credited_pre) > 0.5).astype(jp.float32)
    _side_init_pre = info.get("u6v6_expected_side_init", jp.zeros(()))
    u6v6_gait_phase_pre = u6_decode_gait_phase(
        movement_active_pre,
        _latched_pre,
        _outstanding_pre,
        corrected=self._u6v9_corrected_gait_phase,
    )
    out["u6v6_gait_phase_pre"] = u6v6_gait_phase_pre
    u6v6_expected_side_pre = jp.where(
        _has_prev_pre > 0.5, -_last_credited_pre, _side_init_pre
    )
    out["u6v6_expected_side_pre"] = u6v6_expected_side_pre
    command_reward_gate = jp.where(
        u6v6_gait_phase_pre <= 1.0,
        self._command_reward_gate_phase01,
        jp.where(
            u6v6_gait_phase_pre <= 2.0,
            self._command_reward_gate_phase2,
            self._command_reward_gate_phase3,
        ),
    )
    out["u6_command_reward_gate_mean"] = command_reward_gate
    if self._v11_disable_phase_dense_reward_gate:
        command_reward_gate_applied = jp.ones_like(command_reward_gate)
    else:
        command_reward_gate_applied = command_reward_gate
    under_axis = []
    for i, axis in enumerate(("vx", "vy", "wz")):
        m_a, c_a = (measured[i], cmd[i])
        delta, sigma = (self._dense_delta[i], self._dense_sigma[i])
        terms = ucp.dense_signed_tracking(m_a, c_a, delta, sigma)
        track, prog, wrong = (terms.track, terms.prog, terms.wrong)
        raw_g = jp.exp(-(((m_a - c_a) / sigma) ** 2))
        zero_g = jp.exp(-(((0.0 - c_a) / sigma) ** 2))
        norm_track = jp.clip(
            (raw_g - zero_g) / jp.maximum(1.0 - zero_g, eps), -1.0, 1.0
        )
        use_norm = self._zero_baselined_tracking * axis_active[i]
        track = jp.where(use_norm > 0.5, norm_track, track)
        track_zero = jp.where(use_norm > 0.5, 0.0, 1.0 / (1.0 + (c_a / sigma) ** 2))
        track_corrected = jp.maximum(track - track_zero, 0.0) + jp.minimum(track, 0.0)
        u6v7_baseline_total = u6v7_baseline_total + u6v7_active * track_zero
        u6v7_corrected_total = u6v7_corrected_total + u6v7_active * jp.maximum(
            track_corrected, 0.0
        )
        track = jp.where(u6v7_active > 0.5, track_corrected, track)
        response = jp.clip(jp.sign(c_a) * m_a / jp.maximum(jp.abs(c_a), eps), 0.0, 1.0)
        under_axis.append(axis_active[i] * (1.0 - response))
        if axis == "wz":
            track = track * wz_positive_credit_factor
            prog = prog * wz_positive_credit_factor
        track = jp.maximum(track, 0.0) * command_reward_gate_applied + jp.minimum(
            track, 0.0
        )
        prog = jp.maximum(prog, 0.0) * command_reward_gate_applied + jp.minimum(
            prog, 0.0
        )
        out[f"track_{axis}"] = track
        out[f"prog_{axis}"] = prog
        out[f"wrong_{axis}"] = wrong
    out["u6v7_nonquiet_zero_motion_baseline"] = u6v7_baseline_total
    out["u6v7_nonquiet_corrected_tracking"] = u6v7_corrected_total
    out["u6v7_gait_phase"] = u6v6_gait_phase_pre
    out["u6_under_response_penalty"] = -(
        self._under_response_translational_weight * (under_axis[0] + under_axis[1])
        + self._under_response_yaw_weight * under_axis[2]
    )
    heading_progress_gate = stability_gate * speed_ramp * gait_validity
    out["u4_wz_safety_gate"] = heading_progress_gate
    out["heading_progress_reward"] = (
        heading_progress_valid.astype(jp.float32)
        * heading_progress_gate
        * self._heading_progress_reward_weight
        * jp.clip(heading_response_ratio_value, -1.0, 1.0)
    )
    other_positive_factor = jp.where(
        wz_active_gate > 0.5, self._active_yaw_other_positive_scale, 1.0
    )
    for name in ACTIVE_YAW_OTHER_POSITIVE_TERMS:
        if name in components:
            out[name] = components[name] * other_positive_factor
    equality_factor = jp.where(
        wz_active_gate > 0.5, self._active_yaw_equality_scale, 1.0
    )
    for name in ACTIVE_YAW_EQUALITY_TERMS:
        if name in out:
            out[name] = out[name] * equality_factor
    if "double_support_penalty" in components:
        out["double_support_penalty"] = components["double_support_penalty"] * (
            1.0 + (self._double_support_yaw_multiplier - 1.0) * w.w_turn
        )
    trans_active = jp.maximum(axis_active[0], axis_active[1])
    yaw_active = axis_active[2]
    resp_trans = jp.clip(v_aligned / jp.maximum(tgt_speed, eps), 0.0, 1.0)
    resp_yaw_step = jp.clip(
        jp.sign(cmd[2]) * gyro_z / jp.maximum(jp.abs(cmd[2]), eps), 0.0, 1.0
    )
    n_progress_axes = jp.maximum(trans_active + yaw_active, 1.0)
    progress_score = (
        trans_active * resp_trans + yaw_active * resp_yaw_step
    ) / n_progress_axes
    planar_speed = jp.sqrt(mvx * mvx + mvy * mvy)
    drift_ok = jp.where(
        trans_active > 0.5,
        1.0,
        (planar_speed <= self._valid_step_drift_max).astype(jp.float32),
    )
    progress_ok = (progress_score >= self._valid_step_progress_min).astype(
        jp.float32
    ) * drift_ok
    slip_ok = (
        components.get("stance_slip", jp.zeros(())) <= self._valid_step_stance_slip_max
    ).astype(jp.float32)
    completion_candidate = (
        components.get("support_transfer_completion", jp.zeros(()))
        * (
            1.0
            - jp.clip(
                components.get("qualified_same_side_repeat", jp.zeros(())), 0.0, 1.0
            )
        )
        * (
            1.0
            - jp.clip(
                components.get("support_transfer_timeout", jp.zeros(())), 0.0, 1.0
            )
        )
        * stability_gate
        * slip_ok
        * progress_ok
        * jp.clip(axis_active[0] + axis_active[1] + axis_active[2], 0.0, 1.0)
    )
    out["u6_step_progress_score"] = progress_score
    lc_now, rc_now = self._foot_contacts(data)
    lc_f, rc_f = (jp.float32(lc_now), jp.float32(rc_now))
    prev_lc = info.get("prev_left_contact", jp.zeros(()))
    prev_rc = info.get("prev_right_contact", jp.zeros(()))
    air_l_prev = info.get("u6_air_ticks_left", jp.zeros(()))
    air_r_prev = info.get("u6_air_ticks_right", jp.zeros(()))
    movement_active = jp.clip(
        axis_active[0] + axis_active[1] + axis_active[2], 0.0, 1.0
    )
    out.update(
        u6_event_algebra(
            lc_f=lc_f,
            rc_f=rc_f,
            prev_lc=prev_lc,
            prev_rc=prev_rc,
            air_l_prev=air_l_prev,
            air_r_prev=air_r_prev,
            prev_side=info.get("u6_last_credited_side", jp.zeros(())),
            liftoff_armed=info.get("u6_liftoff_armed", jp.zeros(())),
            commit_outstanding=info.get("u6_commit_outstanding", jp.zeros(())),
            alternation_latched=info.get("u6_alternation_latched", jp.zeros(())),
            stability_gate=stability_gate,
            slip_ok=slip_ok,
            movement_active=movement_active,
            completion_candidate=completion_candidate,
            min_air_ticks=self._bridge_min_air_ticks,
            liftoff_min_air_ticks=self._expected_liftoff_min_air_ticks,
            liftoff_stale_ticks=self._expected_liftoff_stale_ticks,
            expected_side_init=info.get("u6v6_expected_side_init", jp.zeros(())),
            bootstrap_used=info.get("u6_bout_bootstrap_used", jp.zeros(())),
            first_liftoff_bootstrap=self._u6v8r_first_liftoff_bootstrap,
            side_aware=self._u6v5_side_aware,
            require_alternation=self._require_alternation_for_completion,
        )
    )
    fz_left, fz_right = self._foot_vertical_ground_reaction_forces(data)
    _fz_total = fz_left + fz_right + eps
    _ratio_if_left = jp.clip(fz_left / _fz_total, 0.0, 1.0)
    _ratio_if_right = jp.clip(fz_right / _fz_total, 0.0, 1.0)
    expected_foot_load_ratio = jp.where(
        u6v6_expected_side_pre < -0.5,
        _ratio_if_left,
        jp.where(u6v6_expected_side_pre > 0.5, _ratio_if_right, 0.0),
    )
    out["u6_expected_load_ratio"] = expected_foot_load_ratio
    phi_now = 1.0 - expected_foot_load_ratio
    _phi_prev_side = info.get("u6v6_load_phi_prev_side", jp.zeros(()))
    _side_match = jp.float32(jp.abs(u6v6_expected_side_pre - _phi_prev_side) < 0.5)
    phi_prev_eff = jp.where(
        _side_match > 0.5, info.get("u6v6_load_phi_prev", jp.zeros(())), phi_now
    )
    raw_unload_delta = phi_now - phi_prev_eff
    out["u6v6_load_phi_now"] = phi_now
    _has_expected_side = (jp.abs(u6v6_expected_side_pre) > 0.5).astype(jp.float32)
    _wait_liftoff = (u6v6_gait_phase_pre == 1.0).astype(jp.float32)
    unload_active = movement_active * _has_expected_side * _wait_liftoff
    out["u6_expected_unload_progress"] = unload_active * jp.maximum(
        raw_unload_delta, 0.0
    )
    out["u6_expected_reload_regress"] = unload_active * jp.maximum(
        -raw_unload_delta, 0.0
    )
    out["u6_expected_foot_still_loaded"] = unload_active * expected_foot_load_ratio
    _unload_threshold = 0.15
    _ratio_prev_eff = 1.0 - phi_prev_eff
    _crossed = (
        jp.float32(
            (expected_foot_load_ratio <= _unload_threshold)
            & (_ratio_prev_eff > _unload_threshold)
        )
        * _side_match
    )
    out["u6_expected_left_unload_crossing"] = _crossed * jp.float32(
        u6v6_expected_side_pre < -0.5
    )
    out["u6_expected_right_unload_crossing"] = _crossed * jp.float32(
        u6v6_expected_side_pre > 0.5
    )
    if self._v11_minimal_gait_core:
        out.update(
            self._v11_gait_core_components(
                data=data,
                info=info,
                cmd=cmd,
                motion_gate=cmd_state.locomotion_request,
                expected_side=u6v6_expected_side_pre,
                stability_gate=stability_gate,
                lc_f=lc_f,
                rc_f=rc_f,
                prev_lc=prev_lc,
                prev_rc=prev_rc,
                fz_left=fz_left,
                fz_right=fz_right,
                lam_left=_ratio_if_left,
                lam_right=_ratio_if_right,
            )
        )
        out["v11_gate_disagreement"] = jp.float32(
            jp.abs(jp.clip(cmd_state.locomotion_request, 0.0, 1.0) - movement_active)
            > 0.5
        )
        out["v11_u6_movement_active"] = movement_active
    scale = self._action_scale_jp
    q = data.qpos[self._qposadr_jp]
    qd = data.qvel[self._dofadr_jp]
    q_nom = info["bridge_q_ref_base"]
    left_contact, right_contact = self._foot_contacts(data)
    lc, rc = (jp.float32(left_contact), jp.float32(right_contact))
    fm = self._foot_motion(data, info)
    out["quiet_vx"] = 1.0 / (1.0 + (mvx / ucfg.QUIET_VX_SIGMA) ** 2)
    out["quiet_vy"] = 1.0 / (1.0 + (mvy / ucfg.QUIET_VY_SIGMA) ** 2)
    out["quiet_yaw"] = 1.0 / (1.0 + (gyro_z / ucfg.QUIET_YAW_SIGMA) ** 2)
    out["quiet_pose"] = jp.exp(-jp.mean(((q - q_nom) / scale) ** 2))
    out["quiet_joint_velocity"] = jp.mean((qd / (scale / self.dt)) ** 2)
    out["quiet_double_support"] = lc * rc
    out["quiet_foot_stillness"] = (
        jp.linalg.norm(fm["left_vel"][:2]) + jp.linalg.norm(fm["right_vel"][:2])
        if "left_vel" in fm
        else jp.zeros(())
    )
    out["quiet_contact_loss"] = 1.0 - lc * rc
    out["action_magnitude"] = jp.sum(jp.square(clipped_action))
    out["action_rate"] = jp.sum(jp.square(clipped_action - info["last_act"]))
    return out


def SchaftUnifiedJoystickV1__v11_gait_core_components(
    self,
    *,
    data,
    info,
    cmd,
    motion_gate,
    expected_side,
    stability_gate,
    lc_f,
    rc_f,
    prev_lc,
    prev_rc,
    fz_left,
    fz_right,
    lam_left,
    lam_right,
) -> Dict[str, jax.Array]:
    """The V11 dense contact/load-driven gait core.

    NOTHING here reads the U6 discrete phase, the alternation latch, the
    commit-outstanding state, ``bridge_live``, the legacy V18 clock or the
    completion event: every term is gated by the CANONICAL command gate
    (``cmd_state.locomotion_request``) plus a purely PHYSICAL swing/contact
    condition. The FSM keeps its event-detection and telemetry role untouched.

    ``expected_side`` is used only as a SOFT signed selector for the
    load-transfer potential; when it is unknown that potential falls back to a
    symmetric form so either foot may initiate the first step.
    """
    out: Dict[str, jax.Array] = {}
    eps = 1e-06
    sg = jax.lax.stop_gradient
    motion = jp.clip(motion_gate, 0.0, 1.0)
    stable = jp.float32(stability_gate > 0.5)
    both_contact = lc_f * rc_f
    prev_ticks = info.get("v11_motion_double_support_ticks", jp.zeros(()))
    ds_ticks = jp.where(motion * both_contact > 0.5, prev_ticks + 1.0, jp.zeros(()))
    ds_cost = jp.clip(
        (ds_ticks - self._v11_ds_grace_ticks)
        / jp.maximum(self._v11_ds_ramp_ticks, eps),
        0.0,
        1.0,
    )
    out["v11_motion_double_support_ticks"] = ds_ticks
    out["v11_next_motion_double_support_ticks"] = ds_ticks
    out["v11_motion_double_support_duration_penalty"] = ds_cost
    out["v11_motion_ds_over_grace"] = motion * jp.float32(ds_cost > 0.0)
    air_l, air_r = (1.0 - lc_f, 1.0 - rc_f)
    margin = self._v11_swing_candidate_margin
    left_cand = jp.clip(
        air_l * rc_f + both_contact * jp.float32(lam_left < lam_right - margin),
        0.0,
        1.0,
    )
    right_cand = jp.clip(
        air_r * lc_f + both_contact * jp.float32(lam_right < lam_left - margin),
        0.0,
        1.0,
    )
    swing_side = sg(-1.0 * left_cand + 1.0 * right_cand)
    has_swing = sg(jp.clip(left_cand + right_cand, 0.0, 1.0))
    swing_is_left = jp.float32(swing_side < -0.5)
    out["v11_swing_candidate_left"] = motion * left_cand
    out["v11_swing_candidate_right"] = motion * right_cand
    exp_left = jp.float32(expected_side < -0.5)
    exp_right = jp.float32(expected_side > 0.5)
    exp_known = exp_left + exp_right
    phi_load = jp.where(
        exp_known > 0.5,
        exp_left * (lam_right - lam_left) + exp_right * (lam_left - lam_right),
        jp.abs(lam_left - lam_right),
    )
    phi_load = jp.clip(phi_load, -1.0, 1.0)
    load_ctx = sg(exp_known * expected_side)
    load_same = jp.float32(
        jp.abs(load_ctx - info.get("v11_load_context", jp.zeros(()))) < 0.5
    )
    d_load, load_next, load_valid = self._v11_potential_delta(
        phi_load,
        info.get("v11_load_phi_prev", jp.zeros(())),
        info.get("v11_load_phi_valid", jp.zeros(())),
        motion,
        load_same,
    )
    out["v11_load_transfer_progress"] = d_load
    out["v11_next_load_phi"] = load_next
    out["v11_next_load_phi_valid"] = load_valid
    out["v11_next_load_context"] = load_ctx
    left_pos, right_pos = self._foot_positions(data)
    swing_pos = jp.where(swing_is_left > 0.5, left_pos, right_pos)
    stance_pos = jp.where(swing_is_left > 0.5, right_pos, left_pos)
    cmd_planar = jp.array([cmd[0], cmd[1]])
    cmd_mag = jp.linalg.norm(cmd_planar)
    has_dir = jp.float32(cmd_mag > ucfg.QUIET_SPEED_THRESHOLD)
    d_hat = cmd_planar / jp.maximum(cmd_mag, eps)
    phi_swing = jp.clip(
        has_dir * jp.dot(swing_pos[:2] - stance_pos[:2], d_hat), -1.0, 1.0
    )
    swing_ctx = swing_side
    swing_same = jp.float32(
        jp.abs(swing_ctx - info.get("v11_swing_context", jp.zeros(()))) < 0.5
    )
    d_swing, swing_next, swing_valid = self._v11_potential_delta(
        phi_swing,
        info.get("v11_swing_foot_phi_prev", jp.zeros(())),
        info.get("v11_swing_foot_phi_valid", jp.zeros(())),
        motion * has_swing * has_dir,
        swing_same,
    )
    out["v11_swing_foot_command_progress"] = d_swing
    out["v11_next_swing_foot_phi"] = swing_next
    out["v11_next_swing_foot_phi_valid"] = swing_valid
    out["v11_next_swing_context"] = swing_ctx
    swing_height = jp.where(
        swing_is_left > 0.5,
        left_pos[2] - info.get("left_foot_baseline_z", jp.zeros(())),
        right_pos[2] - info.get("right_foot_baseline_z", jp.zeros(())),
    )
    phi_clear = jp.clip(
        swing_height / jp.maximum(self._v11_swing_clearance_target, eps), 0.0, 1.0
    )
    d_clear, clear_next, clear_valid = self._v11_potential_delta(
        phi_clear,
        info.get("v11_clearance_phi_prev", jp.zeros(())),
        info.get("v11_clearance_phi_valid", jp.zeros(())),
        motion * has_swing,
        swing_same,
    )
    out["v11_swing_clearance_progress"] = d_clear
    out["v11_next_clearance_phi"] = clear_next
    out["v11_next_clearance_phi_valid"] = clear_valid
    td_l = (1.0 - prev_lc) * lc_f
    td_r = (1.0 - prev_rc) * rc_f
    landed_now = jp.where(
        td_l * (1.0 - td_r) > 0.5,
        -1.0,
        jp.where(td_r * (1.0 - td_l) > 0.5, 1.0, jp.zeros(())),
    )
    recent_landed = jp.where(
        jp.abs(landed_now) > 0.5,
        landed_now,
        info.get("v11_recent_landed_side", jp.zeros(())),
    )
    recent_landed = sg(recent_landed * motion)
    landed_known = jp.float32(jp.abs(recent_landed) > 0.5)
    landed_is_left = jp.float32(recent_landed < -0.5)
    weight = jp.maximum(self._expected_weight_jp, eps)
    fz_new = jp.where(landed_is_left > 0.5, fz_left, fz_right)
    fz_old = jp.where(landed_is_left > 0.5, fz_right, fz_left)
    contact_new = jp.where(landed_is_left > 0.5, lc_f, rc_f)
    landed_same = jp.float32(
        jp.abs(recent_landed - info.get("v11_recent_landed_side", jp.zeros(()))) < 0.5
    )
    phi_td = landed_known * contact_new * jp.clip(fz_new / weight, 0.0, 1.0)
    d_td, td_next, td_valid = self._v11_potential_delta(
        phi_td,
        info.get("v11_touchdown_phi_prev", jp.zeros(())),
        info.get("v11_touchdown_phi_valid", jp.zeros(())),
        motion * landed_known * contact_new * stable,
        landed_same,
    )
    out["v11_touchdown_load_acceptance_progress"] = d_td
    out["v11_next_touchdown_phi"] = td_next
    out["v11_next_touchdown_phi_valid"] = td_valid
    phi_ex = -landed_known * jp.clip(fz_old / weight, 0.0, 1.0)
    d_ex, ex_next, ex_valid = self._v11_potential_delta(
        phi_ex,
        info.get("v11_exchange_phi_prev", jp.zeros(())),
        info.get("v11_exchange_phi_valid", jp.zeros(())),
        motion * landed_known * stable,
        landed_same,
    )
    out["v11_support_exchange_progress"] = d_ex
    out["v11_next_exchange_phi"] = ex_next
    out["v11_next_exchange_phi_valid"] = ex_valid
    out["v11_next_recent_landed_side"] = recent_landed
    new_load = landed_known * contact_new * jp.clip(fz_new / weight, 0.0, 1.0)
    old_load = jp.clip(fz_old / weight, 0.0, 1.0)
    phi_role = landed_known * jp.clip(jp.minimum(new_load, 1.0 - old_load), 0.0, 1.0)
    role_active = motion * landed_known * stable
    d_role, role_next, role_valid = self._v11_potential_delta(
        phi_role,
        info.get("v12_role_transfer_phi_prev", jp.zeros(())),
        info.get("v12_role_transfer_phi_valid", jp.zeros(())),
        role_active,
        landed_same,
    )
    out["v12_coupled_role_transfer_progress"] = d_role
    out["v12_next_role_transfer_phi"] = role_next
    out["v12_next_role_transfer_phi_valid"] = role_valid
    out["v12_role_transfer_active"] = role_active
    out["v11_lam_left"] = lam_left
    out["v11_lam_right"] = lam_right
    out["v11_touchdown_candidate"] = motion * jp.clip(td_l + td_r, 0.0, 1.0)
    out["v11_motion_gate"] = motion
    return out
