# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0. See LICENSE.
# Adapted from the private research working tree based on MuJoCo Playground.
# Robot descriptions and model parameters are intentionally excluded.

from __future__ import annotations
from typing import Any
import jax
import jax.numpy as jp
from mujoco import mjx


def FootUnload__foot_unload_terms(
    self, data: mjx.Data, info: dict[str, Any]
) -> dict[str, jax.Array]:
    left_contact, right_contact = self._foot_contacts(data)
    left_contact = jp.float32(left_contact)
    right_contact = jp.float32(right_contact)
    target_left = info["target_left_swing"]
    target_right = info["target_right_swing"]
    support_left = info["support_left"]
    support_right = info["support_right"]
    target_contact = target_left * left_contact + target_right * right_contact
    support_contact = support_left * left_contact + support_right * right_contact
    target_no_contact = 1.0 - target_contact
    support_lost = 1.0 - support_contact
    unload_gate = jp.float32(info["step"] >= self._config.unload_start_step)
    left_pos, right_pos = self._foot_positions(data)
    root_y = self.get_root_pos(data)[1]
    support_y = support_left * left_pos[1] + support_right * right_pos[1]
    weight_shift_error = root_y - support_y
    weight_shift = jp.exp(
        -0.5 * jp.square(weight_shift_error / self._config.weight_shift_sigma)
    )
    target_foot_z = target_left * left_pos[2] + target_right * right_pos[2]
    target_baseline_z = (
        target_left * info["left_foot_baseline_z"]
        + target_right * info["right_foot_baseline_z"]
    )
    target_clearance_raw = jp.maximum(target_foot_z - target_baseline_z, 0.0)
    target_clearance_score = jp.clip(
        target_clearance_raw / self._config.target_clearance, 0.0, 1.0
    )
    target_clearance_reward = (
        unload_gate * target_no_contact * support_contact * target_clearance_score
    )
    target_air_time_next = (
        (info["target_air_time"] + self.dt) * target_no_contact * support_contact
    )
    target_air_time_score = jp.clip(
        target_air_time_next / self._config.min_air_time, 0.0, 1.0
    )
    target_air_time_reward = unload_gate * target_air_time_score
    joint_pos = data.qpos[self._qposadr_jp]
    baseline = info["knee_tj_baseline"]
    knee_l = joint_pos[self._knee_l_idx]
    knee_r = joint_pos[self._knee_r_idx]
    knee_retraction_l = jp.maximum(knee_l - baseline[0], 0.0)
    knee_retraction_r = jp.maximum(knee_r - baseline[1], 0.0)
    target_knee_retraction_raw = (
        target_left * knee_retraction_l + target_right * knee_retraction_r
    )
    target_knee_retraction_score = jp.clip(
        target_knee_retraction_raw / self._config.target_retraction, 0.0, 1.0
    )
    foot_motion = self._foot_motion(data, info)
    left_vel_xy = foot_motion["left_vel_xy_norm"]
    right_vel_xy = foot_motion["right_vel_xy_norm"]
    target_foot_vel_xy = target_left * left_vel_xy + target_right * right_vel_xy
    support_foot_vel_xy = support_left * left_vel_xy + support_right * right_vel_xy
    return {
        "left_contact": left_contact,
        "right_contact": right_contact,
        "target_contact": target_contact,
        "support_contact": support_contact,
        "target_no_contact": target_no_contact,
        "support_lost": support_lost,
        "unload_gate": unload_gate,
        "weight_shift": weight_shift,
        "weight_shift_error": weight_shift_error,
        "support_y": support_y,
        "root_y": root_y,
        "left_foot_y": left_pos[1],
        "right_foot_y": right_pos[1],
        "target_foot_z": target_foot_z,
        "target_baseline_z": target_baseline_z,
        "target_clearance_raw": target_clearance_raw,
        "target_clearance_score": target_clearance_score,
        "target_clearance_reward": target_clearance_reward,
        "target_air_time_next": target_air_time_next,
        "target_air_time_score": target_air_time_score,
        "target_air_time_reward": target_air_time_reward,
        "target_no_contact_reward": unload_gate * target_no_contact * support_contact,
        "knee_retraction_l": knee_retraction_l,
        "knee_retraction_r": knee_retraction_r,
        "target_knee_retraction_raw": target_knee_retraction_raw,
        "target_knee_retraction_score": target_knee_retraction_score,
        "target_contact_penalty": unload_gate * target_contact,
        "support_loss_penalty": support_lost,
        "target_foot_vel_xy": target_foot_vel_xy,
        "support_foot_vel_xy": support_foot_vel_xy,
        "swing_drag_penalty": unload_gate * target_contact * target_foot_vel_xy,
        "support_slip_penalty": support_contact * support_foot_vel_xy,
    }


def FootUnloadV2__foot_unload_terms(
    self, data: mjx.Data, info: dict[str, Any]
) -> dict[str, jax.Array]:
    terms = self._invoke_source("FootUnload._foot_unload_terms", data, info)
    roll, pitch = self._roll_pitch(data)
    root_height = self.get_root_pos(data)[2]
    safe = (
        (root_height >= self._config.min_root_height)
        & (jp.abs(roll) <= self._config.max_roll)
        & (jp.abs(pitch) <= self._config.max_pitch)
    )
    safe = jp.float32(safe)
    target_left = info["target_left_swing"]
    target_right = info["target_right_swing"]
    joint_pos = data.qpos[self._qposadr_jp]
    target_hip_aa_qpos = (
        target_left * joint_pos[self._hip_aa_l_idx]
        + target_right * joint_pos[self._hip_aa_r_idx]
    )
    target_ankle_aa_qpos = (
        target_left * joint_pos[self._ankle_aa_l_idx]
        + target_right * joint_pos[self._ankle_aa_r_idx]
    )
    target_knee_tj_qpos = (
        target_left * joint_pos[self._knee_l_idx]
        + target_right * joint_pos[self._knee_r_idx]
    )
    desired_hip_aa = (
        target_left * self._config.left_target_hip_aa_ref
        + target_right * self._config.right_target_hip_aa_ref
    )
    desired_ankle_aa = self._config.target_ankle_aa_ref
    hip_sign_progress = (
        target_left * target_hip_aa_qpos + target_right * -target_hip_aa_qpos
    )
    hip_sign_score = jp.clip(
        hip_sign_progress / jp.abs(self._config.left_target_hip_aa_ref), 0.0, 1.0
    )
    ankle_sign_score = jp.clip(
        -target_ankle_aa_qpos / jp.abs(desired_ankle_aa), 0.0, 1.0
    )
    lateral_unload_sign = 0.5 * (hip_sign_score + ankle_sign_score)
    target_hip_aa_reference = jp.exp(
        -0.5
        * jp.square(
            (target_hip_aa_qpos - desired_hip_aa) / self._config.lateral_ref_sigma
        )
    )
    target_ankle_aa_reference = jp.exp(
        -0.5
        * jp.square(
            (target_ankle_aa_qpos - desired_ankle_aa) / self._config.lateral_ref_sigma
        )
    )
    support_slip_ok = jp.float32(
        terms["support_foot_vel_xy"] <= self._config.support_slip_threshold
    )
    clearance_positive = jp.float32(terms["target_clearance_raw"] > 0.0)
    valid_unload = (
        terms["unload_gate"]
        * terms["target_no_contact"]
        * terms["support_contact"]
        * safe
        * clearance_positive
        * support_slip_ok
    )
    target_clearance_reward = valid_unload * terms["target_clearance_score"]
    target_no_contact_reward = (
        terms["unload_gate"]
        * terms["target_no_contact"]
        * terms["support_contact"]
        * safe
        * support_slip_ok
    )
    target_air_time_reward = (
        terms["unload_gate"]
        * terms["target_air_time_score"]
        * terms["support_contact"]
        * safe
        * support_slip_ok
    )
    target_knee_tj_retraction = (
        lateral_unload_sign * terms["target_knee_retraction_score"]
    )
    terms.update(
        {
            "safe_roll_pitch": safe,
            "valid_unload": valid_unload,
            "target_clearance_reward": target_clearance_reward,
            "target_no_contact_reward": target_no_contact_reward,
            "target_air_time_reward": target_air_time_reward,
            "target_swing_drag_penalty": terms["swing_drag_penalty"],
            "fall_penalty": 1.0 - safe,
            "target_hip_aa_qpos": target_hip_aa_qpos,
            "target_ankle_aa_qpos": target_ankle_aa_qpos,
            "target_knee_tj_qpos": target_knee_tj_qpos,
            "target_hip_aa_desired": desired_hip_aa,
            "target_ankle_aa_desired": desired_ankle_aa,
            "hip_aa_sign_score": hip_sign_score,
            "ankle_aa_sign_score": ankle_sign_score,
            "lateral_unload_sign": lateral_unload_sign,
            "target_hip_aa_reference": target_hip_aa_reference,
            "target_ankle_aa_reference": target_ankle_aa_reference,
            "target_knee_tj_retraction": target_knee_tj_retraction,
            "support_slip_ok": support_slip_ok,
        }
    )
    return terms


def ResidualForwardWalkingV6__forward_terms(
    self,
    data: mjx.Data,
    info: dict[str, Any],
    phase_id: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    components = self._v6_components(data, info, phase_id, terms, foot_motion)
    return {
        "vx_cmd": components["vx_cmd"],
        "vx_actual": components["vx_actual"],
        "vx_abs_error": jp.abs(components["vx_actual"] - components["vx_cmd"]),
        "root_dx": components["root_dx"],
        "root_displacement": components["root_displacement"],
        "forward_velocity_tracking": components["forward_velocity_progress"],
        "forward_displacement_progress": components["forward_displacement_progress"],
        "left_swing_forward_progress": jp.zeros(()),
        "right_swing_forward_progress": jp.zeros(()),
        "swing_foot_forward_progress": components["airborne_forward_progress"],
        "left_swing_drag": jp.zeros(()),
        "right_swing_drag": jp.zeros(()),
        "swing_drag_penalty": jp.zeros(()),
        "left_stance_slip": jp.zeros(()),
        "right_stance_slip": jp.zeros(()),
        "lateral_drift_penalty": components["lateral_drift_penalty"],
        "yaw_drift_penalty": components["yaw_drift_penalty"],
        "left_touchdown": components["left_touchdown"],
        "right_touchdown": components["right_touchdown"],
        "left_touchdown_count": info["left_touchdown_count"]
        + components["left_touchdown"],
        "right_touchdown_count": info["right_touchdown_count"]
        + components["right_touchdown"],
        "left_touchdown_vz": jp.zeros(()),
        "right_touchdown_vz": jp.zeros(()),
        "current_left_no_contact": components["current_left_no_contact"],
        "current_right_no_contact": components["current_right_no_contact"],
        "max_left_no_contact": jp.maximum(
            info["max_consecutive_left_no_contact"],
            components["current_left_no_contact"],
        ),
        "max_right_no_contact": jp.maximum(
            info["max_consecutive_right_no_contact"],
            components["current_right_no_contact"],
        ),
    }


def ResidualForwardWalkingV8__get_reward(
    self,
    data: mjx.Data,
    clipped_action: jax.Array,
    info: dict[str, Any],
    phase_id: jax.Array,
    unload_weight: jax.Array,
    q_ref: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
    swing_contact_count: jax.Array,
) -> dict[str, jax.Array]:
    del clipped_action, unload_weight, q_ref, swing_contact_count
    components = self._v8_components(data, info, phase_id, terms, foot_motion)
    return {name: components[name] for name in self._REWARD_TERMS + self._PENALTY_TERMS}
