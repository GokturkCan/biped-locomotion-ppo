# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0. See LICENSE.
# Adapted from the private research working tree based on MuJoCo Playground.
# Robot descriptions and model parameters are intentionally excluded.

from __future__ import annotations
from typing import Any
import jax
import jax.numpy as jp
from mujoco import mjx


def ResidualForwardWalkingV1__forward_yaw_from_quat(quat: jax.Array) -> jax.Array:
    qw, qx, qy, qz = quat
    return jp.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def ResidualForwardWalkingV8__v8_liftoff_touchdown_state(
    self,
    side: str,
    contact: jax.Array,
    opposite_contact: jax.Array,
    height_gain: jax.Array,
    prev_contact: jax.Array,
    valid_ground_load_gate: jax.Array,
    info: dict[str, Any],
) -> dict[str, jax.Array]:
    contact_f = jp.float32(contact)
    no_contact_f = 1.0 - contact_f
    raw_liftoff = jp.float32((prev_contact > 0.5) & (contact_f < 0.5))
    raw_touchdown = jp.float32((prev_contact < 0.5) & (contact_f > 0.5))
    no_contact_streak = jp.where(
        contact_f > 0.5, 0.0, info[f"v8_consecutive_no_contact_steps_{side}"] + 1.0
    )
    previous_max = info[f"v8_max_height_gain_during_air_{side}"]
    prev_air_qualified = info[f"v8_air_qualified_{side}"]
    max_air_height = jp.where(
        contact_f < 0.5,
        jp.maximum(previous_max, height_gain),
        jp.where(prev_air_qualified > 0.5, previous_max, 0.0),
    )
    contact_streak = jp.where(
        contact_f > 0.5, info[f"v8_consecutive_contact_steps_{side}"] + 1.0, 0.0
    )
    qualified_liftoff = (
        jp.float32(prev_air_qualified < 0.5)
        * jp.float32(no_contact_streak >= self._config.v8_liftoff_no_contact_steps)
        * jp.float32(max_air_height >= self._config.v8_liftoff_height_gain)
        * jp.float32(opposite_contact)
        * jp.float32(valid_ground_load_gate > 0.999)
    )
    air_qualified = jp.maximum(prev_air_qualified, qualified_liftoff)
    steps_since = info[f"v8_steps_since_last_qualified_touchdown_{side}"]
    qualified_touchdown = (
        air_qualified
        * jp.float32(contact_streak == self._config.v8_touchdown_contact_confirm_steps)
        * jp.float32(
            steps_since >= self._config.v8_min_steps_between_qualified_touchdowns
        )
    )
    next_air_qualified = jp.where(qualified_touchdown > 0.5, 0.0, air_qualified)
    next_max_air_height = jp.where(qualified_touchdown > 0.5, 0.0, max_air_height)
    next_steps_since = jp.where(qualified_touchdown > 0.5, 0.0, steps_since + 1.0)
    return {
        "raw_liftoff": raw_liftoff,
        "raw_touchdown": raw_touchdown,
        "qualified_liftoff": qualified_liftoff,
        "qualified_touchdown": qualified_touchdown,
        "no_contact_streak": no_contact_streak,
        "contact_streak": contact_streak,
        "max_air_height": max_air_height,
        "next_no_contact_streak": no_contact_streak,
        "next_max_air_height": next_max_air_height,
        "next_air_qualified": next_air_qualified,
        "next_contact_streak": contact_streak,
        "next_steps_since": next_steps_since,
    }


def ResidualForwardWalkingV9__v9_expected_side_gates(
    self, info: dict[str, Any]
) -> tuple[jax.Array, jax.Array]:
    expected_side = info["v9_expected_side"]
    transfer_open = 1.0 - jp.float32(info["v9_transfer_pending"] > 0.5)
    unknown = jp.float32(jp.abs(expected_side) < 0.5)
    gate_left = transfer_open * jp.maximum(unknown, jp.float32(expected_side < -0.5))
    gate_right = transfer_open * jp.maximum(unknown, jp.float32(expected_side > 0.5))
    return (gate_left, gate_right)


def ResidualForwardWalkingV9__v9_positive_step_contributions(
    self,
    data: mjx.Data,
    info: dict[str, Any],
    phase_id: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
    components: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    target_left = info["target_left_swing"]
    target_right = info["target_right_swing"]
    left_contact = jp.float32(foot_motion["left_contact"])
    right_contact = jp.float32(foot_motion["right_contact"])
    left_no_contact = 1.0 - left_contact
    right_no_contact = 1.0 - right_contact
    swing_phase_gate = jp.float32((phase_id == 2) | (phase_id == 3))
    valid_ground_load_gate = components["valid_ground_load_gate"]
    left_roll, left_pitch, right_roll, right_pitch = self._foot_roll_pitch(data)
    left_flatness_gate = jp.exp(
        -(jp.square(left_roll) + jp.square(left_pitch)) / jp.square(0.1)
    )
    right_flatness_gate = jp.exp(
        -(jp.square(right_roll) + jp.square(right_pitch)) / jp.square(0.1)
    )
    left_unload_gate = jp.clip((0.5 - components["lambda_left"]) / 0.25, 0.0, 1.0)
    right_unload_gate = jp.clip((0.5 - components["lambda_right"]) / 0.25, 0.0, 1.0)
    left_support_contact = right_contact
    right_support_contact = left_contact
    left_single_support = (
        target_left
        * swing_phase_gate
        * valid_ground_load_gate
        * left_support_contact
        * left_no_contact
    )
    right_single_support = (
        target_right
        * swing_phase_gate
        * valid_ground_load_gate
        * right_support_contact
        * right_no_contact
    )
    left_airborne = (
        target_left * swing_phase_gate * left_support_contact * left_no_contact
    )
    right_airborne = (
        target_right * swing_phase_gate * right_support_contact * right_no_contact
    )
    left_dx = foot_motion["left_pos"][0] - info["prev_left_foot_x"]
    right_dx = foot_motion["right_pos"][0] - info["prev_right_foot_x"]
    return {
        "deep_unload_left": target_left
        * swing_phase_gate
        * left_support_contact
        * valid_ground_load_gate
        * jp.square(jp.clip((0.35 - components["lambda_left"]) / 0.3, 0.0, 1.0)),
        "deep_unload_right": target_right
        * swing_phase_gate
        * right_support_contact
        * valid_ground_load_gate
        * jp.square(jp.clip((0.35 - components["lambda_right"]) / 0.3, 0.0, 1.0)),
        "pre_liftoff_clearance_left": target_left
        * swing_phase_gate
        * left_support_contact
        * valid_ground_load_gate
        * left_flatness_gate
        * left_unload_gate
        * jp.clip(components["left_height_gain"] / 0.02, 0.0, 1.0),
        "pre_liftoff_clearance_right": target_right
        * swing_phase_gate
        * right_support_contact
        * valid_ground_load_gate
        * right_flatness_gate
        * right_unload_gate
        * jp.clip(components["right_height_gain"] / 0.02, 0.0, 1.0),
        "effective_liftoff_left": target_left
        * swing_phase_gate
        * left_support_contact
        * jp.float32(components["lambda_left"] < 0.08)
        * jp.float32(components["left_height_gain"] > 0.003),
        "effective_liftoff_right": target_right
        * swing_phase_gate
        * right_support_contact
        * jp.float32(components["lambda_right"] < 0.08)
        * jp.float32(components["right_height_gain"] > 0.003),
        "swing_no_contact_left": target_left
        * swing_phase_gate
        * left_support_contact
        * valid_ground_load_gate
        * left_no_contact,
        "swing_no_contact_right": target_right
        * swing_phase_gate
        * right_support_contact
        * valid_ground_load_gate
        * right_no_contact,
        "actual_single_support_left": left_single_support,
        "actual_single_support_right": right_single_support,
        "air_time_left": left_single_support * terms["target_air_time_score"],
        "air_time_right": right_single_support * terms["target_air_time_score"],
        "airborne_clearance_left": left_airborne
        * jp.clip(components["left_height_gain"] / 0.03, 0.0, 1.0),
        "airborne_clearance_right": right_airborne
        * jp.clip(components["right_height_gain"] / 0.03, 0.0, 1.0),
        "airborne_forward_progress_left": left_airborne
        * jp.clip(left_dx / 0.015, 0.0, 1.0),
        "airborne_forward_progress_right": right_airborne
        * jp.clip(right_dx / 0.015, 0.0, 1.0),
    }


def Walking__get_termination(self, data: mjx.Data) -> jax.Array:
    flags = self._termination_flags(data)
    fallen = flags["term_height"] | flags["term_roll"] | flags["term_pitch"]
    fallen |= flags["term_nan"]
    return jp.where(self._config.early_termination, fallen, jp.zeros_like(fallen))


def Walking__roll_pitch(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
    quat = self.get_orientation(data)
    qw, qx, qy, qz = quat
    roll = jp.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = jp.arcsin(jp.clip(sinp, -1.0, 1.0))
    return (roll, pitch)


def Walking__termination_flags(self, data: mjx.Data) -> dict[str, jax.Array]:
    roll, pitch = self._roll_pitch(data)
    root_height = self.get_root_pos(data)[2]
    finite = jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel))
    return {
        "term_height": root_height < self._config.min_root_height,
        "term_roll": jp.abs(roll) > self._config.max_roll,
        "term_pitch": jp.abs(pitch) > self._config.max_pitch,
        "term_nan": ~finite,
    }
