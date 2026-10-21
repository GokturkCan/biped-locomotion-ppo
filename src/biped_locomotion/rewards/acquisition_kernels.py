# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0. See LICENSE.
# Adapted from the private research working tree based on MuJoCo Playground.
# Robot descriptions and model parameters are intentionally excluded.

from __future__ import annotations
from typing import Any
import jax
import jax.numpy as jp
from mujoco import mjx


def ResidualForwardWalkingV6__v6_components(
    self,
    data: mjx.Data,
    info: dict[str, Any],
    phase_id: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    target_left = info["target_left_swing"]
    target_right = info["target_right_swing"]
    left_contact = foot_motion["left_contact"]
    right_contact = foot_motion["right_contact"]
    left_pos = foot_motion["left_pos"]
    right_pos = foot_motion["right_pos"]
    fz_left, fz_right = self._foot_vertical_ground_reaction_forces(data)
    total_load = fz_left + fz_right
    epsilon = self._load_transfer_force_epsilon_jp
    total_load_fraction = jp.clip(
        total_load / (self._expected_weight_jp + epsilon), 0.0, 2.0
    )
    lambda_left = jp.clip(fz_left / (total_load + epsilon), 0.0, 1.0)
    lambda_right = jp.clip(fz_right / (total_load + epsilon), 0.0, 1.0)
    lambda_swing = jp.clip(
        target_left * lambda_left + target_right * lambda_right, 0.0, 1.0
    )
    lambda_support = jp.clip(
        target_left * lambda_right + target_right * lambda_left, 0.0, 1.0
    )
    swing_phase_gate = jp.float32((phase_id == 2) | (phase_id == 3))
    support_contact = terms["support_contact"]
    swing_contact = terms["target_contact"]
    swing_no_contact = terms["target_no_contact"]
    valid_ground_load_gate = jp.clip(total_load_fraction / 0.6, 0.0, 1.0)
    left_height_gain = left_pos[2] - info["left_foot_z_nominal"]
    right_height_gain = right_pos[2] - info["right_foot_z_nominal"]
    height_gain = target_left * left_height_gain + target_right * right_height_gain
    left_dx = left_pos[0] - info["prev_left_foot_x"]
    right_dx = right_pos[0] - info["prev_right_foot_x"]
    swing_dx = target_left * left_dx + target_right * right_dx
    left_roll, left_pitch, right_roll, right_pitch = self._foot_roll_pitch(data)
    swing_roll = target_left * left_roll + target_right * right_roll
    swing_pitch = target_left * left_pitch + target_right * right_pitch
    flatness_gate = jp.exp(
        -(jp.square(swing_roll) + jp.square(swing_pitch)) / jp.square(0.1)
    )
    unload_gate = jp.clip((0.5 - lambda_swing) / 0.25, 0.0, 1.0)
    continuous_swing_load = (
        swing_phase_gate * valid_ground_load_gate * jp.square(lambda_swing)
    )
    deep_unload = (
        swing_phase_gate
        * support_contact
        * valid_ground_load_gate
        * jp.square(jp.clip((0.35 - lambda_swing) / 0.3, 0.0, 1.0))
    )
    pre_liftoff_clearance = (
        swing_phase_gate
        * support_contact
        * valid_ground_load_gate
        * flatness_gate
        * unload_gate
        * jp.clip(height_gain / 0.02, 0.0, 1.0)
    )
    effective_liftoff = (
        swing_phase_gate
        * support_contact
        * jp.float32(lambda_swing < 0.08)
        * jp.float32(height_gain > 0.003)
    )
    actual_single_support_boolean = (
        jp.float32(left_contact ^ right_contact) * support_contact * swing_no_contact
    )
    loaded_single_support = (
        swing_phase_gate * valid_ground_load_gate * actual_single_support_boolean
    )
    swing_no_contact_reward = (
        swing_phase_gate * support_contact * valid_ground_load_gate * swing_no_contact
    )
    actual_single_support = loaded_single_support
    air_time = loaded_single_support * terms["target_air_time_score"]
    double_support_penalty = swing_phase_gate * support_contact * swing_contact
    support_loss_penalty = swing_phase_gate * (1.0 - support_contact)
    ground_load_loss_penalty = swing_phase_gate * jp.clip(
        (0.6 - total_load_fraction) / 0.6, 0.0, 1.0
    )
    airborne_clearance = (
        swing_phase_gate
        * support_contact
        * swing_no_contact
        * jp.clip(height_gain / 0.03, 0.0, 1.0)
    )
    airborne_forward_progress = (
        swing_phase_gate
        * support_contact
        * swing_no_contact
        * jp.clip(swing_dx / 0.015, 0.0, 1.0)
    )
    prev_swing_no_contact = target_left * (
        1.0 - info["prev_left_contact"]
    ) + target_right * (1.0 - info["prev_right_contact"])
    touchdown_event = (
        swing_phase_gate * prev_swing_no_contact * swing_contact * support_contact
    )
    swing_x = target_left * left_pos[0] + target_right * right_pos[0]
    support_x = target_left * right_pos[0] + target_right * left_pos[0]
    forward_touchdown_score = jp.clip((swing_x - support_x) / 0.06, 0.0, 1.0)
    forward_touchdown = touchdown_event * forward_touchdown_score
    recent_true_touchdown = jp.maximum(
        touchdown_event, jp.float32(info["recent_true_touchdown_steps"] > 0.0)
    )
    locomotion_gate = jp.maximum(actual_single_support, recent_true_touchdown)
    vx_cmd = jp.maximum(info["command"][0], 1e-06)
    vx_actual = self.get_local_linvel(data)[0]
    velocity_ratio = jp.clip(vx_actual / vx_cmd, 0.0, 1.0)
    velocity_score = velocity_ratio * jp.exp(-jp.square((vx_actual - vx_cmd) / 0.05))
    forward_velocity_progress = locomotion_gate * velocity_score
    root_pos = self.get_root_pos(data)
    root_dx = root_pos[0] - info["prev_root_x"]
    forward_displacement_progress = locomotion_gate * jp.clip(
        root_dx / jp.maximum(self._config.forward_displacement_progress_cap, 1e-06),
        0.0,
        1.0,
    )
    upvector = self.get_gravity(data)
    joint_pos = data.qpos[self._qposadr_jp]
    soft_lower_violation = jp.maximum(self._soft_joint_lowers - joint_pos, 0.0)
    soft_upper_violation = jp.maximum(joint_pos - self._soft_joint_uppers, 0.0)
    joint_limit = jp.sum(soft_lower_violation + soft_upper_violation)
    flags = self._termination_flags(data)
    fall = jp.float32(flags["term_height"] | flags["term_roll"] | flags["term_pitch"])
    lateral_error = jp.abs(root_pos[1] - info["root_y_baseline"])
    lateral_drift_penalty = jp.clip(
        (lateral_error - self._config.forward_lateral_drift_deadzone)
        / jp.maximum(self._config.forward_lateral_drift_scale, 1e-06),
        0.0,
        1.0,
    )
    yaw = self._forward_yaw_from_quat(self.get_orientation(data))
    yaw_rate = self.get_gyro(data)[2]
    yaw_drift_penalty = jp.clip(
        (jp.abs(yaw) + 0.1 * jp.abs(yaw_rate))
        / jp.maximum(self._config.forward_yaw_drift_scale, 1e-06),
        0.0,
        1.0,
    )
    left_touchdown = jp.float32((info["prev_left_contact"] < 0.5) & left_contact)
    right_touchdown = jp.float32((info["prev_right_contact"] < 0.5) & right_contact)
    current_left_no_contact = jp.where(
        left_contact, 0.0, info["current_consecutive_left_no_contact"] + 1.0
    )
    current_right_no_contact = jp.where(
        right_contact, 0.0, info["current_consecutive_right_no_contact"] + 1.0
    )
    return {
        "alive": jp.ones(()),
        "upright": jp.clip(upvector[2], 0.0, 1.0),
        "root_height": jp.exp(
            -jp.square(root_pos[2] - self._config.target_height)
            / self._config.reward_config.height_sigma
        ),
        "continuous_swing_load": continuous_swing_load,
        "deep_unload": deep_unload,
        "pre_liftoff_clearance": pre_liftoff_clearance,
        "effective_liftoff": effective_liftoff,
        "swing_no_contact": swing_no_contact_reward,
        "actual_single_support": actual_single_support,
        "air_time": air_time,
        "double_support_penalty": double_support_penalty,
        "support_loss_penalty": support_loss_penalty,
        "ground_load_loss_penalty": ground_load_loss_penalty,
        "airborne_clearance": airborne_clearance,
        "airborne_forward_progress": airborne_forward_progress,
        "forward_touchdown": forward_touchdown,
        "forward_velocity_progress": forward_velocity_progress,
        "forward_displacement_progress": forward_displacement_progress,
        "joint_limit": joint_limit,
        "fall": fall,
        "lateral_drift_penalty": lateral_drift_penalty,
        "yaw_drift_penalty": yaw_drift_penalty,
        "fz_left": fz_left,
        "fz_right": fz_right,
        "total_load": total_load,
        "total_load_fraction": total_load_fraction,
        "valid_ground_load_gate": valid_ground_load_gate,
        "lambda_left": lambda_left,
        "lambda_right": lambda_right,
        "lambda_swing": lambda_swing,
        "lambda_support": lambda_support,
        "height_gain": height_gain,
        "left_height_gain": left_height_gain,
        "right_height_gain": right_height_gain,
        "flatness_gate": flatness_gate,
        "unload_gate": unload_gate,
        "swing_dx": swing_dx,
        "touchdown_event": touchdown_event,
        "recent_true_touchdown": recent_true_touchdown,
        "forward_touchdown_score": forward_touchdown_score,
        "locomotion_gate": locomotion_gate,
        "velocity_score": velocity_score,
        "vx_actual": vx_actual,
        "vx_cmd": info["command"][0],
        "root_dx": root_dx,
        "root_displacement": root_pos[0] - info["root_x_baseline"],
        "left_touchdown": left_touchdown,
        "right_touchdown": right_touchdown,
        "current_left_no_contact": current_left_no_contact,
        "current_right_no_contact": current_right_no_contact,
        "left_actual_single_support": jp.float32(left_contact & ~right_contact),
        "right_actual_single_support": jp.float32(right_contact & ~left_contact),
    }


def ResidualForwardWalkingV7__v7_components(
    self,
    data: mjx.Data,
    info: dict[str, Any],
    phase_id: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    components = self._v6_components(data, info, phase_id, terms, foot_motion)
    roll, pitch = self._roll_pitch(data)
    gyro = self.get_gyro(data)
    gyro_xy_norm = jp.sqrt(jp.square(gyro[0]) + jp.square(gyro[1]))
    upright_gate = jp.exp(-(jp.square(roll) + jp.square(pitch)) / jp.square(0.25))
    root_height_gate = components["root_height"]
    angular_stability_gate = jp.exp(
        -(jp.square(gyro[0]) + jp.square(gyro[1])) / jp.square(1.5)
    )
    stability_gate = upright_gate * root_height_gate * angular_stability_gate
    actual_single_support = components["actual_single_support"]
    recent_true_touchdown = components["recent_true_touchdown"]
    stable_single_support = actual_single_support * stability_gate
    post_touchdown_stability = recent_true_touchdown * stability_gate
    left_touchdown = components["left_touchdown"]
    right_touchdown = components["right_touchdown"]
    touchdown_event = jp.maximum(left_touchdown, right_touchdown)
    current_touchdown_side = right_touchdown - left_touchdown
    previous_touchdown_side = info["last_true_touchdown_side"]
    landed_side = jp.where(
        touchdown_event > 0.5, current_touchdown_side, info["landed_touchdown_side"]
    )
    landed_left = jp.float32(landed_side < -0.5)
    landed_right = jp.float32(landed_side > 0.5)
    landed_load_fraction = (
        landed_left * components["lambda_left"]
        + landed_right * components["lambda_right"]
    )
    landed_contact = landed_left * jp.float32(
        foot_motion["left_contact"]
    ) + landed_right * jp.float32(foot_motion["right_contact"])
    load_acceptance_gate = jp.clip((landed_load_fraction - 0.15) / 0.35, 0.0, 1.0)
    touchdown_load_acceptance = (
        recent_true_touchdown
        * components["valid_ground_load_gate"]
        * landed_contact
        * upright_gate
        * load_acceptance_gate
    )
    left_touchdown_load_acceptance = touchdown_load_acceptance * landed_left
    right_touchdown_load_acceptance = touchdown_load_acceptance * landed_right
    valid_new_touchdown = touchdown_event * jp.float32(
        jp.abs(current_touchdown_side) > 0.5
    )
    has_previous_touchdown = jp.float32(jp.abs(previous_touchdown_side) > 0.5)
    alternating_touchdown = (
        valid_new_touchdown
        * has_previous_touchdown
        * jp.float32(current_touchdown_side == -previous_touchdown_side)
    )
    same_side_touchdown_repeat = (
        valid_new_touchdown
        * has_previous_touchdown
        * jp.float32(current_touchdown_side == previous_touchdown_side)
    )
    locomotion_gate = jp.maximum(actual_single_support, recent_true_touchdown)
    locomotion_instability = locomotion_gate * jp.clip(
        (jp.square(roll) + jp.square(pitch)) / jp.square(0.35)
        + (jp.square(gyro[0]) + jp.square(gyro[1])) / jp.square(2.0),
        0.0,
        1.0,
    )
    flags = self._termination_flags(data)
    fall_event = jp.float32(
        flags["term_height"]
        | flags["term_roll"]
        | flags["term_pitch"]
        | flags["term_nan"]
    )
    terminal_event = fall_event * jp.float32(self._config.early_termination)
    components.update(
        {
            "stable_single_support": stable_single_support,
            "post_touchdown_stability": post_touchdown_stability,
            "touchdown_load_acceptance": touchdown_load_acceptance,
            "left_touchdown_load_acceptance": left_touchdown_load_acceptance,
            "right_touchdown_load_acceptance": right_touchdown_load_acceptance,
            "alternating_touchdown": alternating_touchdown,
            "same_side_touchdown_repeat": same_side_touchdown_repeat,
            "locomotion_instability": locomotion_instability,
            "v7_upright_gate": upright_gate,
            "v7_root_height_gate": root_height_gate,
            "v7_angular_stability_gate": angular_stability_gate,
            "last_touchdown_side": jp.where(
                touchdown_event > 0.5, current_touchdown_side, previous_touchdown_side
            ),
            "roll": roll,
            "pitch": pitch,
            "gyro_xy_norm": gyro_xy_norm,
            "termination/fall_event": terminal_event,
            "termination/term_height": terminal_event
            * jp.float32(flags["term_height"]),
            "termination/term_roll": terminal_event * jp.float32(flags["term_roll"]),
            "termination/term_pitch": terminal_event * jp.float32(flags["term_pitch"]),
            "termination/terminal_root_height": terminal_event
            * self.get_root_pos(data)[2],
            "termination/terminal_abs_roll": terminal_event * jp.abs(roll),
            "termination/terminal_abs_pitch": terminal_event * jp.abs(pitch),
            "termination/terminal_roll_pitch_angular_velocity": terminal_event
            * gyro_xy_norm,
        }
    )
    return components


def ResidualForwardWalkingV8__v8_components(
    self,
    data: mjx.Data,
    info: dict[str, Any],
    phase_id: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    components = self._v7_components(data, info, phase_id, terms, foot_motion)
    left_contact = foot_motion["left_contact"]
    right_contact = foot_motion["right_contact"]
    left = self._v8_liftoff_touchdown_state(
        "left",
        left_contact,
        right_contact,
        components["left_height_gain"],
        info["prev_left_contact"],
        components["valid_ground_load_gate"],
        info,
    )
    right = self._v8_liftoff_touchdown_state(
        "right",
        right_contact,
        left_contact,
        components["right_height_gain"],
        info["prev_right_contact"],
        components["valid_ground_load_gate"],
        info,
    )
    qualified_touchdown = jp.maximum(
        left["qualified_touchdown"], right["qualified_touchdown"]
    )
    current_side = right["qualified_touchdown"] - left["qualified_touchdown"]
    previous_side = info["v8_last_qualified_touchdown_side"]
    has_previous_side = jp.float32(jp.abs(previous_side) > 0.5)
    valid_current_side = qualified_touchdown * jp.float32(jp.abs(current_side) > 0.5)
    first_qualified_touchdown = valid_current_side * (1.0 - has_previous_side)
    qualified_alternating_touchdown = (
        valid_current_side
        * has_previous_side
        * jp.float32(current_side == -previous_side)
    )
    qualified_same_side_repeat = (
        valid_current_side
        * has_previous_side
        * jp.float32(current_side == previous_side)
    )
    eligible_touchdown = jp.maximum(
        first_qualified_touchdown, qualified_alternating_touchdown
    )
    recent_eligible_touchdown = jp.maximum(
        eligible_touchdown, jp.float32(info["v8_recent_eligible_touchdown_steps"] > 0.0)
    )
    next_recent_eligible_steps = jp.where(
        eligible_touchdown > 0.5,
        jp.float32(self._config.v8_recent_eligible_touchdown_steps - 1),
        jp.maximum(info["v8_recent_eligible_touchdown_steps"] - 1.0, 0.0),
    )
    roll = components["roll"]
    pitch = components["pitch"]
    gyro = self.get_gyro(data)
    upright_gate = components["v7_upright_gate"]
    root_height_gate = components["v7_root_height_gate"]
    angular_stability_gate = components["v7_angular_stability_gate"]
    stability_gate = upright_gate * root_height_gate * angular_stability_gate
    landed_side = jp.where(
        eligible_touchdown > 0.5, current_side, info["v8_support_transfer_landed_side"]
    )
    previous_support_side = jp.where(
        eligible_touchdown > 0.5,
        -current_side,
        info["v8_support_transfer_previous_support_side"],
    )
    landed_left = jp.float32(landed_side < -0.5)
    landed_right = jp.float32(landed_side > 0.5)
    previous_support_left = jp.float32(previous_support_side < -0.5)
    previous_support_right = jp.float32(previous_support_side > 0.5)
    landed_load_fraction = (
        landed_left * components["lambda_left"]
        + landed_right * components["lambda_right"]
    )
    previous_support_load_fraction = (
        previous_support_left * components["lambda_left"]
        + previous_support_right * components["lambda_right"]
    )
    landed_contact = landed_left * jp.float32(left_contact) + landed_right * jp.float32(
        right_contact
    )
    landed_load_acceptance_gate = jp.clip(
        (landed_load_fraction - 0.15) / 0.35, 0.0, 1.0
    )
    touchdown_load_acceptance = (
        recent_eligible_touchdown
        * components["valid_ground_load_gate"]
        * landed_contact
        * upright_gate
        * landed_load_acceptance_gate
    )
    left_landed_load_acceptance = touchdown_load_acceptance * landed_left
    right_landed_load_acceptance = touchdown_load_acceptance * landed_right
    current_transfer_steps = jp.where(
        eligible_touchdown > 0.5,
        jp.float32(self._config.v8_support_transfer_steps),
        info["v8_support_transfer_window_steps"],
    )
    transfer_completed_prev = jp.where(
        eligible_touchdown > 0.5, 0.0, info["v8_support_transfer_completed"]
    )
    transfer_timeout_prev = jp.where(
        eligible_touchdown > 0.5, 0.0, info["v8_support_transfer_timeout_fired"]
    )
    active_transfer_window = jp.float32(current_transfer_steps > 0.0)
    landed_load_gate = jp.clip((landed_load_fraction - 0.25) / 0.3, 0.0, 1.0)
    previous_support_release_gate = jp.clip(
        (0.65 - previous_support_load_fraction) / 0.2, 0.0, 1.0
    )
    support_transfer_progress = (
        active_transfer_window
        * stability_gate
        * landed_load_gate
        * previous_support_release_gate
    )
    support_transfer_completion = (
        active_transfer_window
        * (1.0 - transfer_completed_prev)
        * jp.float32(landed_load_fraction >= 0.55)
        * jp.float32(previous_support_load_fraction <= 0.45)
        * landed_contact
        * jp.float32(components["valid_ground_load_gate"] > 0.999)
        * jp.float32(upright_gate >= 0.7)
        * jp.float32(root_height_gate >= 0.7)
        * jp.float32(angular_stability_gate >= 0.5)
    )
    support_transfer_timeout = (
        (1.0 - eligible_touchdown)
        * active_transfer_window
        * jp.float32(current_transfer_steps <= 1.0)
        * (1.0 - transfer_completed_prev)
        * (1.0 - transfer_timeout_prev)
        * (1.0 - support_transfer_completion)
    )
    next_transfer_window_steps = jp.where(
        eligible_touchdown > 0.5,
        jp.float32(self._config.v8_support_transfer_steps - 1),
        jp.where(
            support_transfer_timeout > 0.5,
            0.0,
            jp.maximum(info["v8_support_transfer_window_steps"] - 1.0, 0.0),
        ),
    )
    next_transfer_completed = jp.maximum(
        transfer_completed_prev, support_transfer_completion
    )
    next_timeout_fired = jp.maximum(transfer_timeout_prev, support_transfer_timeout)
    next_last_side = jp.where(valid_current_side > 0.5, current_side, previous_side)
    components.update(
        {
            "air_time": eligible_touchdown,
            "forward_touchdown": eligible_touchdown
            * components["forward_touchdown_score"],
            "alternating_touchdown": qualified_alternating_touchdown,
            "qualified_same_side_repeat": qualified_same_side_repeat,
            "post_touchdown_stability": recent_eligible_touchdown * stability_gate,
            "touchdown_load_acceptance": touchdown_load_acceptance,
            "left_touchdown_load_acceptance": left_landed_load_acceptance,
            "right_touchdown_load_acceptance": right_landed_load_acceptance,
            "support_transfer_progress": support_transfer_progress,
            "support_transfer_completion": support_transfer_completion,
            "support_transfer_timeout": support_transfer_timeout,
            "locomotion_instability": jp.maximum(
                components["actual_single_support"], recent_eligible_touchdown
            )
            * jp.clip(
                (jp.square(roll) + jp.square(pitch)) / jp.square(0.35)
                + (jp.square(gyro[0]) + jp.square(gyro[1])) / jp.square(2.0),
                0.0,
                1.0,
            ),
            "raw_liftoff_left": left["raw_liftoff"],
            "raw_liftoff_right": right["raw_liftoff"],
            "raw_touchdown_left": left["raw_touchdown"],
            "raw_touchdown_right": right["raw_touchdown"],
            "qualified_liftoff_left": left["qualified_liftoff"],
            "qualified_liftoff_right": right["qualified_liftoff"],
            "qualified_touchdown_left": left["qualified_touchdown"],
            "qualified_touchdown_right": right["qualified_touchdown"],
            "qualified_touchdown": qualified_touchdown,
            "eligible_touchdown": eligible_touchdown,
            "recent_eligible_touchdown": recent_eligible_touchdown,
            "qualified_alternating_touchdown": qualified_alternating_touchdown,
            "qualified_same_side_repeat_count": qualified_same_side_repeat,
            "last_qualified_touchdown_side": next_last_side,
            "landed_load_fraction": landed_load_fraction,
            "previous_support_load_fraction": previous_support_load_fraction,
            "left_transfer_completion": support_transfer_completion * landed_left,
            "right_transfer_completion": support_transfer_completion * landed_right,
            "next_last_qualified_touchdown_side": next_last_side,
            "next_recent_eligible_touchdown_steps": next_recent_eligible_steps,
            "next_support_transfer_window_steps": next_transfer_window_steps,
            "next_support_transfer_landed_side": landed_side,
            "next_support_transfer_previous_support_side": previous_support_side,
            "next_support_transfer_completed": next_transfer_completed,
            "next_support_transfer_timeout_fired": next_timeout_fired,
        }
    )
    for side, state in (("left", left), ("right", right)):
        components[f"no_contact_streak_{side}"] = state["no_contact_streak"]
        components[f"contact_streak_{side}"] = state["contact_streak"]
        components[f"max_air_height_gain_{side}"] = state["max_air_height"]
        components[f"next_no_contact_streak_{side}"] = state["next_no_contact_streak"]
        components[f"next_max_air_height_gain_{side}"] = state["next_max_air_height"]
        components[f"next_air_qualified_{side}"] = state["next_air_qualified"]
        components[f"next_contact_streak_{side}"] = state["next_contact_streak"]
        components[f"next_steps_since_qualified_touchdown_{side}"] = state[
            "next_steps_since"
        ]
    return components


def ResidualForwardWalkingV9__v8_components(
    self,
    data: mjx.Data,
    info: dict[str, Any],
    phase_id: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    return self._v9_components(data, info, phase_id, terms, foot_motion)


def ResidualForwardWalkingV9__v9_components(
    self,
    data: mjx.Data,
    info: dict[str, Any],
    phase_id: jax.Array,
    terms: dict[str, jax.Array],
    foot_motion: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    components = self._invoke_source(
        "ResidualForwardWalkingV8._v8_components",
        data,
        info,
        phase_id,
        terms,
        foot_motion,
    )
    gate_left, gate_right = self._v9_expected_side_gates(info)
    positive = self._v9_positive_step_contributions(
        data, info, phase_id, terms, foot_motion, components
    )

    def gated(name: str) -> jax.Array:
        return (
            gate_left * positive[f"{name}_left"]
            + gate_right * positive[f"{name}_right"]
        )

    qualified_liftoff_left = components["qualified_liftoff_left"]
    qualified_liftoff_right = components["qualified_liftoff_right"]
    qualified_touchdown_left = components["qualified_touchdown_left"]
    qualified_touchdown_right = components["qualified_touchdown_right"]
    expected_qualified_liftoff = (
        gate_left * qualified_liftoff_left + gate_right * qualified_liftoff_right
    )
    unexpected_qualified_liftoff = (1.0 - gate_left) * qualified_liftoff_left + (
        1.0 - gate_right
    ) * qualified_liftoff_right
    expected_qualified_touchdown = (
        gate_left * qualified_touchdown_left + gate_right * qualified_touchdown_right
    )
    unexpected_qualified_touchdown = (1.0 - gate_left) * qualified_touchdown_left + (
        1.0 - gate_right
    ) * qualified_touchdown_right
    raw_eligible_touchdown = components["eligible_touchdown"]
    eligible_touchdown_left = raw_eligible_touchdown * qualified_touchdown_left
    eligible_touchdown_right = raw_eligible_touchdown * qualified_touchdown_right
    eligible_touchdown = (
        gate_left * eligible_touchdown_left + gate_right * eligible_touchdown_right
    )
    current_side = qualified_touchdown_right - qualified_touchdown_left
    recent_eligible_touchdown = jp.maximum(
        eligible_touchdown, jp.float32(info["v8_recent_eligible_touchdown_steps"] > 0.0)
    )
    next_recent_eligible_steps = jp.where(
        eligible_touchdown > 0.5,
        jp.float32(self._config.v8_recent_eligible_touchdown_steps - 1),
        jp.maximum(info["v8_recent_eligible_touchdown_steps"] - 1.0, 0.0),
    )
    roll = components["roll"]
    pitch = components["pitch"]
    gyro = self.get_gyro(data)
    upright_gate = components["v7_upright_gate"]
    root_height_gate = components["v7_root_height_gate"]
    angular_stability_gate = components["v7_angular_stability_gate"]
    stability_gate = upright_gate * root_height_gate * angular_stability_gate
    left_contact = foot_motion["left_contact"]
    right_contact = foot_motion["right_contact"]
    landed_side = jp.where(
        eligible_touchdown > 0.5, current_side, info["v8_support_transfer_landed_side"]
    )
    previous_support_side = jp.where(
        eligible_touchdown > 0.5,
        -current_side,
        info["v8_support_transfer_previous_support_side"],
    )
    landed_left = jp.float32(landed_side < -0.5)
    landed_right = jp.float32(landed_side > 0.5)
    previous_support_left = jp.float32(previous_support_side < -0.5)
    previous_support_right = jp.float32(previous_support_side > 0.5)
    landed_load_fraction = (
        landed_left * components["lambda_left"]
        + landed_right * components["lambda_right"]
    )
    previous_support_load_fraction = (
        previous_support_left * components["lambda_left"]
        + previous_support_right * components["lambda_right"]
    )
    landed_contact = landed_left * jp.float32(left_contact) + landed_right * jp.float32(
        right_contact
    )
    landed_load_acceptance_gate = jp.clip(
        (landed_load_fraction - 0.15) / 0.35, 0.0, 1.0
    )
    touchdown_load_acceptance = (
        recent_eligible_touchdown
        * components["valid_ground_load_gate"]
        * landed_contact
        * upright_gate
        * landed_load_acceptance_gate
    )
    current_transfer_steps = jp.where(
        eligible_touchdown > 0.5,
        jp.float32(self._config.v8_support_transfer_steps),
        info["v8_support_transfer_window_steps"],
    )
    transfer_completed_prev = jp.where(
        eligible_touchdown > 0.5, 0.0, info["v8_support_transfer_completed"]
    )
    transfer_timeout_prev = jp.where(
        eligible_touchdown > 0.5, 0.0, info["v8_support_transfer_timeout_fired"]
    )
    active_transfer_window = jp.float32(current_transfer_steps > 0.0)
    landed_load_gate = jp.clip((landed_load_fraction - 0.25) / 0.3, 0.0, 1.0)
    previous_support_release_gate = jp.clip(
        (0.65 - previous_support_load_fraction) / 0.2, 0.0, 1.0
    )
    support_transfer_progress = (
        active_transfer_window
        * stability_gate
        * landed_load_gate
        * previous_support_release_gate
    )
    support_transfer_completion = (
        active_transfer_window
        * (1.0 - transfer_completed_prev)
        * jp.float32(landed_load_fraction >= 0.55)
        * jp.float32(previous_support_load_fraction <= 0.45)
        * landed_contact
        * jp.float32(components["valid_ground_load_gate"] > 0.999)
        * jp.float32(upright_gate >= 0.7)
        * jp.float32(root_height_gate >= 0.7)
        * jp.float32(angular_stability_gate >= 0.5)
    )
    support_transfer_timeout = (
        (1.0 - eligible_touchdown)
        * active_transfer_window
        * jp.float32(current_transfer_steps <= 1.0)
        * (1.0 - transfer_completed_prev)
        * (1.0 - transfer_timeout_prev)
        * (1.0 - support_transfer_completion)
    )
    next_transfer_window_steps = jp.where(
        eligible_touchdown > 0.5,
        jp.float32(self._config.v8_support_transfer_steps - 1),
        jp.where(
            support_transfer_timeout > 0.5,
            0.0,
            jp.maximum(info["v8_support_transfer_window_steps"] - 1.0, 0.0),
        ),
    )
    next_transfer_completed = jp.maximum(
        transfer_completed_prev, support_transfer_completion
    )
    next_timeout_fired = jp.maximum(transfer_timeout_prev, support_transfer_timeout)
    expected_side = info["v9_expected_side"]
    pending_landed_side = jp.where(
        eligible_touchdown > 0.5, current_side, info["v9_pending_landed_side"]
    )
    completed_landed_side = landed_side
    timeout_from_unknown = support_transfer_timeout * jp.float32(
        jp.abs(expected_side) < 0.5
    )
    next_expected_side = jp.where(
        support_transfer_completion > 0.5,
        -completed_landed_side,
        jp.where(timeout_from_unknown > 0.5, landed_side, expected_side),
    )
    next_transfer_pending = jp.where(
        (support_transfer_completion > 0.5) | (support_transfer_timeout > 0.5),
        0.0,
        jp.where(eligible_touchdown > 0.5, 1.0, info["v9_transfer_pending"]),
    )
    next_pending_landed_side = jp.where(
        next_transfer_pending > 0.5, pending_landed_side, 0.0
    )
    valid_qualified_touchdown = components["qualified_touchdown"]
    previous_qualified_side = info["v9_previous_qualified_touchdown_side"]
    same_side_qualified_touchdown = (
        valid_qualified_touchdown
        * jp.float32(jp.abs(previous_qualified_side) > 0.5)
        * jp.float32(current_side == previous_qualified_side)
    )
    next_current_streak = jp.where(
        valid_qualified_touchdown > 0.5,
        jp.where(
            same_side_qualified_touchdown > 0.5,
            info["v9_current_same_side_qualified_touchdown_streak"] + 1.0,
            1.0,
        ),
        info["v9_current_same_side_qualified_touchdown_streak"],
    )
    next_max_streak = jp.maximum(
        info["v9_max_same_side_qualified_touchdown_streak"], next_current_streak
    )
    next_previous_qualified_side = jp.where(
        valid_qualified_touchdown > 0.5, current_side, previous_qualified_side
    )
    known_expected = jp.float32(jp.abs(expected_side) > 0.5)
    active_expected_wait = known_expected * (
        1.0 - jp.float32(info["v9_transfer_pending"] > 0.5)
    )
    steps_to_expected_touchdown = expected_qualified_touchdown * (
        info["v9_steps_since_expected_side_set"] + 1.0
    )
    next_steps_since_expected = jp.where(
        support_transfer_completion > 0.5,
        0.0,
        jp.where(
            timeout_from_unknown > 0.5,
            0.0,
            info["v9_steps_since_expected_side_set"] + active_expected_wait,
        ),
    )
    left_activity = jp.maximum(
        jp.maximum(
            positive["deep_unload_left"], positive["pre_liftoff_clearance_left"]
        ),
        jp.maximum(
            jp.maximum(
                positive["effective_liftoff_left"], positive["swing_no_contact_left"]
            ),
            jp.maximum(
                jp.maximum(
                    positive["actual_single_support_left"], positive["air_time_left"]
                ),
                jp.maximum(
                    positive["airborne_clearance_left"],
                    positive["airborne_forward_progress_left"],
                ),
            ),
        ),
    )
    right_activity = jp.maximum(
        jp.maximum(
            positive["deep_unload_right"], positive["pre_liftoff_clearance_right"]
        ),
        jp.maximum(
            jp.maximum(
                positive["effective_liftoff_right"], positive["swing_no_contact_right"]
            ),
            jp.maximum(
                jp.maximum(
                    positive["actual_single_support_right"], positive["air_time_right"]
                ),
                jp.maximum(
                    positive["airborne_clearance_right"],
                    positive["airborne_forward_progress_right"],
                ),
            ),
        ),
    )
    rewarded_expected_swing = jp.float32(
        gate_left * left_activity + gate_right * right_activity > 0.0
    )
    blocked_unexpected_swing = jp.float32(
        (1.0 - gate_left) * left_activity + (1.0 - gate_right) * right_activity > 0.0
    )
    components.update(
        {
            "deep_unload": gated("deep_unload"),
            "pre_liftoff_clearance": gated("pre_liftoff_clearance"),
            "effective_liftoff": gated("effective_liftoff"),
            "swing_no_contact": gated("swing_no_contact"),
            "actual_single_support": gated("actual_single_support"),
            "air_time": eligible_touchdown,
            "airborne_clearance": gated("airborne_clearance"),
            "airborne_forward_progress": gated("airborne_forward_progress"),
            "stable_single_support": gated("actual_single_support") * stability_gate,
            "eligible_touchdown": eligible_touchdown,
            "recent_eligible_touchdown": recent_eligible_touchdown,
            "post_touchdown_stability": recent_eligible_touchdown * stability_gate,
            "touchdown_load_acceptance": touchdown_load_acceptance,
            "left_touchdown_load_acceptance": touchdown_load_acceptance * landed_left,
            "right_touchdown_load_acceptance": touchdown_load_acceptance * landed_right,
            "support_transfer_progress": support_transfer_progress,
            "support_transfer_completion": support_transfer_completion,
            "support_transfer_timeout": support_transfer_timeout,
            "landed_load_fraction": landed_load_fraction,
            "previous_support_load_fraction": previous_support_load_fraction,
            "left_transfer_completion": support_transfer_completion * landed_left,
            "right_transfer_completion": support_transfer_completion * landed_right,
            "next_recent_eligible_touchdown_steps": next_recent_eligible_steps,
            "next_support_transfer_window_steps": next_transfer_window_steps,
            "next_support_transfer_landed_side": landed_side,
            "next_support_transfer_previous_support_side": previous_support_side,
            "next_support_transfer_completed": next_transfer_completed,
            "next_support_transfer_timeout_fired": next_timeout_fired,
            "v9_gate_left": gate_left,
            "v9_gate_right": gate_right,
            "v9_expected_side_qualified_liftoff_count": expected_qualified_liftoff,
            "v9_unexpected_side_qualified_liftoff_count": unexpected_qualified_liftoff,
            "v9_expected_side_qualified_touchdown_count": expected_qualified_touchdown,
            "v9_unexpected_side_qualified_touchdown_count": unexpected_qualified_touchdown,
            "v9_completed_lr_cycle_count": support_transfer_completion
            * previous_support_left
            * landed_right,
            "v9_completed_rl_cycle_count": support_transfer_completion
            * previous_support_right
            * landed_left,
            "v9_steps_to_expected_touchdown_sum": steps_to_expected_touchdown,
            "v9_expected_touchdown_count_for_latency": expected_qualified_touchdown,
            "v9_rewarded_expected_swing_step_count": rewarded_expected_swing,
            "v9_blocked_unexpected_swing_step_count": blocked_unexpected_swing,
            "next_v9_expected_side": next_expected_side,
            "next_v9_transfer_pending": next_transfer_pending,
            "next_v9_pending_landed_side": next_pending_landed_side,
            "next_v9_previous_qualified_touchdown_side": next_previous_qualified_side,
            "next_v9_current_same_side_qualified_touchdown_streak": next_current_streak,
            "next_v9_max_same_side_qualified_touchdown_streak": next_max_streak,
            "next_v9_steps_since_expected_side_set": next_steps_since_expected,
        }
    )
    return components
