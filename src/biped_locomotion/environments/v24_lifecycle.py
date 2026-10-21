# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0. See LICENSE.
# Adapted from the private research working tree based on MuJoCo Playground.
# Robot descriptions and model parameters are intentionally excluded.

from __future__ import annotations
import jax
import jax.numpy as jp
from mujoco import mjx
from biped_locomotion.environments import mjx_support as mjx_env


def FootUnload__reset(self, rng: jax.Array) -> mjx_env.State:
    qvel = jp.zeros(self.mjx_model.nv)
    data = mjx_env.make_data(
        self.mj_model,
        qpos=self._init_q,
        qvel=qvel,
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)
    data = mjx_env.step(
        self.mjx_model, data, self._default_pose, self._config.settle_steps
    )
    data = data.replace(qvel=jp.zeros(self.mjx_model.nv))
    data = mjx.forward(self.mjx_model, data)
    rng, target_rng = jax.random.split(rng)
    target_left = jp.float32(jax.random.bernoulli(target_rng, 0.5))
    target_right = 1.0 - target_left
    desired_left_contact = 1.0 - target_left
    desired_right_contact = 1.0 - target_right
    support_left = target_right
    support_right = target_left
    left_pos, right_pos = self._foot_positions(data)
    joint_pos = data.qpos[self._qposadr_jp]
    knee_l = joint_pos[self._knee_l_idx]
    knee_r = joint_pos[self._knee_r_idx]
    info = {
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "step": 0,
        "command": jp.zeros(3),
        "motor_targets": self._default_pose,
        "prev_left_foot_xy": left_pos[:2],
        "prev_right_foot_xy": right_pos[:2],
        "feet_air_time": jp.zeros(2),
        "target_air_time": jp.zeros(()),
        "target_left_swing": target_left,
        "target_right_swing": target_right,
        "desired_left_contact": desired_left_contact,
        "desired_right_contact": desired_right_contact,
        "support_left": support_left,
        "support_right": support_right,
        "left_foot_baseline_z": left_pos[2],
        "right_foot_baseline_z": right_pos[2],
        "knee_tj_baseline": jp.array([knee_l, knee_r]),
    }
    metrics = self._empty_metrics()
    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)


def FootUnloadBridgeLiteV0__reset(self, rng: jax.Array) -> mjx_env.State:
    state = self._invoke_source("FootUnloadV2.reset", rng)
    state.info["max_foot_height_l"] = jp.zeros(())
    state.info["max_foot_height_r"] = jp.zeros(())
    state.info["bridge_q_ref_base"] = state.data.qpos[self._qposadr_jp]
    return state


def FootUnloadV2__reset(self, rng: jax.Array) -> mjx_env.State:
    state = self._invoke_source("FootUnload.reset", rng)
    state.info["root_y_baseline"] = self.get_root_pos(state.data)[1]
    return state


def ResidualForwardWalkingV1__reset(self, rng: jax.Array) -> mjx_env.State:
    state = self._invoke_source("ResidualSteppingInPlaceV3.reset", rng)
    info = dict(state.info)
    rng, zero_rng, vx_rng = jax.random.split(info["rng"], 3)
    zero_sample = (
        jax.random.uniform(zero_rng, ()) < self._config.zero_command_probability
    )
    vx_sample = jax.random.uniform(
        vx_rng,
        (),
        minval=self._config.forward_vx_min,
        maxval=self._config.forward_vx_max,
    )
    vx_cmd = jp.where(zero_sample, 0.0, vx_sample)
    command = jp.array([vx_cmd, 0.0, 0.0])
    left_pos, right_pos = self._foot_positions(state.data)
    left_contact, right_contact = self._foot_contacts(state.data)
    root_x = self.get_root_pos(state.data)[0]
    info["rng"] = rng
    info["command"] = command
    info["root_x_baseline"] = root_x
    info["prev_root_x"] = root_x
    info["prev_left_foot_x"] = left_pos[0]
    info["prev_right_foot_x"] = right_pos[0]
    info["prev_left_foot_z"] = left_pos[2]
    info["prev_right_foot_z"] = right_pos[2]
    info["prev_left_contact"] = jp.float32(left_contact)
    info["prev_right_contact"] = jp.float32(right_contact)
    info["current_consecutive_left_no_contact"] = jp.zeros(())
    info["current_consecutive_right_no_contact"] = jp.zeros(())
    info["max_consecutive_left_no_contact"] = jp.zeros(())
    info["max_consecutive_right_no_contact"] = jp.zeros(())
    info["left_touchdown_count"] = jp.zeros(())
    info["right_touchdown_count"] = jp.zeros(())
    obs = self._get_obs(state.data, info)
    return state.replace(obs=obs, info=info)


def ResidualForwardWalkingV16__reset(self, rng: jax.Array) -> mjx_env.State:
    state = self._invoke_source("ResidualForwardWalkingV9.reset", rng)
    info = dict(state.info)
    info["v16_attempt_active"] = jp.zeros(())
    info["v16_attempt_side"] = jp.zeros(())
    info["v16_single_support_steps"] = jp.zeros(())
    info["v16_touchdown_latched"] = jp.zeros(())
    info["v16_forward_placement"] = jp.zeros(())
    info["v16_forward_placement_quality"] = jp.zeros(())
    return state.replace(info=info)


def ResidualForwardWalkingV18__reset(self, rng: jax.Array) -> mjx_env.State:
    state = self._invoke_source("ResidualForwardWalkingV16.reset", rng)
    info = dict(state.info)
    legacy = self._v18_legacy_phase(info["step"])
    phase = self._v18_event_phase(info, legacy)
    info = self._v18_apply_phase(info, phase)
    phase_info = dict(info)
    phase_info["step"] = phase["local_step"]
    q_ref, _, _ = self._q_ref_phase(
        phase["phase_id"],
        phase["target_left_swing"],
        phase["target_right_swing"],
        phase_info,
    )
    info["periodic_q_ref"] = q_ref
    obs = self._get_obs(state.data, info)
    return state.replace(obs=obs, info=info)


def ResidualForwardWalkingV18__step(
    self, state: mjx_env.State, action: jax.Array
) -> mjx_env.State:
    info = state.info
    step_count = info["step"]
    legacy = self._v18_legacy_phase(step_count)
    phase = self._v18_event_phase(info, legacy)
    step_info = dict(info)
    step_info = self._v18_apply_phase(step_info, phase)
    control_info = dict(step_info)
    control_info["step"] = phase["local_step"]
    q_ref, unload_weight, _ = self._q_ref_phase(
        phase["phase_id"],
        phase["target_left_swing"],
        phase["target_right_swing"],
        control_info,
    )
    ctrl = self._v3_control_law(
        action,
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
    left_swing_count = jp.float32(self._contact_count(data, self._left_foot_geom_id))
    right_swing_count = jp.float32(self._contact_count(data, self._right_foot_geom_id))
    swing_contact_count = (
        phase["target_left_swing"] * left_swing_count
        + phase["target_right_swing"] * right_swing_count
    )
    rewards = self._get_reward(
        data,
        clipped_action,
        step_info,
        phase["phase_id"],
        unload_weight,
        q_ref,
        terms,
        foot_motion,
        swing_contact_count,
    )
    scales = self._config.reward_config.scales
    scaled_rewards = {k: v * scales[k] for k, v in rewards.items()}
    reward = jp.clip(sum(scaled_rewards.values()) * self.dt, -10000.0, 10000.0)
    target_clipped_steps = info["target_clipped_steps"] + ctrl["target_clipped"]
    min_swing_contact_count_left = jp.where(
        phase["target_left_swing"] > 0.5,
        jp.minimum(info["min_swing_contact_count_left"], left_swing_count),
        info["min_swing_contact_count_left"],
    )
    min_swing_contact_count_right = jp.where(
        phase["target_right_swing"] > 0.5,
        jp.minimum(info["min_swing_contact_count_right"], right_swing_count),
        info["min_swing_contact_count_right"],
    )
    left_roll, _, right_roll, _ = self._foot_roll_pitch(data)
    max_abs_roll_left = jp.maximum(info["max_abs_roll_left"], jp.abs(left_roll))
    max_abs_roll_right = jp.maximum(info["max_abs_roll_right"], jp.abs(right_roll))
    left_height = foot_motion["left_pos"][2] - step_info["left_foot_baseline_z"]
    right_height = foot_motion["right_pos"][2] - step_info["right_foot_baseline_z"]
    max_foot_height_l = jp.maximum(info["max_foot_height_l"], left_height)
    max_foot_height_r = jp.maximum(info["max_foot_height_r"], right_height)
    metrics = self._update_metrics(
        state.metrics,
        data,
        clipped_action,
        action,
        motor_targets,
        scaled_rewards,
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
        min_swing_contact_count_left,
        min_swing_contact_count_right,
        max_foot_height_l,
        max_foot_height_r,
        max_abs_roll_left,
        max_abs_roll_right,
    )
    terminal_fall = metrics["termination/fall_event"]
    reward = reward + terminal_fall * self._config.terminal_fall_cost
    metrics["termination/terminal_fall_cost"] = (
        terminal_fall * self._config.terminal_fall_cost
    )
    components = self._v9_components(
        data, step_info, phase["phase_id"], terms, foot_motion
    )
    next_step = step_count + 1
    new_info = dict(step_info)
    new_info["last_last_act"] = info["last_act"]
    new_info["last_act"] = clipped_action
    new_info["motor_targets"] = motor_targets
    new_info["step"] = next_step
    new_info["prev_left_foot_xy"] = foot_motion["left_pos"][:2]
    new_info["prev_right_foot_xy"] = foot_motion["right_pos"][:2]
    new_info["feet_air_time"] = foot_motion["feet_air_time"]
    new_info["target_air_time"] = terms["target_air_time_next"]
    new_info["max_foot_height_l"] = max_foot_height_l
    new_info["max_foot_height_r"] = max_foot_height_r
    new_info["target_clipped_steps"] = target_clipped_steps
    new_info["min_swing_contact_count_left"] = min_swing_contact_count_left
    new_info["min_swing_contact_count_right"] = min_swing_contact_count_right
    new_info["max_abs_roll_left"] = max_abs_roll_left
    new_info["max_abs_roll_right"] = max_abs_roll_right
    left_pos, right_pos = self._foot_positions(data)
    left_contact, right_contact = self._foot_contacts(data)
    current_left_no_contact = jp.where(
        left_contact, 0.0, info["current_consecutive_left_no_contact"] + 1.0
    )
    current_right_no_contact = jp.where(
        right_contact, 0.0, info["current_consecutive_right_no_contact"] + 1.0
    )
    left_touchdown = jp.float32((info["prev_left_contact"] < 0.5) & left_contact)
    right_touchdown = jp.float32((info["prev_right_contact"] < 0.5) & right_contact)
    new_info["prev_root_x"] = self.get_root_pos(data)[0]
    new_info["prev_left_foot_x"] = left_pos[0]
    new_info["prev_right_foot_x"] = right_pos[0]
    new_info["prev_left_foot_z"] = left_pos[2]
    new_info["prev_right_foot_z"] = right_pos[2]
    new_info["prev_left_contact"] = jp.float32(left_contact)
    new_info["prev_right_contact"] = jp.float32(right_contact)
    new_info["current_consecutive_left_no_contact"] = current_left_no_contact
    new_info["current_consecutive_right_no_contact"] = current_right_no_contact
    new_info["max_consecutive_left_no_contact"] = jp.maximum(
        info["max_consecutive_left_no_contact"], current_left_no_contact
    )
    new_info["max_consecutive_right_no_contact"] = jp.maximum(
        info["max_consecutive_right_no_contact"], current_right_no_contact
    )
    new_info["left_touchdown_count"] = info["left_touchdown_count"] + left_touchdown
    new_info["right_touchdown_count"] = info["right_touchdown_count"] + right_touchdown
    touchdown_left_metric = metrics["touchdown/left_count"]
    touchdown_right_metric = metrics["touchdown/right_count"]
    touchdown_event = jp.maximum(touchdown_left_metric, touchdown_right_metric)
    touchdown_side = touchdown_right_metric - touchdown_left_metric
    previous_side = info["last_true_touchdown_side"]
    touchdown_side = jp.where(touchdown_event > 0.5, touchdown_side, previous_side)
    new_info["last_true_touchdown_side"] = touchdown_side
    new_info["landed_touchdown_side"] = touchdown_side
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
    new_info["v8_last_qualified_touchdown_side"] = components[
        "next_last_qualified_touchdown_side"
    ]
    new_info["v8_recent_eligible_touchdown_steps"] = components[
        "next_recent_eligible_touchdown_steps"
    ]
    new_info["v8_support_transfer_window_steps"] = components[
        "next_support_transfer_window_steps"
    ]
    new_info["v8_support_transfer_landed_side"] = components[
        "next_support_transfer_landed_side"
    ]
    new_info["v8_support_transfer_previous_support_side"] = components[
        "next_support_transfer_previous_support_side"
    ]
    new_info["v8_support_transfer_completed"] = components[
        "next_support_transfer_completed"
    ]
    new_info["v8_support_transfer_timeout_fired"] = components[
        "next_support_transfer_timeout_fired"
    ]
    new_info["v9_expected_side"] = components["next_v9_expected_side"]
    new_info["v9_transfer_pending"] = components["next_v9_transfer_pending"]
    new_info["v9_pending_landed_side"] = components["next_v9_pending_landed_side"]
    new_info["v9_previous_qualified_touchdown_side"] = components[
        "next_v9_previous_qualified_touchdown_side"
    ]
    new_info["v9_current_same_side_qualified_touchdown_streak"] = components[
        "next_v9_current_same_side_qualified_touchdown_streak"
    ]
    new_info["v9_max_same_side_qualified_touchdown_streak"] = components[
        "next_v9_max_same_side_qualified_touchdown_streak"
    ]
    new_info["v9_steps_since_expected_side_set"] = components[
        "next_v9_steps_since_expected_side_set"
    ]
    new_info["v16_attempt_active"] = components["next_v16_attempt_active"]
    new_info["v16_attempt_side"] = components["next_v16_attempt_side"]
    new_info["v16_single_support_steps"] = components["next_v16_single_support_steps"]
    new_info["v16_touchdown_latched"] = components["next_v16_touchdown_latched"]
    new_info["v16_forward_placement"] = components["next_v16_forward_placement"]
    new_info["v16_forward_placement_quality"] = components[
        "next_v16_forward_placement_quality"
    ]
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


def ResidualForwardWalkingV24__reset(self, rng: jax.Array) -> mjx_env.State:
    state = self._invoke_source("ResidualForwardWalkingV18.reset", rng)
    info = dict(state.info)
    info["v24_last_left_placement"] = jp.zeros(())
    info["v24_last_right_placement"] = jp.zeros(())
    info["v24_left_valid"] = jp.zeros(())
    info["v24_right_valid"] = jp.zeros(())
    info["v24_pair_valid"] = jp.zeros(())
    info["v24_abs_imbalance_previous"] = jp.zeros(())
    return state.replace(info=info)


def ResidualForwardWalkingV24__step(
    self, state: mjx_env.State, action: jax.Array
) -> mjx_env.State:
    next_state = self._invoke_source("ResidualForwardWalkingV18.step", state, action)
    info = state.info
    legacy = self._v18_legacy_phase(info["step"])
    phase = self._v18_event_phase(info, legacy)
    step_info = self._v18_apply_phase(dict(info), phase)
    terms = self._foot_unload_terms(next_state.data, step_info)
    foot_motion = self._foot_motion(next_state.data, step_info)
    components = self._v9_components(
        next_state.data, step_info, phase["phase_id"], terms, foot_motion
    )
    new_info = dict(next_state.info)
    new_info["v24_last_left_placement"] = components["next_v24_last_left_placement"]
    new_info["v24_last_right_placement"] = components["next_v24_last_right_placement"]
    new_info["v24_left_valid"] = components["next_v24_left_valid"]
    new_info["v24_right_valid"] = components["next_v24_right_valid"]
    new_info["v24_pair_valid"] = components["next_v24_pair_valid"]
    new_info["v24_abs_imbalance_previous"] = components[
        "next_v24_abs_imbalance_previous"
    ]
    obs = self._get_obs(next_state.data, new_info)
    return next_state.replace(obs=obs, info=new_info)


def ResidualForwardWalkingV5__reset(self, rng: jax.Array) -> mjx_env.State:
    state = self._invoke_source("ResidualForwardWalkingV1.reset", rng)
    info = dict(state.info)
    rng, phase_rng = jax.random.split(info["rng"])
    start_right_half = jax.random.bernoulli(phase_rng, 0.5)
    start_step = jp.where(start_right_half, self._half_cycle_steps, 0)
    target_left, target_right, phase_id, local_step, gait_phase = self._stepping_phase(
        start_step
    )
    phase_info = dict(info)
    phase_info["target_left_swing"] = target_left
    phase_info["target_right_swing"] = target_right
    phase_info["step"] = local_step
    q_ref, _, _ = self._q_ref_phase(phase_id, target_left, target_right, phase_info)
    info["rng"] = rng
    info["step"] = start_step
    info["target_left_swing"] = target_left
    info["target_right_swing"] = target_right
    info["support_left"] = target_right
    info["support_right"] = target_left
    info["desired_left_contact"] = 1.0 - target_left
    info["desired_right_contact"] = 1.0 - target_right
    info["periodic_q_ref"] = q_ref
    info["gait_phase"] = gait_phase
    obs = self._get_obs(state.data, info)
    return state.replace(obs=obs, info=info)


def ResidualForwardWalkingV6__reset(self, rng: jax.Array) -> mjx_env.State:
    state = self._invoke_source("ResidualForwardWalkingV5.reset", rng)
    info = dict(state.info)
    info["left_foot_z_nominal"] = info["left_foot_baseline_z"]
    info["right_foot_z_nominal"] = info["right_foot_baseline_z"]
    info["recent_true_touchdown_steps"] = jp.zeros(())
    obs = self._get_obs(state.data, info)
    return state.replace(obs=obs, info=info)


def ResidualForwardWalkingV7__reset(self, rng: jax.Array) -> mjx_env.State:
    state = self._invoke_source("ResidualForwardWalkingV6.reset", rng)
    info = dict(state.info)
    info["last_true_touchdown_side"] = jp.zeros(())
    info["landed_touchdown_side"] = jp.zeros(())
    return state.replace(info=info)


def ResidualForwardWalkingV8__reset(self, rng: jax.Array) -> mjx_env.State:
    state = self._invoke_source("ResidualForwardWalkingV7.reset", rng)
    info = dict(state.info)
    for side in ("left", "right"):
        info[f"v8_consecutive_no_contact_steps_{side}"] = jp.zeros(())
        info[f"v8_max_height_gain_during_air_{side}"] = jp.zeros(())
        info[f"v8_air_qualified_{side}"] = jp.zeros(())
        info[f"v8_consecutive_contact_steps_{side}"] = jp.zeros(())
        info[f"v8_steps_since_last_qualified_touchdown_{side}"] = jp.float32(
            self._config.v8_min_steps_between_qualified_touchdowns
        )
    info["v8_last_qualified_touchdown_side"] = jp.zeros(())
    info["v8_recent_eligible_touchdown_steps"] = jp.zeros(())
    info["v8_support_transfer_window_steps"] = jp.zeros(())
    info["v8_support_transfer_landed_side"] = jp.zeros(())
    info["v8_support_transfer_previous_support_side"] = jp.zeros(())
    info["v8_support_transfer_completed"] = jp.zeros(())
    info["v8_support_transfer_timeout_fired"] = jp.zeros(())
    return state.replace(info=info)


def ResidualForwardWalkingV9__reset(self, rng: jax.Array) -> mjx_env.State:
    state = self._invoke_source("ResidualForwardWalkingV8.reset", rng)
    info = dict(state.info)
    info["v9_expected_side"] = jp.zeros(())
    info["v9_transfer_pending"] = jp.zeros(())
    info["v9_pending_landed_side"] = jp.zeros(())
    info["v9_previous_qualified_touchdown_side"] = jp.zeros(())
    info["v9_current_same_side_qualified_touchdown_streak"] = jp.zeros(())
    info["v9_max_same_side_qualified_touchdown_streak"] = jp.zeros(())
    info["v9_steps_since_expected_side_set"] = jp.zeros(())
    return state.replace(info=info)


def ResidualSteppingInPlaceV1__reset(self, rng: jax.Array) -> mjx_env.State:
    state = self._invoke_source("FootUnloadBridgeLiteV0.reset", rng)
    info = dict(state.info)
    info["command"] = jp.zeros(3)
    target_left, target_right, phase_id, local_step, gait_phase = self._stepping_phase(
        info["step"]
    )
    phase_info = dict(info)
    phase_info["target_left_swing"] = target_left
    phase_info["target_right_swing"] = target_right
    phase_info["step"] = local_step
    q_ref, _, _ = self._q_ref_phase(phase_id, target_left, target_right, phase_info)
    info["target_left_swing"] = target_left
    info["target_right_swing"] = target_right
    info["support_left"] = target_right
    info["support_right"] = target_left
    info["desired_left_contact"] = 1.0 - target_left
    info["desired_right_contact"] = 1.0 - target_right
    info["periodic_q_ref"] = q_ref
    info["gait_phase"] = gait_phase
    obs = self._get_obs(state.data, info)
    metrics = self._empty_metrics()
    reward, done = jp.zeros(2)
    return state.replace(obs=obs, reward=reward, done=done, metrics=metrics, info=info)


def ResidualSteppingInPlaceV3__reset(self, rng: jax.Array) -> mjx_env.State:
    state = self._invoke_source("ResidualSteppingInPlaceV1.reset", rng)
    info = dict(state.info)
    info["target_clipped_steps"] = jp.zeros(())
    info["min_swing_contact_count_left"] = jp.float32(
        self._config.contact_reduction_nominal_count
    )
    info["min_swing_contact_count_right"] = jp.float32(
        self._config.contact_reduction_nominal_count
    )
    info["max_abs_roll_left"] = jp.zeros(())
    info["max_abs_roll_right"] = jp.zeros(())
    return state.replace(info=info)
