# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0. See LICENSE.
# Adapted from the private research working tree based on MuJoCo Playground.
# Robot descriptions and model parameters are intentionally excluded.

from __future__ import annotations
from typing import Any
import jax
import jax.numpy as jp
from mujoco import mjx


def ResidualForwardWalkingV2__get_obs(
    self, data: mjx.Data, info: dict[str, Any]
) -> dict[str, jax.Array]:
    joint_pos = data.qpos[self._qposadr_jp]
    joint_vel = data.qvel[self._dofadr_jp]
    q_nominal = info.get("bridge_q_ref_base", self._default_pose)
    gait_phase = info.get("gait_phase", jp.zeros(()))
    left_contact, right_contact = self._foot_contacts(data)
    state = jp.concatenate(
        [
            joint_pos - q_nominal,
            joint_vel,
            self.get_gravity(data),
            self.get_local_linvel(data),
            self.get_gyro(data),
            jp.array(
                [jp.sin(2.0 * jp.pi * gait_phase), jp.cos(2.0 * jp.pi * gait_phase)]
            ),
            info["last_act"],
            jp.array([jp.float32(left_contact), jp.float32(right_contact)]),
            info["command"],
        ]
    )
    return {"state": state, "privileged_state": self._get_privileged_obs(data, state)}


def ResidualForwardWalkingV22__get_obs(
    self, data: mjx.Data, info: dict[str, Any]
) -> dict[str, jax.Array]:
    raw_command = info["command"]
    vx_cmd = raw_command[0]
    yaw_rate_cmd = raw_command[2]
    yaw = self._forward_yaw_from_quat(self.get_orientation(data))
    angle = -yaw
    heading_error = jp.arctan2(jp.sin(angle), jp.cos(angle))
    observation_command = jp.array(
        [vx_cmd * jp.cos(heading_error), vx_cmd * jp.sin(heading_error), yaw_rate_cmd]
    )
    obs_info = dict(info)
    obs_info["command"] = observation_command
    return self._invoke_source("ResidualForwardWalkingV2._get_obs", data, obs_info)


def ResidualForwardWalkingV24__get_obs(
    self, data: mjx.Data, info: dict[str, Any]
) -> dict[str, jax.Array]:
    core = self._invoke_source("ResidualForwardWalkingV22._get_obs", data, info)
    core_state = core["state"]
    core_privileged = core["privileged_state"]
    yaw = self._forward_yaw_from_quat(self.get_orientation(data))
    angle = -yaw
    heading_error = jp.arctan2(jp.sin(angle), jp.cos(angle))
    sin_heading_error = jp.sin(heading_error)
    cos_heading_error = jp.cos(heading_error)
    root_pos = self.get_root_pos(data)
    root_y_baseline = info.get("root_y_baseline", root_pos[1])
    e_y = root_pos[1] - root_y_baseline
    e_y_obs = e_y / 0.1
    epsilon = 1e-06
    last_left = info.get("v24_last_left_placement", jp.zeros(()))
    last_right = info.get("v24_last_right_placement", jp.zeros(()))
    pair_valid = info.get("v24_pair_valid", jp.zeros(()))
    bilateral_imbalance_b = pair_valid * (
        (last_right - last_left) / (last_right + last_left + epsilon)
    )
    abs_imbalance_previous = info.get("v24_abs_imbalance_previous", jp.zeros(()))
    mirrored = jp.array(
        [
            e_y_obs,
            sin_heading_error,
            cos_heading_error,
            bilateral_imbalance_b,
            pair_valid,
        ]
    )
    actor_state = jp.concatenate([core_state, mirrored])
    privileged_state = jp.concatenate(
        [
            core_privileged,
            mirrored,
            jp.array([last_left, last_right, abs_imbalance_previous]),
        ]
    )
    return {"state": actor_state, "privileged_state": privileged_state}


def WalkingW1__get_privileged_obs(self, data: mjx.Data, obs: jax.Array) -> jax.Array:
    joint_pos = data.qpos[self._qposadr_jp]
    left_contact, right_contact = self._foot_contacts(data)
    force_norm = data.actuator_force / jp.maximum(jp.abs(self._effort_uppers), 1.0)
    return jp.concatenate(
        [
            obs,
            jp.array([left_contact, right_contact], dtype=jp.float32),
            self.get_global_linvel(data),
            self.get_global_angvel(data),
            force_norm,
            joint_pos[self._knee_l_idx : self._knee_l_idx + 1],
            joint_pos[self._knee_r_idx : self._knee_r_idx + 1],
        ]
    )
