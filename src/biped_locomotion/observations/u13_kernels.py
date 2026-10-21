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
from biped_locomotion.gait.unified_events import u6_decode_gait_phase


def SchaftUnifiedJoystickV1__get_obs(
    self, data: mjx.Data, info: Dict[str, Any]
) -> Dict[str, jax.Array]:
    """Build the unified 61/86 prefixes; U13 appends reserved zeros for 64/89.

    Deliberately does **not** call ``V24._get_obs`` so no legacy command
    projection or ``root_y_baseline`` channel can leak in (§18.1).
    """
    cmd_state = (
        ucp.from_info(info)
        if ucp.has_command_state(info)
        else ucp.neutral_command_state()
    )
    cmd = cmd_state.command
    u = self._config.unified
    joint_pos = data.qpos[self._qposadr_jp]
    joint_vel = data.qvel[self._dofadr_jp]
    q_nominal = info.get("bridge_q_ref_base", self._default_pose)
    last_act = info.get("last_act", jp.zeros(ucfg.ACTION_SIZE))
    left_contact, right_contact = self._foot_contacts(data)
    gait_phase = info.get("gait_phase", jp.zeros(()))
    gate = cmd_state.locomotion_request
    yaw = self._forward_yaw_from_quat(self.get_orientation(data))
    heading_error = ucp.wrap_to_pi(cmd_state.heading_target - yaw)
    e_path_y = info.get("unified_e_path_y", jp.zeros(()))
    pair_valid = info.get("v24_pair_valid", jp.zeros(()))
    p_l = info.get("v24_last_left_placement", jp.zeros(()))
    p_r = info.get("v24_last_right_placement", jp.zeros(()))
    imbalance = pair_valid * (p_r - p_l) / (p_r + p_l + 1e-06)
    speed_cmd = jp.sqrt(cmd[0] ** 2 + cmd[1] ** 2)
    cmd_wz_normalized = cmd[2] / u.s_wz
    effective_wz_obs_scale = jp.where(
        self._enable_sign_specific_yaw_teacher,
        self._actor_wz_observation_scale,
        jp.ones(()),
    )
    actor_cmd_wz = effective_wz_obs_scale * cmd_wz_normalized
    state = jp.concatenate(
        [
            joint_pos - q_nominal,
            joint_vel,
            self.get_gravity(data),
            self.get_local_linvel(data),
            self.get_gyro(data),
            jp.array(
                [
                    gate * jp.sin(2.0 * jp.pi * gait_phase),
                    gate * jp.cos(2.0 * jp.pi * gait_phase),
                ]
            ),
            last_act,
            jp.array([jp.float32(left_contact), jp.float32(right_contact)]),
            jp.array(
                [
                    cmd[0] / u.s_vx,
                    cmd[1] / u.s_vy,
                    actor_cmd_wz,
                    e_path_y / u.lateral_path_scale,
                    jp.sin(heading_error),
                    jp.cos(heading_error),
                    imbalance,
                    pair_valid,
                    gate,
                    speed_cmd / u.s_speed,
                    cmd_state.start_swing_side,
                    ucp.transition_progress(cmd_state),
                ]
            ),
        ]
    )
    force_norm = data.actuator_force / jp.maximum(jp.abs(self._effort_uppers), 1.0)
    critic_state = state.at[51].set(cmd_wz_normalized)
    privileged = jp.concatenate(
        [
            critic_state,
            jp.array([jp.float32(left_contact), jp.float32(right_contact)]),
            self.get_global_linvel(data),
            self.get_global_angvel(data),
            force_norm,
            joint_pos[self._knee_l_idx : self._knee_l_idx + 1],
            joint_pos[self._knee_r_idx : self._knee_r_idx + 1],
            jp.array([p_l, p_r, info.get("v24_abs_imbalance_previous", jp.zeros(()))]),
        ]
    )
    if getattr(self, "_enable_gait_phase_observation", False):
        _eps_g = 1e-06
        _movement_obs = jp.clip(
            sum(
                (
                    (jp.abs(cmd[i]) > self._dense_delta[i]).astype(jp.float32)
                    for i in range(3)
                )
            ),
            0.0,
            1.0,
        )
        _last_credited_obs = info.get("u6_last_credited_side", jp.zeros(()))
        _has_prev_obs = (jp.abs(_last_credited_obs) > 0.5).astype(jp.float32)
        _side_init_obs = info.get("u6v6_expected_side_init", jp.zeros(()))
        expected_side_obs = jp.where(
            _has_prev_obs > 0.5, -_last_credited_obs, _side_init_obs
        )
        expected_side_obs = jp.where(_movement_obs < 0.5, 0.0, expected_side_obs)
        _latched_obs = info.get("u6_alternation_latched", jp.zeros(()))
        _outstanding_obs = info.get("u6_commit_outstanding", jp.zeros(()))
        gait_phase_obs = u6_decode_gait_phase(
            _movement_obs,
            _latched_obs,
            _outstanding_obs,
            corrected=self._u6v9_corrected_gait_phase,
        )
        fz_left_obs, fz_right_obs = self._foot_vertical_ground_reaction_forces(data)
        _total_fz_obs = fz_left_obs + fz_right_obs + _eps_g
        _ratio_l_obs = jp.clip(fz_left_obs / _total_fz_obs, 0.0, 1.0)
        _ratio_r_obs = jp.clip(fz_right_obs / _total_fz_obs, 0.0, 1.0)
        expected_load_ratio_obs = jp.where(
            expected_side_obs < -0.5,
            _ratio_l_obs,
            jp.where(expected_side_obs > 0.5, _ratio_r_obs, 0.0),
        )
        gait_obs = jp.array(
            [expected_side_obs, gait_phase_obs, expected_load_ratio_obs]
        )
        if self._u13_neutralize_gait_phase_observation:
            gait_obs = jp.zeros_like(gait_obs)
        state = jp.concatenate([state, gait_obs])
        privileged = jp.concatenate([privileged, gait_obs])
    obs = {"state": state, "privileged_state": privileged}
    if getattr(self, "_routing_enabled", False):
        obs[ucfg.ROUTING_OBS_KEY] = jax.nn.one_hot(
            cmd_state.category, ucfg.ROUTING_OBS_SIZE, dtype=jp.float32
        )
    return obs
