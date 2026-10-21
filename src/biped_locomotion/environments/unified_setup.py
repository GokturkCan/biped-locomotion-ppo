# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0. See LICENSE.
# Adapted from the private research working tree based on MuJoCo Playground.
# Robot descriptions and model parameters are intentionally excluded.

from __future__ import annotations
from typing import Dict, Optional, Union
import jax.numpy as jp
from ml_collections import config_dict
from biped_locomotion.commands import process as ucp
from biped_locomotion.rewards import unified_contract as ucfg
from biped_locomotion.environments import disabled_teacher as sst


def SchaftUnifiedJoystickV1__init(
    self,
    config: Optional[config_dict.ConfigDict] = None,
    config_overrides: Optional[Dict[str, Union[str, int, list]]] = None,
):
    None
    ucfg.validate_config(self._config)
    u = self._config.unified
    self._compat = u.command_mode == ucfg.COMMAND_MODE_COMPAT
    self._canonical_side = u.start_side_mode == ucfg.START_SIDE_MODE_CANONICAL
    self._routing_enabled = bool(
        getattr(u, "enable_routed_experts", False)
        or getattr(u, "enable_routed_lateral_expert", False)
    )
    self._sched = ucp.spec_from_config(self._config)
    self._reward_scales = {
        name: jp.asarray(float(u.reward_scales[name]), dtype=jp.float32)
        for name in ucfg.ALL_REWARD_TERMS
    }
    self._dense_delta = jp.asarray(list(u.dense_delta), dtype=jp.float32)
    self._dense_sigma = jp.asarray(list(u.dense_sigma), dtype=jp.float32)
    self._pinned = tuple((int(i) for i in u.pinned_indices))
    self._wz_validity_gate_weight = jp.asarray(
        float(u.wz_validity_gate_weight), dtype=jp.float32
    )
    self._forward_turn_discount = jp.asarray(
        float(u.forward_turn_discount), dtype=jp.float32
    )
    self._u4_mode_probs = jp.asarray(list(u.u4_mode_probs), dtype=jp.float32)
    self._forward_turn_discount_by_mode = jp.asarray(
        list(u.forward_turn_discount_by_mode), dtype=jp.float32
    )
    self._wz_forward_speed_gate_weight = jp.asarray(
        float(u.wz_forward_speed_gate_weight), dtype=jp.float32
    )
    self._wz_extended_validity_gate_weight = jp.asarray(
        float(u.wz_extended_validity_gate_weight), dtype=jp.float32
    )
    self._wz_under_response_weight = jp.asarray(
        float(u.wz_under_response_weight), dtype=jp.float32
    )
    self._double_support_yaw_multiplier = jp.asarray(
        float(u.double_support_yaw_multiplier), dtype=jp.float32
    )
    self._wz_dc_alpha = ucp.wz_dc_alpha(
        self.dt, jp.asarray(float(u.wz_dc_tau), dtype=jp.float32)
    )
    self._wz_dense_uses_dc = jp.asarray(float(u.wz_dense_uses_dc), dtype=jp.float32)
    self._heading_progress_reward_weight = jp.asarray(
        float(u.heading_progress_reward_weight), dtype=jp.float32
    )
    self._strict_yaw_hold_probability = jp.asarray(
        float(u.strict_yaw_hold_probability), dtype=jp.float32
    )
    self._active_yaw_other_positive_scale = jp.asarray(
        float(u.active_yaw_other_positive_scale), dtype=jp.float32
    )
    self._active_yaw_equality_scale = jp.asarray(
        float(u.active_yaw_equality_scale), dtype=jp.float32
    )
    self._lateral_forward_progress_gate_weight = jp.asarray(
        float(u.lateral_forward_progress_gate_weight), dtype=jp.float32
    )
    self._lateral_vx_leakage_weight = jp.asarray(
        float(u.lateral_vx_leakage_weight), dtype=jp.float32
    )
    self._zero_baselined_tracking = jp.asarray(
        1.0 if bool(u.zero_baselined_tracking) else 0.0, dtype=jp.float32
    )
    self._under_response_translational_weight = jp.asarray(
        float(u.under_response_translational_weight), dtype=jp.float32
    )
    self._under_response_yaw_weight = jp.asarray(
        float(u.under_response_yaw_weight), dtype=jp.float32
    )
    self._valid_step_completion_weight = jp.asarray(
        float(u.valid_step_completion_weight), dtype=jp.float32
    )
    self._valid_step_progress_min = jp.asarray(
        float(u.valid_step_progress_min), dtype=jp.float32
    )
    self._valid_step_drift_max = jp.asarray(
        float(u.valid_step_drift_max), dtype=jp.float32
    )
    self._valid_step_stance_slip_max = jp.asarray(
        float(u.valid_step_stance_slip_max), dtype=jp.float32
    )
    self._post_liftoff_touchdown_bridge_weight = jp.asarray(
        float(u.post_liftoff_touchdown_bridge_weight), dtype=jp.float32
    )
    self._alternating_touchdown_bridge_weight = jp.asarray(
        float(u.alternating_touchdown_bridge_weight), dtype=jp.float32
    )
    self._bridge_min_air_ticks = jp.asarray(
        float(u.bridge_min_air_ticks), dtype=jp.float32
    )
    self._u6v10_unified_opposite_touchdown = bool(
        u.u6v10_unified_opposite_touchdown_bridge
    )
    self._opposite_touchdown_bridge_weight = jp.asarray(
        float(u.opposite_touchdown_bridge_weight), dtype=jp.float32
    )
    self._u6v10_reward_accounting = bool(u.u6v10_reward_accounting_telemetry)
    self._u6v10_active_terms = tuple(
        (name for name in ucfg.ALL_REWARD_TERMS if float(u.reward_scales[name]) != 0.0)
    )
    self._u6v5_side_aware = bool(u.u6v5_side_aware_bridge)
    self._require_alternation_for_completion = bool(
        u.require_alternation_for_completion
    )
    self._side_bootstrap_bridge_weight = jp.asarray(
        float(u.side_bootstrap_bridge_weight), dtype=jp.float32
    )
    self._expected_liftoff_bridge_weight = jp.asarray(
        float(u.expected_liftoff_bridge_weight), dtype=jp.float32
    )
    self._expected_liftoff_min_air_ticks = jp.asarray(
        float(u.expected_liftoff_min_air_ticks), dtype=jp.float32
    )
    self._expected_liftoff_stale_ticks = jp.asarray(
        float(u.expected_liftoff_stale_ticks), dtype=jp.float32
    )
    self._enable_gait_phase_observation = bool(u.enable_gait_phase_observation)
    self._u13_neutralize_gait_phase_observation = bool(
        getattr(u, "u13_neutralize_gait_phase_observation", False)
    )
    self._u13_track_planar_weight = jp.asarray(
        float(getattr(u, "u13_track_planar_weight", 0.0)), dtype=jp.float32
    )
    self._u13_track_yaw_weight = jp.asarray(
        float(getattr(u, "u13_track_yaw_weight", 0.0)), dtype=jp.float32
    )
    self._u13_telemetry = bool(
        getattr(u, "u13_neutralize_gait_phase_observation", False)
    )
    self._expected_unload_progress_weight = jp.asarray(
        float(u.expected_unload_progress_weight), dtype=jp.float32
    )
    self._expected_foot_still_loaded_weight = jp.asarray(
        float(u.expected_foot_still_loaded_weight), dtype=jp.float32
    )
    self._command_reward_gate_phase01 = jp.asarray(
        float(u.command_reward_gate_phase01), dtype=jp.float32
    )
    self._command_reward_gate_phase2 = jp.asarray(
        float(u.command_reward_gate_phase2), dtype=jp.float32
    )
    self._command_reward_gate_phase3 = jp.asarray(
        float(u.command_reward_gate_phase3), dtype=jp.float32
    )
    self._u6v7_zero_motion_baseline = jp.asarray(
        1.0 if bool(u.u6v7_zero_motion_tracking_baseline) else 0.0, dtype=jp.float32
    )
    self._u6v7_phase_balanced_ppo = bool(u.u6v7_phase_balanced_ppo)
    self._u6v9_recurrent_hidden_size = int(u.u6v9_recurrent_hidden_size)
    self._u6v9_corrected_gait_phase = bool(u.u6v9_corrected_gait_phase_decoder)
    self._u6v8r_first_liftoff_bootstrap = bool(u.u6v8r_first_liftoff_bootstrap)
    self._v11_minimal_gait_core = bool(u.v11_minimal_gait_core)
    self._v11_disable_phase_dense_reward_gate = bool(
        u.v11_disable_phase_dense_reward_gate
    )
    self._v11_motion_ds_penalty_weight = jp.asarray(
        float(u.v11_motion_double_support_penalty_weight), dtype=jp.float32
    )
    self._v11_ds_grace_ticks = jp.asarray(
        float(u.v11_double_support_grace_ticks), dtype=jp.float32
    )
    self._v11_ds_ramp_ticks = jp.asarray(
        float(u.v11_double_support_ramp_ticks), dtype=jp.float32
    )
    self._v11_load_transfer_weight = jp.asarray(
        float(u.v11_load_transfer_progress_weight), dtype=jp.float32
    )
    self._v11_swing_foot_weight = jp.asarray(
        float(u.v11_swing_foot_command_progress_weight), dtype=jp.float32
    )
    self._v11_swing_clearance_weight = jp.asarray(
        float(u.v11_swing_clearance_progress_weight), dtype=jp.float32
    )
    self._v11_swing_clearance_target = jp.asarray(
        float(u.v11_swing_clearance_target), dtype=jp.float32
    )
    self._v11_touchdown_weight = jp.asarray(
        float(u.v11_touchdown_load_acceptance_progress_weight), dtype=jp.float32
    )
    self._v11_support_exchange_weight = jp.asarray(
        float(u.v11_support_exchange_progress_weight), dtype=jp.float32
    )
    self._v12_role_transfer_weight = jp.asarray(
        float(u.v12_coupled_role_transfer_progress_weight), dtype=jp.float32
    )
    self._v11_potential_clip = jp.asarray(
        float(u.v11_potential_delta_clip), dtype=jp.float32
    )
    self._v11_swing_candidate_margin = jp.asarray(
        float(u.v11_swing_candidate_load_margin), dtype=jp.float32
    )
    self._enable_sign_specific_yaw_teacher = bool(u.enable_sign_specific_yaw_teacher)
    self._yaw_teacher_blend = jp.asarray(float(u.yaw_teacher_blend), dtype=jp.float32)
    self._actor_wz_observation_scale = jp.asarray(
        float(u.actor_wz_observation_scale), dtype=jp.float32
    )
    self._yaw_teacher_positive_scale = jp.asarray(
        float(u.yaw_teacher_positive_scale), dtype=jp.float32
    )
    self._yaw_teacher_negative_scale = jp.asarray(
        float(u.yaw_teacher_negative_scale), dtype=jp.float32
    )
    self._yaw_teacher_gate_alpha = sst.gate_ramp_alpha(self.dt)
