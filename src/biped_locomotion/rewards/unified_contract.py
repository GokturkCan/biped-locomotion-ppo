# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0. See LICENSE.
# Adapted from the private research working tree based on MuJoCo Playground.
# Robot descriptions and model parameters are intentionally excluded.

from __future__ import annotations
from typing import Dict, Final, Tuple

ACTOR_OBS_SIZE: Final[int] = 61
CRITIC_OBS_SIZE: Final[int] = 86
ACTION_SIZE: Final[int] = 12
U6V6_GAIT_OBS_EXTRA: Final[int] = 3
U6V6_ACTOR_OBS_SIZE: Final[int] = ACTOR_OBS_SIZE + U6V6_GAIT_OBS_EXTRA
U6V6_CRITIC_OBS_SIZE: Final[int] = CRITIC_OBS_SIZE + U6V6_GAIT_OBS_EXTRA
U6V6_GAIT_OBS_OFFSET_EXPECTED_SIDE: Final[int] = 0
U6V6_GAIT_OBS_OFFSET_GAIT_PHASE: Final[int] = 1
U6V6_GAIT_OBS_OFFSET_LOAD_RATIO: Final[int] = 2
U6V6_ACTOR_GAIT_PHASE_INDEX: Final[int] = (
    ACTOR_OBS_SIZE + U6V6_GAIT_OBS_OFFSET_GAIT_PHASE
)
U6V6_CRITIC_GAIT_PHASE_INDEX: Final[int] = (
    CRITIC_OBS_SIZE + U6V6_GAIT_OBS_OFFSET_GAIT_PHASE
)
NUM_GAIT_PHASES: Final[int] = 4
PINNED_INDICES: Final[Tuple[int, ...]] = (49, 50, 51, 57, 58, 59, 60)
S_VX: Final[float] = 0.09
S_VY: Final[float] = 0.06
S_WZ: Final[float] = 0.14
S_SPEED: Final[float] = 0.09
LATERAL_PATH_SCALE: Final[float] = 0.1
SIM_DT: Final[float] = 0.002
CTRL_DT: Final[float] = 0.02
N_SUBSTEPS: Final[int] = 10
SETTLE_STEPS: Final[int] = 1500
EPISODE_LENGTH: Final[int] = 1000
ENVELOPE_VX: Final[Tuple[float, float]] = (-0.18, 0.3)
ENVELOPE_VY: Final[Tuple[float, float]] = (-0.18, 0.18)
ENVELOPE_WZ: Final[Tuple[float, float]] = (-0.45, 0.45)
ENVELOPE_MAX_SPEED: Final[float] = 0.3
QUIET_SPEED_THRESHOLD: Final[float] = 0.015
QUIET_YAW_THRESHOLD: Final[float] = 0.02
DEBOUNCE_TICKS: Final[int] = 5
TRANSITION_TICKS: Final[int] = 25
QUIET_HOLD_TICKS: Final[Tuple[int, int]] = (100, 200)
MOVING_HOLD_TICKS: Final[Tuple[int, int]] = (150, 250)
CATEGORY_SWITCH_PROBABILITY: Final[float] = 0.8
INSTANT_TRANSITION_PROB: Final[float] = 0.05
SLEW_VX: Final[float] = 0.1
SLEW_VY: Final[float] = 0.1
SLEW_WZ: Final[float] = 0.4
MIN_MOVING_MAGNITUDE: Final[float] = 0.03
CATEGORY_NAMES: Final[Tuple[str, ...]] = (
    "QUIET",
    "FORWARD",
    "BACKWARD",
    "LATERAL",
    "YAW",
    "MIXED",
)
CAT_QUIET: Final[int] = 0
CAT_FORWARD: Final[int] = 1
CAT_BACKWARD: Final[int] = 2
CAT_LATERAL: Final[int] = 3
CAT_YAW: Final[int] = 4
CAT_MIXED: Final[int] = 5
NUM_CATEGORIES: Final[int] = len(CATEGORY_NAMES)
ROUTING_OBS_KEY: Final[str] = "command_category"
ROUTING_OBS_SIZE: Final[int] = NUM_CATEGORIES
COMMAND_MODE_COMPAT: Final[str] = "compat"
COMMAND_MODE_SHIPPED: Final[str] = "shipped"
COMMAND_MODES: Final[Tuple[str, ...]] = (COMMAND_MODE_COMPAT, COMMAND_MODE_SHIPPED)
START_SIDE_MODE_V24_UNKNOWN: Final[str] = "v24_unknown"
START_SIDE_MODE_CANONICAL: Final[str] = "canonical"
START_SIDE_MODES: Final[Tuple[str, ...]] = (
    START_SIDE_MODE_V24_UNKNOWN,
    START_SIDE_MODE_CANONICAL,
)
DENSE_DELTA_VX: Final[float] = 0.015
DENSE_DELTA_VY: Final[float] = 0.015
DENSE_DELTA_WZ: Final[float] = 0.02
DENSE_SIGMA_VX: Final[float] = 0.04
DENSE_SIGMA_VY: Final[float] = 0.05
DENSE_SIGMA_WZ: Final[float] = 0.1
ALWAYS_TERMS: Final[Tuple[str, ...]] = (
    "alive",
    "upright",
    "root_height",
    "joint_limit",
    "stance_slip",
    "locomotion_instability",
    "action_rate",
    "action_magnitude",
)
V24_VELOCITY_PATH_TERMS: Final[Tuple[str, ...]] = (
    "forward_velocity_tracking",
    "forward_velocity_progress",
    "yaw_drift_penalty",
    "lateral_drift_penalty",
    "persistent_lateral_path",
    "forward_displacement_progress",
)
DEAD_ON_V24_PATH: Final[Tuple[str, ...]] = ("excessive_clearance", "fall")
GAIT_EVENT_TERMS: Final[Tuple[str, ...]] = (
    "effective_liftoff",
    "swing_no_contact",
    "air_time",
    "airborne_clearance",
    "pre_liftoff_clearance",
    "deep_unload",
    "touchdown_load_acceptance",
    "post_touchdown_stability",
    "alternating_touchdown",
    "qualified_same_side_repeat",
    "support_transfer_progress",
    "support_transfer_completion",
    "support_transfer_timeout",
    "continuous_swing_load",
    "ground_load_loss_penalty",
    "support_loss_penalty",
    "double_support_penalty",
    "stable_single_support",
    "actual_single_support",
    "expected_swing_contact_motion_drag",
)
PLACEMENT_TERMS: Final[Tuple[str, ...]] = (
    "v16_valid_step",
    "completed_placement_band",
    "forward_touchdown",
    "airborne_forward_progress",
    "bilateral_symmetry",
    "imbalance_growth",
)
DENSE_TERMS: Final[Tuple[str, ...]] = (
    "track_vx",
    "prog_vx",
    "wrong_vx",
    "track_vy",
    "prog_vy",
    "wrong_vy",
    "track_wz",
    "prog_wz",
    "wrong_wz",
)
QUIET_TERMS: Final[Tuple[str, ...]] = (
    "quiet_vx",
    "quiet_vy",
    "quiet_yaw",
    "quiet_pose",
    "quiet_joint_velocity",
    "quiet_double_support",
    "quiet_foot_stillness",
    "quiet_contact_loss",
)
MOTION_TERMS: Final[Tuple[str, ...]] = (
    V24_VELOCITY_PATH_TERMS + GAIT_EVENT_TERMS + PLACEMENT_TERMS + DENSE_TERMS
)
ALL_REWARD_TERMS: Final[Tuple[str, ...]] = ALWAYS_TERMS + MOTION_TERMS + QUIET_TERMS
TERM_GROUP: Final[Dict[str, str]] = {
    **{t: "ALWAYS" for t in ALWAYS_TERMS},
    **{t: "MOTION" for t in MOTION_TERMS},
    **{t: "QUIET" for t in QUIET_TERMS},
}
BASELINE_REWARD_SCALES: Final[Dict[str, float]] = {
    "alive": 0.05,
    "upright": 0.2,
    "root_height": 0.2,
    "joint_limit": -2.0,
    "stance_slip": -0.25,
    "locomotion_instability": -4.0,
    "action_rate": 0.0,
    "action_magnitude": 0.0,
    "forward_velocity_tracking": 4.0,
    "forward_velocity_progress": 4.0,
    "yaw_drift_penalty": -0.2,
    "lateral_drift_penalty": -0.5,
    "persistent_lateral_path": -1.0,
    "forward_displacement_progress": 1.0,
    "effective_liftoff": 10.0,
    "swing_no_contact": 15.0,
    "air_time": 4.0,
    "airborne_clearance": 4.0,
    "pre_liftoff_clearance": 8.0,
    "deep_unload": 12.0,
    "touchdown_load_acceptance": 4.0,
    "post_touchdown_stability": 2.0,
    "alternating_touchdown": 25.0,
    "qualified_same_side_repeat": -25.0,
    "support_transfer_progress": 2.0,
    "support_transfer_completion": 40.0,
    "support_transfer_timeout": -25.0,
    "continuous_swing_load": -24.0,
    "ground_load_loss_penalty": -20.0,
    "support_loss_penalty": -20.0,
    "double_support_penalty": -4.0,
    "stable_single_support": 4.0,
    "actual_single_support": 20.0,
    "expected_swing_contact_motion_drag": -0.5,
    "v16_valid_step": 50.0,
    "completed_placement_band": -20.0,
    "forward_touchdown": 50.0,
    "airborne_forward_progress": 6.0,
    "bilateral_symmetry": -1.0,
    "imbalance_growth": -5.0,
    "track_vx": 2.0,
    "prog_vx": 3.0,
    "wrong_vx": -3.0,
    "track_vy": 2.0,
    "prog_vy": 3.0,
    "wrong_vy": -3.0,
    "track_wz": 1.5,
    "prog_wz": 2.0,
    "wrong_wz": -2.0,
    "quiet_vx": 4.0,
    "quiet_vy": 3.0,
    "quiet_yaw": 2.0,
    "quiet_pose": 2.0,
    "quiet_joint_velocity": -0.2,
    "quiet_double_support": 4.0,
    "quiet_foot_stillness": -1.0,
    "quiet_contact_loss": -25.0,
}
V24_TRACKING_SIGMA: Final[float] = 0.04
V24_PROGRESS_SIGMA: Final[float] = 0.05
V24_ROOT_HEIGHT_TARGET: Final[float] = 0.5
V24_ROOT_HEIGHT_SIGMA: Final[float] = 0.06
YAW_DRIFT_NORM: Final[float] = 0.2
LATERAL_DRIFT_DEADBAND: Final[float] = 0.04
PERSISTENT_LATERAL_DEADBAND: Final[float] = 0.14
FORWARD_DISPLACEMENT_CLIP: Final[float] = 0.002
QUIET_VX_SIGMA: Final[float] = 0.02
QUIET_VY_SIGMA: Final[float] = 0.08
QUIET_YAW_SIGMA: Final[float] = 0.15
REWARD_CLIP: Final[float] = 10000.0
U13_TRACK_PLANAR_SIGMA: Final[float] = 0.04
U13_TRACK_YAW_SIGMA: Final[float] = 0.15
U13_INIT_ACTION_STD: Final[Tuple[float, ...]] = (
    0.202,
    0.259,
    0.259,
    0.165,
    0.202,
    0.202,
    0.202,
    0.259,
    0.259,
    0.165,
    0.202,
    0.202,
)
U13_SCALE_MIN_STD: Final[float] = 0.001
U13_NORMALIZER_STD_EPS: Final[float] = 1e-06
U13_DISABLED_TERMS: Final[Tuple[str, ...]] = (
    "forward_velocity_tracking",
    "forward_velocity_progress",
    "forward_displacement_progress",
    "yaw_drift_penalty",
    "lateral_drift_penalty",
    "persistent_lateral_path",
    "expected_swing_contact_motion_drag",
    "v16_valid_step",
    "completed_placement_band",
    "forward_touchdown",
    "airborne_forward_progress",
    "bilateral_symmetry",
    "imbalance_growth",
    "track_vx",
    "prog_vx",
    "wrong_vx",
    "track_vy",
    "prog_vy",
    "wrong_vy",
    "track_wz",
    "prog_wz",
    "wrong_wz",
)
U13_CORE1_SCALES: Final[Dict[str, float]] = {
    "deep_unload": 12.0,
    "pre_liftoff_clearance": 8.0,
    "effective_liftoff": 10.0,
    "swing_no_contact": 15.0,
    "actual_single_support": 20.0,
    "stable_single_support": 4.0,
    "air_time": 4.0,
    "airborne_clearance": 4.0,
    "touchdown_load_acceptance": 4.0,
    "post_touchdown_stability": 2.0,
    "support_transfer_progress": 2.0,
    "support_transfer_completion": 40.0,
    "alternating_touchdown": 25.0,
    "qualified_same_side_repeat": -25.0,
    "support_transfer_timeout": -25.0,
    "support_loss_penalty": -20.0,
    "ground_load_loss_penalty": -20.0,
    "continuous_swing_load": -24.0,
    "double_support_penalty": -4.0,
}
U13_CORE0_SCALES: Final[Dict[str, float]] = {
    "alive": 0.05,
    "upright": 0.2,
    "root_height": 0.2,
    "joint_limit": -2.0,
    "stance_slip": -0.25,
    "locomotion_instability": -4.0,
}


class ConfigError(ValueError):
    """Raised when a unified config violates the frozen contract."""


def resolve_reward_scales(overrides: Dict[str, float] | None) -> Dict[str, float]:
    """Return baseline scales with ``overrides`` applied, validating names.

    Raises:
      ConfigError: if an override names a term outside the frozen 58-term set.
    """
    scales = dict(BASELINE_REWARD_SCALES)
    if overrides:
        unknown = set(overrides) - set(ALL_REWARD_TERMS)
        if unknown:
            raise ConfigError(
                f"reward-scale override for unknown term(s): {sorted(unknown)}"
            )
        scales.update({k: float(v) for k, v in overrides.items()})
    return scales


def validate_config(cfg):
    if len(cfg.action_scale) != 12:
        raise ConfigError("12 action scales must be supplied by ModelSpec")
    if abs(sum(cfg.unified.category_probs) - 1.0) > 1e-06:
        raise ConfigError("category probabilities must sum to one")
