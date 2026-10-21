// Native port of the V24 gait event / state machine -- the PERSISTED-STATE path
// only (the v8_*/v9_*/v16_*/v24_* state fields and the V18 gait_phase that the
// 57-value observation depends on). Reward, metric and diagnostic computations
// are deliberately excluded: none of them feeds back into persisted state.
// The Python counterparts live in src/biped_locomotion/gait/ and
// src/biped_locomotion/rewards/ (see docs/implementation_mapping.md).
//
// Constants below are the fully resolved V24 values and match
// configs/v24_environment.json. The only robot-specific value, the root-height
// target, is supplied by the caller (RuntimeBindings::target_height).
#pragma once

#include <mujoco/mujoco.h>

namespace biped {

// ---------------------------------------------------------------------------
// Resolved config constants (match configs/v24_environment.json).
// ---------------------------------------------------------------------------
constexpr double kHeightSigma = 0.06;                 // reward_config.height_sigma
constexpr double kLoadTransferForceEpsilon = 1.0e-6;  // config.load_transfer_force_epsilon
constexpr double kValidGroundLoadFractionScale = 0.60;

constexpr double kV8LiftoffNoContactSteps = 4.0;
constexpr double kV8LiftoffHeightGain = 0.005;
constexpr double kV8TouchdownContactConfirmSteps = 3.0;
constexpr double kV8MinStepsBetweenQualifiedTouchdowns = 8.0;
constexpr double kV8RecentEligibleTouchdownSteps = 20.0;
constexpr double kV8SupportTransferSteps = 20.0;

constexpr double kV16ForwardPlacementTarget = 0.10;       // resolved (V16 def)
constexpr double kV16SingleSupportTargetSteps = 1.0;      // resolved (V17 override, NOT V16's 8.0)

constexpr double kV18PrepSteps = 15.0;
constexpr double kV18NominalSwingSteps = 20.0;
constexpr double kV18NominalTransferSteps = 15.0;

// FootUnloadBridgeLiteV0._post_init: resolved bridge
// phase step counts, shared with _stepping_phase/_bridge_phase.
constexpr double kBridgePrepShiftSteps = 40.0;
constexpr double kBridgeUnloadSteps = 25.0;
constexpr double kBridgeSwingHoldSteps = 25.0;
constexpr double kBridgeTouchdownSteps = 20.0;
constexpr double kBridgeRecenterSteps = 30.0;
constexpr double kHalfCycleSteps = kBridgePrepShiftSteps + kBridgeUnloadSteps +
                                    kBridgeSwingHoldSteps + kBridgeTouchdownSteps +
                                    kBridgeRecenterSteps;  // 140
constexpr double kCycleSteps = 2.0 * kHalfCycleSteps;      // 280

constexpr double kBilateralEpsilonD = 1.0e-6;  // V24's epsilon (obs_builder.h uses kBilateralEps as float)

// ---------------------------------------------------------------------------
// Persisted per-episode state (mirrors the subset of mjx_env.State.info this
// runtime actually needs -- everything reset() zeros except the two noted
// exceptions).
// ---------------------------------------------------------------------------
struct EventMachineState {
  // Global step counter (info["step"]); drives _stepping_phase/_bridge_phase
  // bootstrap fallback. Reset to 0.
  double step = 0.0;

  // Per-episode constants captured ONCE at reset from the settled pose
  // (, ultimately left/right_foot_baseline_z set at
  // FootUnload.reset) -- NOT touched by step().
  // Root-height target (m); robot-specific, from RuntimeBindings::target_height.
  double target_height = 0.0;
  double left_foot_z_nominal = 0.0;
  double right_foot_z_nominal = 0.0;

  // NOTE: the source environment's info["prev_left_contact"]/["prev_right_contact"]
  // are intentionally NOT
  // modeled here: they feed _v8_liftoff_touchdown_state's raw_liftoff/
  // raw_touchdown outputs only, which are diagnostics-only and are not read
  // by any persisted next_* field anywhere in the V8-V24 chain (confirmed by
  // direct reading; see event_machine.cpp's V8LiftoffTouchdownState, which
  // has no prev_contact parameter for exactly this reason).

  // v8_*.
  double v8_consecutive_no_contact_steps_left = 0.0;
  double v8_consecutive_no_contact_steps_right = 0.0;
  double v8_max_height_gain_during_air_left = 0.0;
  double v8_max_height_gain_during_air_right = 0.0;
  double v8_air_qualified_left = 0.0;
  double v8_air_qualified_right = 0.0;
  double v8_consecutive_contact_steps_left = 0.0;
  double v8_consecutive_contact_steps_right = 0.0;
  // NOT zero at reset --: initialized to
  // v8_min_steps_between_qualified_touchdowns.
  double v8_steps_since_last_qualified_touchdown_left = kV8MinStepsBetweenQualifiedTouchdowns;
  double v8_steps_since_last_qualified_touchdown_right = kV8MinStepsBetweenQualifiedTouchdowns;
  double v8_last_qualified_touchdown_side = 0.0;
  // NOTE: v8_recent_eligible_touchdown_steps / v8_support_transfer_* are
  // written into info using V9's (not V8's) formulas once V9 is present in
  // the MRO -- see the header comment. Field names kept as "v8_" to match
  // the source environment's info dict key names verbatim.
  double v8_recent_eligible_touchdown_steps = 0.0;
  double v8_support_transfer_window_steps = 0.0;
  double v8_support_transfer_landed_side = 0.0;
  double v8_support_transfer_previous_support_side = 0.0;
  double v8_support_transfer_completed = 0.0;
  double v8_support_transfer_timeout_fired = 0.0;

  // v9_*.
  double v9_expected_side = 0.0;
  double v9_transfer_pending = 0.0;
  double v9_pending_landed_side = 0.0;
  double v9_previous_qualified_touchdown_side = 0.0;
  double v9_current_same_side_qualified_touchdown_streak = 0.0;
  double v9_max_same_side_qualified_touchdown_streak = 0.0;
  double v9_steps_since_expected_side_set = 0.0;

  // v16_*.
  double v16_attempt_active = 0.0;
  double v16_attempt_side = 0.0;
  double v16_single_support_steps = 0.0;
  double v16_touchdown_latched = 0.0;
  double v16_forward_placement = 0.0;
  double v16_forward_placement_quality = 0.0;

  // v24_*. These map
  // directly to ObsInfo::v24_last_left_placement / v24_last_right_placement /
  // v24_pair_valid (obs_builder.h).
  double v24_last_left_placement = 0.0;
  double v24_last_right_placement = 0.0;
  double v24_left_valid = 0.0;
  double v24_right_valid = 0.0;
  double v24_pair_valid = 0.0;
  double v24_abs_imbalance_previous = 0.0;
};

// Captures the per-episode foot-z-nominal constants from the CURRENT mjData
// (d->geom_xpos of the two foot geoms) and the root-height target, and zeros
// all other state via `*state = EventMachineState{}` (never memset: the two
// v8_steps_since_last_qualified_touchdown_{side} fields default-initialise to
// kV8MinStepsBetweenQualifiedTouchdowns, not zero).
//
// Call this only after the settle sequence (kSettleSteps mj_step calls with
// zero control, then qvel = 0, then mj_forward) -- the same point at which the
// settled reference pose is captured. Calling it earlier captures a wrong
// foot-z baseline and silently shifts the height-gain gating.
void ResetEventMachineState(EventMachineState* state, const mjModel* m, const mjData* d,
                             int left_foot_geom_id, int right_foot_geom_id, double target_height);

// Per-tick physical inputs, all derived from mjData AFTER this tick's
// mj_step calls (matches the source environment's V18.step(), which computes contact/GRF/
// orientation from `data` post-integration).
struct EventMachineInputs {
  bool left_contact = false;
  bool right_contact = false;
  double left_foot_pos_x = 0.0;
  double left_foot_pos_z = 0.0;
  double right_foot_pos_x = 0.0;
  double right_foot_pos_z = 0.0;
  double fz_left = 0.0;   // ComputeFootGrf(...).left
  double fz_right = 0.0;  // ComputeFootGrf(...).right
  double roll = 0.0;      // body roll from the orientation quaternion
  double pitch = 0.0;
  double gyro_x = 0.0;
  double gyro_y = 0.0;
  double root_z = 0.0;
};

// Transient (non-persisted) per-tick outputs -- event flags and continuous
// quantities useful for deterministic unit tests and telemetry, but not
// themselves carried into the next tick's EventMachineState (the underlying
// state fields they're derived from ARE persisted; these are convenience
// re-exposures of this tick's computation, matching the source environment's `components`
// dict entries of the same conceptual name).
struct EventMachineOutputs {
  // V18 gait_phase -- feeds obs indices 33/34
  // (sin/cos) via BuildObservationFromRaw's ObsInfo::gait_phase.
  double gait_phase = 0.0;

  // V8 per-side liftoff/touchdown qualification edges this tick
  //.
  bool qualified_liftoff_left = false;
  bool qualified_liftoff_right = false;
  bool qualified_touchdown_left = false;
  bool qualified_touchdown_right = false;

  // V9's expected-side-gated eligible touchdown this tick
  //.
  bool eligible_touchdown = false;

  // V9 support-transfer completion/timeout this tick.
  bool support_transfer_completion = false;
  bool support_transfer_timeout = false;

  // V16 forward-placement touchdown event/sample this tick
  // -- V24's direct inputs.
  bool v16_forward_placement_count = false;
  double v16_forward_placement_sample = 0.0;

  // V24 bilateral state, re-derived post-update (same formula as
  // obs_builder.cpp's obs[55]/[56] and).
  double bilateral_imbalance_signed = 0.0;
  double imbalance_growth = 0.0;
};

// Advances EventMachineState by exactly one 50Hz POLICY TICK -- i.e. called
// ONCE per control step, after all kSubstepsPerTick (10) MuJoCo physics
// substeps for that tick have already run and mjData reflects the
// post-integration state (mirroring the source environment's V18.step(), which computes
// contact/GRF/orientation from `data` AFTER `mjx_env.step(..., n_substeps)`,
//). NEVER call this once per individual 2ms physics
// substep -- the event machine's step-count-based gates (e.g.
// v8_liftoff_no_contact_steps, v18_prep_steps) are counted in POLICY ticks,
// not physics substeps, exactly as the training-time `info["step"]` counter
// increments once per RL environment step (10 substeps), not once per
// substep.
EventMachineOutputs UpdateEventMachine(EventMachineState* state, const EventMachineInputs& in,
                                        const mjModel* m);

}  // namespace biped
