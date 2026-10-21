// UpdateEventMachine() is called EXACTLY ONCE PER 50Hz POLICY TICK, after the
// 10 MuJoCo physics substeps for that tick have already executed -- never
// once per 2ms physics substep. See event_machine.h's header comment on
// UpdateEventMachine for why (step-count-based gates are counted in policy
// ticks, matching the source environment's info["step"] increment cadence).
#include "biped_native/event_machine.h"

#include <algorithm>
#include <cmath>

namespace biped {

namespace {

double Clip(double x, double lo, double hi) { return std::min(std::max(x, lo), hi); }

// _bridge_phase: 0=PREP_SHIFT,1=UNLOAD,2=SWING_HOLD,
// 3=TOUCHDOWN,4=RECENTER.
int BridgePhase(double local_step) {
  const double b0 = kBridgePrepShiftSteps;
  const double b1 = b0 + kBridgeUnloadSteps;
  const double b2 = b1 + kBridgeSwingHoldSteps;
  const double b3 = b2 + kBridgeTouchdownSteps;
  int phase_id = 0;
  if (local_step >= b0) ++phase_id;
  if (local_step >= b1) ++phase_id;
  if (local_step >= b2) ++phase_id;
  if (local_step >= b3) ++phase_id;
  return std::min(std::max(phase_id, 0), 4);
}

// _stepping_phase: returns (phase_id, local_step,
// gait_phase) -- target_left/right omitted, not needed downstream (see
// event_machine.h header comment: target_left_swing/phase_id are reward-
// gating only, confirmed not to feed any persisted next_* field).
struct SteppingPhase {
  int phase_id;
  double local_step;
  double gait_phase;
};

SteppingPhase ComputeSteppingPhase(double step) {
  const double cycle_step = std::fmod(step, kCycleSteps);
  const bool is_left_half = cycle_step < kHalfCycleSteps;
  const double local_step = is_left_half ? cycle_step : cycle_step - kHalfCycleSteps;
  SteppingPhase out;
  out.phase_id = BridgePhase(local_step);
  out.local_step = local_step;
  out.gait_phase = cycle_step / kCycleSteps;
  return out;
}

// _v18_legacy_phase's gait_phase is just
// ComputeSteppingPhase(step).gait_phase -- the rest of the legacy dict
// (prep/swing/transfer/target_left_swing/...) is reward/control-law-only,
// not needed for the observation-state port.
double LegacyGaitPhase(double step) { return ComputeSteppingPhase(step).gait_phase; }

// _v18_event_phase's gait_phase computation only;
// the rest of that function's returned dict (target_left_swing,
// desired_*_contact, phase_id, ...) feeds only the control law / reward path,
// confirmed not to affect any persisted next_* state field.
double EventGaitPhase(double expected_side, bool transfer_pending, double age) {
  const double expected_left = (expected_side < -0.5) ? 1.0 : 0.0;
  const double expected_right = (expected_side > 0.5) ? 1.0 : 0.0;
  const double pending = transfer_pending ? 1.0 : 0.0;

  const double prep_steps = std::max(kV18PrepSteps, 1.0);
  const double swing_steps = std::max(kV18NominalSwingSteps, 1.0);
  const double transfer_steps = std::max(kV18NominalTransferSteps, 1.0);
  const double total_steps = prep_steps + swing_steps + transfer_steps;

  const double prep = (1.0 - pending) * ((age < prep_steps) ? 1.0 : 0.0);
  const double swing = (1.0 - pending) * ((age >= prep_steps) ? 1.0 : 0.0);
  const double transfer = pending;

  const double swing_age = std::max(age - prep_steps, 0.0);
  const double swing_progress_steps = std::min(swing_age, swing_steps);
  const double prep_progress_steps = std::min(age, prep_steps);
  const double prep_progress = prep_progress_steps / prep_steps;
  const double swing_progress = swing_progress_steps / swing_steps;

  const double prep_cycle_steps = prep_progress * prep_steps;
  const double swing_cycle_steps = prep_steps + swing_progress * swing_steps;
  const double transfer_cycle_steps = prep_steps + swing_steps + 0.5 * transfer_steps;
  const double event_cycle_steps =
      prep * prep_cycle_steps + swing * swing_cycle_steps + transfer * transfer_cycle_steps;
  const double event_half_progress = Clip(event_cycle_steps / total_steps, 0.0, 1.0);
  const double event_base = 0.5 * expected_right;
  return event_base + 0.5 * event_half_progress;
}

// select("gait_phase", ...): known ? event : legacy.
double ComputeGaitPhase(double step, double v9_expected_side, bool v9_transfer_pending,
                         double v9_steps_since_expected_side_set) {
  const bool known = std::fabs(v9_expected_side) > 0.5;
  if (!known) return LegacyGaitPhase(step);
  return EventGaitPhase(v9_expected_side, v9_transfer_pending, v9_steps_since_expected_side_set);
}

// _v9_expected_side_gates.
struct ExpectedSideGates {
  double gate_left;
  double gate_right;
};

ExpectedSideGates V9ExpectedSideGates(double expected_side, bool transfer_pending) {
  const double transfer_open = transfer_pending ? 0.0 : 1.0;
  const double unknown = (std::fabs(expected_side) < 0.5) ? 1.0 : 0.0;
  ExpectedSideGates g;
  g.gate_left = transfer_open * std::max(unknown, (expected_side < -0.5) ? 1.0 : 0.0);
  g.gate_right = transfer_open * std::max(unknown, (expected_side > 0.5) ? 1.0 : 0.0);
  return g;
}

// _v8_liftoff_touchdown_state, state-relevant
// outputs only (raw_liftoff/raw_touchdown are diagnostics-only in Python and
// are not consumed anywhere in the persisted-state path; omitted here).
struct LiftoffTouchdownResult {
  double qualified_liftoff;
  double qualified_touchdown;
  double next_no_contact_streak;
  double next_max_air_height;
  double next_air_qualified;
  double next_contact_streak;
  double next_steps_since;
};

LiftoffTouchdownResult V8LiftoffTouchdownState(bool contact, bool opposite_contact,
                                                double height_gain, double valid_ground_load_gate,
                                                double no_contact_streak_in, double max_air_in,
                                                double air_qualified_in, double contact_streak_in,
                                                double steps_since_in) {
  const double contact_f = contact ? 1.0 : 0.0;
  const double no_contact_streak = contact_f > 0.5 ? 0.0 : no_contact_streak_in + 1.0;
  const double previous_max = max_air_in;
  const double prev_air_qualified = air_qualified_in;
  const double max_air_height = (contact_f < 0.5)
                                     ? std::max(previous_max, height_gain)
                                     : (prev_air_qualified > 0.5 ? previous_max : 0.0);
  const double contact_streak = (contact_f > 0.5) ? contact_streak_in + 1.0 : 0.0;

  const double qualified_liftoff =
      ((prev_air_qualified < 0.5) ? 1.0 : 0.0) *
      ((no_contact_streak >= kV8LiftoffNoContactSteps) ? 1.0 : 0.0) *
      ((max_air_height >= kV8LiftoffHeightGain) ? 1.0 : 0.0) * (opposite_contact ? 1.0 : 0.0) *
      ((valid_ground_load_gate > 0.999) ? 1.0 : 0.0);
  const double air_qualified = std::max(prev_air_qualified, qualified_liftoff);
  const double steps_since = steps_since_in;
  const double qualified_touchdown =
      air_qualified * ((contact_streak == kV8TouchdownContactConfirmSteps) ? 1.0 : 0.0) *
      ((steps_since >= kV8MinStepsBetweenQualifiedTouchdowns) ? 1.0 : 0.0);

  LiftoffTouchdownResult r;
  r.qualified_liftoff = qualified_liftoff;
  r.qualified_touchdown = qualified_touchdown;
  r.next_no_contact_streak = no_contact_streak;
  r.next_max_air_height = (qualified_touchdown > 0.5) ? 0.0 : max_air_height;
  r.next_air_qualified = (qualified_touchdown > 0.5) ? 0.0 : air_qualified;
  r.next_contact_streak = contact_streak;
  r.next_steps_since = (qualified_touchdown > 0.5) ? 0.0 : steps_since + 1.0;
  return r;
}

}  // namespace

void ResetEventMachineState(EventMachineState* state, const mjModel* m, const mjData* d,
                             int left_foot_geom_id, int right_foot_geom_id, double target_height) {
  *state = EventMachineState{};
  state->target_height = target_height;
  state->left_foot_z_nominal = d->geom_xpos[left_foot_geom_id * 3 + 2];
  state->right_foot_z_nominal = d->geom_xpos[right_foot_geom_id * 3 + 2];
  (void)m;
}

EventMachineOutputs UpdateEventMachine(EventMachineState* s, const EventMachineInputs& in,
                                        const mjModel* m) {
  // --- GRF / load fractions / gates (_v6_components; ---
  // --- _v7_components's stability gates) ------------
  double total_model_mass = 0.0;
  for (int i = 0; i < m->nbody; ++i) total_model_mass += m->body_mass[i];
  const double gravity_magnitude = std::fabs(m->opt.gravity[2]);
  const double expected_weight = total_model_mass * gravity_magnitude;

  const double total_load = in.fz_left + in.fz_right;
  const double total_load_fraction =
      Clip(total_load / (expected_weight + kLoadTransferForceEpsilon), 0.0, 2.0);
  const double lambda_left = Clip(in.fz_left / (total_load + kLoadTransferForceEpsilon), 0.0, 1.0);
  const double lambda_right =
      Clip(in.fz_right / (total_load + kLoadTransferForceEpsilon), 0.0, 1.0);
  const double valid_ground_load_gate =
      Clip(total_load_fraction / kValidGroundLoadFractionScale, 0.0, 1.0);

  const double left_height_gain = in.left_foot_pos_z - s->left_foot_z_nominal;
  const double right_height_gain = in.right_foot_pos_z - s->right_foot_z_nominal;

  const double upright_gate =
      std::exp(-(in.roll * in.roll + in.pitch * in.pitch) / (0.25 * 0.25));
  const double root_height_gate =
      std::exp(-(in.root_z - s->target_height) * (in.root_z - s->target_height) / kHeightSigma);
  const double angular_stability_gate =
      std::exp(-(in.gyro_x * in.gyro_x + in.gyro_y * in.gyro_y) / (1.5 * 1.5));
  // (combined stability_gate omitted: support_transfer_completion below gates
  // on upright_gate/root_height_gate/angular_stability_gate individually,
  //; the combined product only feeds reward-only
  // stable_single_support/post_touchdown_stability/support_transfer_progress.)

  // --- V8 per-side liftoff/touchdown state (_v8_liftoff_touchdown_state) --
  const LiftoffTouchdownResult left = V8LiftoffTouchdownState(
      in.left_contact, in.right_contact, left_height_gain,
      valid_ground_load_gate, s->v8_consecutive_no_contact_steps_left,
      s->v8_max_height_gain_during_air_left, s->v8_air_qualified_left,
      s->v8_consecutive_contact_steps_left, s->v8_steps_since_last_qualified_touchdown_left);
  const LiftoffTouchdownResult right = V8LiftoffTouchdownState(
      in.right_contact, in.left_contact, right_height_gain,
      valid_ground_load_gate, s->v8_consecutive_no_contact_steps_right,
      s->v8_max_height_gain_during_air_right, s->v8_air_qualified_right,
      s->v8_consecutive_contact_steps_right, s->v8_steps_since_last_qualified_touchdown_right);

  // --- V8 combine ---
  const double qualified_touchdown = std::max(left.qualified_touchdown, right.qualified_touchdown);
  const double current_side = right.qualified_touchdown - left.qualified_touchdown;
  const double previous_side_v8 = s->v8_last_qualified_touchdown_side;
  const double has_previous_side = (std::fabs(previous_side_v8) > 0.5) ? 1.0 : 0.0;
  const double valid_current_side =
      qualified_touchdown * ((std::fabs(current_side) > 0.5) ? 1.0 : 0.0);
  const double first_qualified_touchdown = valid_current_side * (1.0 - has_previous_side);
  const double qualified_alternating_touchdown =
      valid_current_side * has_previous_side * ((current_side == -previous_side_v8) ? 1.0 : 0.0);
  const double v8_raw_eligible_touchdown =
      std::max(first_qualified_touchdown, qualified_alternating_touchdown);
  const double next_v8_last_qualified_touchdown_side =
      (valid_current_side > 0.5) ? current_side : previous_side_v8;

  // --- V9 expected-side gating (_v9_expected_side_gates, _v9_components) --
  const ExpectedSideGates gates = V9ExpectedSideGates(s->v9_expected_side, s->v9_transfer_pending > 0.5);
  const double eligible_touchdown_left = v8_raw_eligible_touchdown * left.qualified_touchdown;
  const double eligible_touchdown_right = v8_raw_eligible_touchdown * right.qualified_touchdown;
  const double eligible_touchdown =
      gates.gate_left * eligible_touchdown_left + gates.gate_right * eligible_touchdown_right;
  const double recent_eligible_touchdown =
      std::max(eligible_touchdown, (s->v8_recent_eligible_touchdown_steps > 0.0) ? 1.0 : 0.0);
  const double next_recent_eligible_touchdown_steps =
      (eligible_touchdown > 0.5) ? (kV8RecentEligibleTouchdownSteps - 1.0)
                                  : std::max(s->v8_recent_eligible_touchdown_steps - 1.0, 0.0);

  const double landed_side =
      (eligible_touchdown > 0.5) ? current_side : s->v8_support_transfer_landed_side;
  const double previous_support_side =
      (eligible_touchdown > 0.5) ? -current_side : s->v8_support_transfer_previous_support_side;
  const double landed_left = (landed_side < -0.5) ? 1.0 : 0.0;
  const double landed_right = (landed_side > 0.5) ? 1.0 : 0.0;
  const double previous_support_left = (previous_support_side < -0.5) ? 1.0 : 0.0;
  const double previous_support_right = (previous_support_side > 0.5) ? 1.0 : 0.0;
  const double landed_load_fraction = landed_left * lambda_left + landed_right * lambda_right;
  const double previous_support_load_fraction =
      previous_support_left * lambda_left + previous_support_right * lambda_right;
  const double landed_contact =
      landed_left * (in.left_contact ? 1.0 : 0.0) + landed_right * (in.right_contact ? 1.0 : 0.0);

  const double current_transfer_steps =
      (eligible_touchdown > 0.5) ? kV8SupportTransferSteps : s->v8_support_transfer_window_steps;
  const double transfer_completed_prev =
      (eligible_touchdown > 0.5) ? 0.0 : s->v8_support_transfer_completed;
  const double transfer_timeout_prev =
      (eligible_touchdown > 0.5) ? 0.0 : s->v8_support_transfer_timeout_fired;
  const double active_transfer_window = (current_transfer_steps > 0.0) ? 1.0 : 0.0;
  // (landed_load_gate/previous_support_release_gate omitted: they only feed
  // the source environment's reward-only support_transfer_progress,
  // which is not part of the persisted-state path.)
  const double support_transfer_completion =
      active_transfer_window * (1.0 - transfer_completed_prev) *
      ((landed_load_fraction >= 0.55) ? 1.0 : 0.0) *
      ((previous_support_load_fraction <= 0.45) ? 1.0 : 0.0) * landed_contact *
      ((valid_ground_load_gate > 0.999) ? 1.0 : 0.0) * ((upright_gate >= 0.70) ? 1.0 : 0.0) *
      ((root_height_gate >= 0.70) ? 1.0 : 0.0) * ((angular_stability_gate >= 0.50) ? 1.0 : 0.0);
  const double support_transfer_timeout =
      (1.0 - eligible_touchdown) * active_transfer_window *
      ((current_transfer_steps <= 1.0) ? 1.0 : 0.0) * (1.0 - transfer_completed_prev) *
      (1.0 - transfer_timeout_prev) * (1.0 - support_transfer_completion);
  const double next_support_transfer_window_steps =
      (eligible_touchdown > 0.5)
          ? (kV8SupportTransferSteps - 1.0)
          : ((support_transfer_timeout > 0.5)
                 ? 0.0
                 : std::max(s->v8_support_transfer_window_steps - 1.0, 0.0));
  const double next_support_transfer_completed =
      std::max(transfer_completed_prev, support_transfer_completion);
  const double next_support_transfer_timeout_fired =
      std::max(transfer_timeout_prev, support_transfer_timeout);

  const double left_transfer_completion = support_transfer_completion * landed_left;
  const double right_transfer_completion = support_transfer_completion * landed_right;

  // v9_expected_side / v9_transfer_pending / v9_pending_landed_side update.
  const double expected_side_in = s->v9_expected_side;
  const double pending_landed_side =
      (eligible_touchdown > 0.5) ? current_side : s->v9_pending_landed_side;
  const double completed_landed_side = landed_side;
  const double timeout_from_unknown =
      support_transfer_timeout * ((std::fabs(expected_side_in) < 0.5) ? 1.0 : 0.0);
  const double next_v9_expected_side =
      (support_transfer_completion > 0.5)
          ? -completed_landed_side
          : ((timeout_from_unknown > 0.5) ? landed_side : expected_side_in);
  const double next_v9_transfer_pending =
      ((support_transfer_completion > 0.5) || (support_transfer_timeout > 0.5))
          ? 0.0
          : ((eligible_touchdown > 0.5) ? 1.0 : s->v9_transfer_pending);
  const double next_v9_pending_landed_side =
      (next_v9_transfer_pending > 0.5) ? pending_landed_side : 0.0;

  // v9_previous_qualified_touchdown_side / same-side streak update.
  const double previous_qualified_side = s->v9_previous_qualified_touchdown_side;
  const double same_side_qualified_touchdown =
      qualified_touchdown * ((std::fabs(previous_qualified_side) > 0.5) ? 1.0 : 0.0) *
      ((current_side == previous_qualified_side) ? 1.0 : 0.0);
  const double next_v9_current_same_side_qualified_touchdown_streak =
      (qualified_touchdown > 0.5)
          ? ((same_side_qualified_touchdown > 0.5)
                 ? s->v9_current_same_side_qualified_touchdown_streak + 1.0
                 : 1.0)
          : s->v9_current_same_side_qualified_touchdown_streak;
  const double next_v9_max_same_side_qualified_touchdown_streak =
      std::max(s->v9_max_same_side_qualified_touchdown_streak,
                next_v9_current_same_side_qualified_touchdown_streak);
  const double next_v9_previous_qualified_touchdown_side =
      (qualified_touchdown > 0.5) ? current_side : previous_qualified_side;

  // v9_steps_since_expected_side_set update.
  const double known_expected = (std::fabs(expected_side_in) > 0.5) ? 1.0 : 0.0;
  const double active_expected_wait = known_expected * ((s->v9_transfer_pending > 0.5) ? 0.0 : 1.0);
  const double next_v9_steps_since_expected_side_set =
      (support_transfer_completion > 0.5)
          ? 0.0
          : ((timeout_from_unknown > 0.5)
                 ? 0.0
                 : s->v9_steps_since_expected_side_set + active_expected_wait);

  // --- V16 attempt/placement -----
  const ExpectedSideGates expected_gates = gates;  // same info, same formula
  const double expected_known = (std::fabs(s->v9_expected_side) > 0.5) ? 1.0 : 0.0;
  const double expected_left = expected_known * expected_gates.gate_left;
  const double expected_right = expected_known * expected_gates.gate_right;

  const double active_prev = (s->v16_attempt_active > 0.5) ? 1.0 : 0.0;
  const double inactive = 1.0 - active_prev;
  const double start_left = inactive * expected_left * left.qualified_liftoff;
  const double start_right = inactive * expected_right * right.qualified_liftoff;
  const double start = Clip(start_left + start_right, 0.0, 1.0);
  const double start_side = start_right - start_left;

  const double attempt_side = (start > 0.5) ? start_side : s->v16_attempt_side;
  const double active = Clip(active_prev + start, 0.0, 1.0);
  const double attempt_left = active * ((attempt_side < -0.5) ? 1.0 : 0.0);
  const double attempt_right = active * ((attempt_side > 0.5) ? 1.0 : 0.0);

  const double left_c = in.left_contact ? 1.0 : 0.0;
  const double right_c = in.right_contact ? 1.0 : 0.0;
  const double physical_single_support =
      attempt_left * (1.0 - left_c) * right_c + attempt_right * (1.0 - right_c) * left_c;
  const double single_support_steps =
      active_prev * s->v16_single_support_steps + active * physical_single_support;
  const double single_support_target = std::max(kV16SingleSupportTargetSteps, 1.0e-6);
  const double single_support_quality = Clip(single_support_steps / single_support_target, 0.0, 1.0);

  const double touchdown_left = attempt_left * left.qualified_touchdown;
  const double touchdown_right = attempt_right * right.qualified_touchdown;
  const double touchdown = Clip(touchdown_left + touchdown_right, 0.0, 1.0);
  const double d_forward = attempt_left * (in.left_foot_pos_x - in.right_foot_pos_x) +
                            attempt_right * (in.right_foot_pos_x - in.left_foot_pos_x);
  const double forward_placement = std::max(d_forward, 0.0);
  const double placement_target = std::max(kV16ForwardPlacementTarget, 1.0e-6);
  const double forward_placement_quality = Clip(forward_placement / placement_target, 0.0, 1.0);
  const double latched_forward_placement =
      (touchdown > 0.5) ? forward_placement : s->v16_forward_placement;
  const double latched_forward_quality =
      (touchdown > 0.5) ? forward_placement_quality : s->v16_forward_placement_quality;
  const double touchdown_latched =
      Clip(active * s->v16_touchdown_latched + touchdown, 0.0, 1.0);

  const double side_consistent_completion =
      attempt_left * left_transfer_completion + attempt_right * right_transfer_completion;
  const double valid_completion = active * touchdown_latched * side_consistent_completion;

  const double side_consistent_timeout =
      support_transfer_timeout * (attempt_left * ((landed_side < -0.5) ? 1.0 : 0.0) +
                                   attempt_right * ((landed_side > 0.5) ? 1.0 : 0.0));
  const double conflicting_liftoff =
      active * (attempt_left * right.qualified_liftoff + attempt_right * left.qualified_liftoff);
  const double cancel =
      (1.0 - valid_completion) * Clip(side_consistent_timeout + conflicting_liftoff, 0.0, 1.0);
  const double clear = Clip(valid_completion + cancel, 0.0, 1.0);
  const double next_active = active * (1.0 - clear);

  const double next_v16_attempt_side = next_active * attempt_side;
  const double next_v16_single_support_steps = next_active * single_support_steps;
  const double next_v16_touchdown_latched = next_active * touchdown_latched;
  const double next_v16_forward_placement = next_active * latched_forward_placement;
  const double next_v16_forward_placement_quality = next_active * latched_forward_quality;
  const double v16_forward_placement_count = touchdown;
  const double v16_forward_placement_sample = touchdown * forward_placement;

  // --- V24 bilateral memory ------
  constexpr double kEps = kBilateralEpsilonD;
  const double v24_touchdown = v16_forward_placement_count;
  const double v24_attempt_side_read = s->v16_attempt_side;  // OLD state, per
  const double is_left = v24_touchdown * ((v24_attempt_side_read < -0.5) ? 1.0 : 0.0);
  const double is_right = v24_touchdown * ((v24_attempt_side_read > 0.5) ? 1.0 : 0.0);
  const double placement_sample = v16_forward_placement_sample;

  const double next_v24_last_left_placement =
      (is_left > 0.5) ? placement_sample : s->v24_last_left_placement;
  const double next_v24_last_right_placement =
      (is_right > 0.5) ? placement_sample : s->v24_last_right_placement;
  const double next_v24_left_valid = Clip(s->v24_left_valid + is_left, 0.0, 1.0);
  const double next_v24_right_valid = Clip(s->v24_right_valid + is_right, 0.0, 1.0);
  const double next_v24_pair_valid = next_v24_left_valid * next_v24_right_valid;
  const double previous_pair_valid = s->v24_pair_valid;

  const double signed_imbalance =
      next_v24_pair_valid * ((next_v24_last_right_placement - next_v24_last_left_placement) /
                              (next_v24_last_right_placement + next_v24_last_left_placement + kEps));
  const double abs_imbalance_current = std::fabs(signed_imbalance);
  const double abs_imbalance_previous = s->v24_abs_imbalance_previous;
  const double growth_gate = v24_touchdown * next_v24_pair_valid * previous_pair_valid;
  const double imbalance_growth =
      growth_gate * Clip(std::max(abs_imbalance_current - abs_imbalance_previous, 0.0) / 0.05, 0.0, 1.0);
  const double next_v24_abs_imbalance_previous = abs_imbalance_current;

  // --- Commit persisted state -------
  s->v8_consecutive_no_contact_steps_left = left.next_no_contact_streak;
  s->v8_consecutive_no_contact_steps_right = right.next_no_contact_streak;
  s->v8_max_height_gain_during_air_left = left.next_max_air_height;
  s->v8_max_height_gain_during_air_right = right.next_max_air_height;
  s->v8_air_qualified_left = left.next_air_qualified;
  s->v8_air_qualified_right = right.next_air_qualified;
  s->v8_consecutive_contact_steps_left = left.next_contact_streak;
  s->v8_consecutive_contact_steps_right = right.next_contact_streak;
  s->v8_steps_since_last_qualified_touchdown_left = left.next_steps_since;
  s->v8_steps_since_last_qualified_touchdown_right = right.next_steps_since;
  s->v8_last_qualified_touchdown_side = next_v8_last_qualified_touchdown_side;

  // NOTE: these "v8_" named fields are written from V9's (not V8's) formulas
  // once V9 is present in the MRO -- see event_machine.h header comment.
  s->v8_recent_eligible_touchdown_steps = next_recent_eligible_touchdown_steps;
  s->v8_support_transfer_window_steps = next_support_transfer_window_steps;
  s->v8_support_transfer_landed_side = landed_side;
  s->v8_support_transfer_previous_support_side = previous_support_side;
  s->v8_support_transfer_completed = next_support_transfer_completed;
  s->v8_support_transfer_timeout_fired = next_support_transfer_timeout_fired;

  s->v9_expected_side = next_v9_expected_side;
  s->v9_transfer_pending = next_v9_transfer_pending;
  s->v9_pending_landed_side = next_v9_pending_landed_side;
  s->v9_previous_qualified_touchdown_side = next_v9_previous_qualified_touchdown_side;
  s->v9_current_same_side_qualified_touchdown_streak =
      next_v9_current_same_side_qualified_touchdown_streak;
  s->v9_max_same_side_qualified_touchdown_streak =
      next_v9_max_same_side_qualified_touchdown_streak;
  s->v9_steps_since_expected_side_set = next_v9_steps_since_expected_side_set;

  s->v16_attempt_active = next_active;
  s->v16_attempt_side = next_v16_attempt_side;
  s->v16_single_support_steps = next_v16_single_support_steps;
  s->v16_touchdown_latched = next_v16_touchdown_latched;
  s->v16_forward_placement = next_v16_forward_placement;
  s->v16_forward_placement_quality = next_v16_forward_placement_quality;

  s->v24_last_left_placement = next_v24_last_left_placement;
  s->v24_last_right_placement = next_v24_last_right_placement;
  s->v24_left_valid = next_v24_left_valid;
  s->v24_right_valid = next_v24_right_valid;
  s->v24_pair_valid = next_v24_pair_valid;
  s->v24_abs_imbalance_previous = next_v24_abs_imbalance_previous;

  // --- gait_phase for the NEXT observation, computed from the JUST-UPDATED
  // --- v9_* state (: next_legacy/next_phase use
  // --- new_info, i.e. post-update state) ----------------------------------
  s->step = s->step + 1.0;
  const double gait_phase = ComputeGaitPhase(s->step, s->v9_expected_side,
                                              s->v9_transfer_pending > 0.5,
                                              s->v9_steps_since_expected_side_set);

  EventMachineOutputs out;
  out.gait_phase = gait_phase;
  out.qualified_liftoff_left = left.qualified_liftoff > 0.5;
  out.qualified_liftoff_right = right.qualified_liftoff > 0.5;
  out.qualified_touchdown_left = left.qualified_touchdown > 0.5;
  out.qualified_touchdown_right = right.qualified_touchdown > 0.5;
  out.eligible_touchdown = eligible_touchdown > 0.5;
  out.support_transfer_completion = support_transfer_completion > 0.5;
  out.support_transfer_timeout = support_transfer_timeout > 0.5;
  out.v16_forward_placement_count = v16_forward_placement_count > 0.5;
  out.v16_forward_placement_sample = v16_forward_placement_sample;
  out.bilateral_imbalance_signed = signed_imbalance;
  out.imbalance_growth = imbalance_growth;
  // recent_eligible_touchdown/single_support_quality are Python intermediates
  // that only feed reward-only outputs (touchdown_load_acceptance,
  // v16_valid_step) -- computed here for auditability against the source environment but
  // not consumed further.
  (void)recent_eligible_touchdown;
  (void)single_support_quality;
  return out;
}

}  // namespace biped
