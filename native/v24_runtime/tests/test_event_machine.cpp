// Synthetic (fixture-free) unit tests for event_machine.cpp. Every input
// sequence is hand-constructed to walk the V8/V9/V16/V24 gates through their
// exact documented thresholds; no recorded rollout data is read.
//
// The synthetic model (synthetic_model.h) is used only to read body masses and
// gravity, needed to compute expected_weight the same way UpdateEventMachine
// does; no physics stepping is involved.
//
// Usage: test_event_machine
#include <cmath>
#include <cstdio>
#include <string>

#include <mujoco/mujoco.h>

#include "biped_native/event_machine.h"
#include "synthetic_model.h"

namespace {

// Test-local, synthetic root-height target (the real value is a private binding).
constexpr double kTestTargetHeight = 0.5;

int g_checks = 0;
int g_failures = 0;

void Check(bool cond, const std::string& msg) {
  ++g_checks;
  if (!cond) {
    ++g_failures;
    std::printf("[test_event_machine] FAIL: %s\n", msg.c_str());
  }
}

void CheckNear(double actual, double expected, double tol, const std::string& msg) {
  Check(std::fabs(actual - expected) <= tol,
        msg + " (actual=" + std::to_string(actual) + " expected=" + std::to_string(expected) + ")");
}

// One 50Hz-tick's worth of synthetic physical inputs. Roll/pitch/gyro are
// kept at 0 and root_z at the test target height throughout so upright_gate/
// root_height_gate/angular_stability_gate are always ~1 -- this test is
// about the event/contact/GRF-driven state machine, not the stability gates.
biped::EventMachineInputs MakeInputs(bool contact_l, bool contact_r, double z_l, double z_r,
                                       double x_l, double x_r, double fz_l, double fz_r) {
  biped::EventMachineInputs in;
  in.left_contact = contact_l;
  in.right_contact = contact_r;
  in.left_foot_pos_z = z_l;
  in.right_foot_pos_z = z_r;
  in.left_foot_pos_x = x_l;
  in.right_foot_pos_x = x_r;
  in.fz_left = fz_l;
  in.fz_right = fz_r;
  in.roll = 0.0;
  in.pitch = 0.0;
  in.gyro_x = 0.0;
  in.gyro_y = 0.0;
  in.root_z = kTestTargetHeight;
  return in;
}

}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  mjModel* m = synthetic::MakeModel();
  double total_model_mass = 0.0;
  for (int i = 0; i < m->nbody; ++i) total_model_mass += m->body_mass[i];
  const double expected_weight = total_model_mass * std::fabs(m->opt.gravity[2]);
  std::printf("[test_event_machine] expected_weight=%.4f N\n", expected_weight);

  // ===========================================================================
  // Test 0: bootstrap state defaults, checked on a FRESHLY-CONSTRUCTED state
  // (before any UpdateEventMachine call) -- the two
  // v8_steps_since_last_qualified_touchdown_{side} fields must default to
  // kV8MinStepsBetweenQualifiedTouchdowns(=8), not 0 .
  // ===========================================================================
  {
    biped::EventMachineState s0{};
    CheckNear(s0.v8_steps_since_last_qualified_touchdown_left, 8.0, 1e-12,
              "bootstrap: v8_steps_since_last_qualified_touchdown_left must default to 8");
    CheckNear(s0.v8_steps_since_last_qualified_touchdown_right, 8.0, 1e-12,
              "bootstrap: v8_steps_since_last_qualified_touchdown_right must default to 8");
    CheckNear(s0.v9_expected_side, 0.0, 1e-12, "bootstrap: v9_expected_side must default to 0");
    CheckNear(s0.v24_pair_valid, 0.0, 1e-12, "bootstrap: v24_pair_valid must default to 0");
  }

  // ===========================================================================
  // Test 1: bootstrap gait_phase -- with v9_expected_side unknown, gait_phase
  // must fall back to the deterministic step-count schedule (_stepping_phase),
  // exactly (step mod 280)/280, regardless of physical inputs.
  // ===========================================================================
  {
    biped::EventMachineState s{};
  s.target_height = kTestTargetHeight;
    const biped::EventMachineInputs standing =
        MakeInputs(true, true, 0.0, 0.0, 0.0, 0.0, expected_weight / 2.0, expected_weight / 2.0);
    biped::EventMachineOutputs out{};
    for (int i = 0; i < 5; ++i) out = biped::UpdateEventMachine(&s, standing, m);
    CheckNear(out.gait_phase, 5.0 / 280.0, 1e-9,
              "bootstrap gait_phase must equal legacy (step mod 280)/280");
    Check(!out.eligible_touchdown, "standing ticks must not produce a spurious eligible_touchdown");
    CheckNear(s.v9_expected_side, 0.0, 1e-12, "v9_expected_side must remain unknown while standing");
  }

  // ===========================================================================
  // Test 2: full liftoff->touchdown->transfer-completion cycle qualifies on
  // exactly the documented ticks (no_contact_streak==4, contact_streak==3),
  // and a bootstrap (side-unknown) completion still flips v9_expected_side
  // to the side OPPOSITE the one that just landed (confirmed sign
  // convention: expected_side < -0.5 -> LEFT, > 0.5 -> RIGHT).
  // ===========================================================================
  biped::EventMachineState s{};
  s.target_height = kTestTargetHeight;
  {
    const double kAirZ = 0.02, kGroundZ = 0.0;
    biped::EventMachineOutputs out{};
    // 4 no-contact ticks (left airborne, right planted).
    for (int i = 1; i <= 4; ++i) {
      biped::EventMachineInputs in =
          MakeInputs(false, true, kAirZ, kGroundZ, -0.05, 0.0, 0.0, expected_weight);
      out = biped::UpdateEventMachine(&s, in, m);
      if (i < 4) {
        Check(!out.qualified_liftoff_left,
              "qualified_liftoff_left must not fire before no_contact_streak==4 (tick " +
                  std::to_string(i) + ")");
      }
    }
    Check(out.qualified_liftoff_left, "qualified_liftoff_left must fire on exactly the 4th "
                                       "no-contact tick");

    // 3 contact-confirm ticks (left touches down); on the 3rd, skew GRF so
    // landed_load_fraction>=0.55 and previous_support<=0.45 to also complete
    // the transfer on this same tick.
    for (int i = 1; i <= 3; ++i) {
      const double fz_l = (i == 3) ? 0.8 * expected_weight : 0.5 * expected_weight;
      const double fz_r = (i == 3) ? 0.2 * expected_weight : 0.5 * expected_weight;
      biped::EventMachineInputs in = MakeInputs(true, true, kGroundZ, kGroundZ, -0.05, 0.0, fz_l, fz_r);
      out = biped::UpdateEventMachine(&s, in, m);
      if (i < 3) {
        Check(!out.qualified_touchdown_left,
              "qualified_touchdown_left must not fire before contact_streak==3 (tick " +
                  std::to_string(i) + ")");
      }
    }
    Check(out.qualified_touchdown_left, "qualified_touchdown_left must fire on exactly the 3rd "
                                         "contact-confirm tick");
    Check(out.eligible_touchdown, "the first-ever qualified touchdown must be eligible "
                                   "(bootstrap: both side gates open)");
    Check(out.support_transfer_completion,
          "support_transfer_completion must fire same-tick given the engineered load split");
    CheckNear(s.v9_expected_side, 1.0, 1e-9,
              "a bootstrap LEFT completion must set v9_expected_side to RIGHT (+1)");
    Check(!out.v16_forward_placement_count,
          "no V16 placement should be recorded for a liftoff that started while "
          "v9_expected_side was still unknown (V16 attempts only start once a side is known)");
  }

  // ===========================================================================
  // Test 3: three-completion bilateral sequence (right, then left, then
  // right again) -- validates V16 placement recording once expected_side is
  // known, V24 pair-valid monotonicity, the signed-imbalance sign
  // convention, and the "first pair scores zero growth" guarantee.
  // ===========================================================================
  double right_placement_1 = 0.0, left_placement_1 = 0.0, right_placement_2 = 0.0;
  {
    // --- Completion A: RIGHT liftoff+touchdown (expected_side is RIGHT from
    // --- test 2, so this is a legitimate expected-side attempt). Swing foot
    // --- (right) placed forward of stance foot (left) at x=0.09.
    const double kAirZ = 0.02, kGroundZ = 0.0;
    biped::EventMachineOutputs out{};
    for (int i = 1; i <= 4; ++i) {
      biped::EventMachineInputs in = MakeInputs(true, false, kGroundZ, kAirZ, 0.0, 0.09, expected_weight, 0.0);
      out = biped::UpdateEventMachine(&s, in, m);
    }
    Check(out.qualified_liftoff_right, "completion A: qualified_liftoff_right must fire on tick 4");
    for (int i = 1; i <= 3; ++i) {
      const double fz_l = (i == 3) ? 0.2 * expected_weight : 0.5 * expected_weight;
      const double fz_r = (i == 3) ? 0.8 * expected_weight : 0.5 * expected_weight;
      biped::EventMachineInputs in = MakeInputs(true, true, kGroundZ, kGroundZ, 0.0, 0.09, fz_l, fz_r);
      out = biped::UpdateEventMachine(&s, in, m);
    }
    Check(out.qualified_touchdown_right, "completion A: qualified_touchdown_right must fire on tick 3");
    Check(out.support_transfer_completion, "completion A: transfer must complete same-tick");
    Check(out.v16_forward_placement_count,
          "completion A: V16 placement MUST be recorded (expected_side was RIGHT, matching "
          "this liftoff's side)");
    right_placement_1 = out.v16_forward_placement_sample;
    Check(right_placement_1 > 0.0, "completion A: forward placement sample must be positive "
                                    "(swing foot landed ahead of stance foot)");
    CheckNear(s.v24_right_valid, 1.0, 1e-9, "completion A: v24_right_valid must become 1");
    CheckNear(s.v24_pair_valid, 0.0, 1e-9,
              "completion A: v24_pair_valid must still be 0 (left side not yet valid)");
    CheckNear(out.imbalance_growth, 0.0, 1e-12, "completion A: imbalance_growth must be 0 (no valid pair yet)");
    CheckNear(s.v9_expected_side, -1.0, 1e-9,
              "completion A: v9_expected_side must flip to LEFT (-1) after a RIGHT completion");
  }
  {
    // --- Completion B: LEFT liftoff+touchdown -- forms the FIRST valid pair.
    const double kAirZ = 0.02, kGroundZ = 0.0;
    biped::EventMachineOutputs out{};
    for (int i = 1; i <= 4; ++i) {
      biped::EventMachineInputs in = MakeInputs(false, true, kAirZ, kGroundZ, 0.10, 0.0, 0.0, expected_weight);
      out = biped::UpdateEventMachine(&s, in, m);
    }
    for (int i = 1; i <= 3; ++i) {
      const double fz_l = (i == 3) ? 0.8 * expected_weight : 0.5 * expected_weight;
      const double fz_r = (i == 3) ? 0.2 * expected_weight : 0.5 * expected_weight;
      biped::EventMachineInputs in = MakeInputs(true, true, kGroundZ, kGroundZ, 0.10, 0.0, fz_l, fz_r);
      out = biped::UpdateEventMachine(&s, in, m);
    }
    Check(out.v16_forward_placement_count, "completion B: V16 placement MUST be recorded "
                                            "(expected_side was LEFT, matching this liftoff)");
    left_placement_1 = out.v16_forward_placement_sample;
    Check(left_placement_1 > 0.0, "completion B: forward placement sample must be positive");
    CheckNear(s.v24_left_valid, 1.0, 1e-9, "completion B: v24_left_valid must become 1");
    CheckNear(s.v24_pair_valid, 1.0, 1e-9,
              "completion B: v24_pair_valid must become 1 -- the FIRST valid pair");
    CheckNear(out.imbalance_growth, 0.0, 1e-12,
              "completion B: imbalance_growth MUST be 0 on the first-ever completed pair (growth "
              "is gated on previous_pair_valid, which was false going into this tick) -- this is "
              "the V24 bilateral-memory guarantee");
    const double expected_signed_imbalance =
        (right_placement_1 - left_placement_1) /
        (right_placement_1 + left_placement_1 + biped::kBilateralEpsilonD);
    CheckNear(out.bilateral_imbalance_signed, expected_signed_imbalance, 1e-9,
              "signed bilateral imbalance must equal (right-left)/(right+left+eps) once "
              "pair_valid, matching obs_builder.cpp's obs[55] formula");
    CheckNear(s.v9_expected_side, 1.0, 1e-9, "completion B: v9_expected_side must flip back to RIGHT");
  }
  {
    // --- Completion C: RIGHT liftoff+touchdown again -- now BOTH sides were
    // --- already valid going in, so growth scoring must be active.
    const double kAirZ = 0.02, kGroundZ = 0.0;
    biped::EventMachineOutputs out{};
    for (int i = 1; i <= 4; ++i) {
      biped::EventMachineInputs in = MakeInputs(true, false, kGroundZ, kAirZ, 0.0, 0.30, expected_weight, 0.0);
      out = biped::UpdateEventMachine(&s, in, m);
    }
    for (int i = 1; i <= 3; ++i) {
      const double fz_l = (i == 3) ? 0.2 * expected_weight : 0.5 * expected_weight;
      const double fz_r = (i == 3) ? 0.8 * expected_weight : 0.5 * expected_weight;
      biped::EventMachineInputs in = MakeInputs(true, true, kGroundZ, kGroundZ, 0.0, 0.30, fz_l, fz_r);
      out = biped::UpdateEventMachine(&s, in, m);
    }
    Check(out.v16_forward_placement_count, "completion C: V16 placement must be recorded");
    right_placement_2 = out.v16_forward_placement_sample;
    Check(right_placement_2 > right_placement_1,
          "completion C: this test engineered a larger forward placement (x=0.30 swing vs "
          "0.09 previously) so the imbalance delta is unambiguous");
    CheckNear(s.v24_pair_valid, 1.0, 1e-9, "completion C: pair_valid must remain 1 (monotonic)");
    Check(out.imbalance_growth > 0.0,
          "completion C: imbalance_growth must be > 0 now that BOTH left_valid and right_valid "
          "were already true going into this tick (unlike completion B, where "
          "previous_pair_valid was still false) and the placement delta grew");
  }

  std::printf("[test_event_machine] %d checks, %d failures\n", g_checks, g_failures);
  std::printf("[test_event_machine] %s\n", g_failures == 0 ? "PASS" : "FAIL");
  mj_deleteModel(m);
  return g_failures == 0 ? 0 : 1;
}
