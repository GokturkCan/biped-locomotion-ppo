// Unit tests for push_controller.h: force pulse timing and clearing. Uses the
// synthetic test model only for a validly sized mjData (see synthetic_model.h).
//
// Usage: test_push_controller
#include <cmath>
#include <cstdio>
#include <string>

#include <mujoco/mujoco.h>

#include "synthetic_model.h"

#include "biped_native/push_controller.h"
#include "biped_native/runtime_layout.h"

namespace {
int g_checks = 0;
int g_failures = 0;

void Check(bool cond, const std::string& msg) {
  ++g_checks;
  if (!cond) {
    ++g_failures;
    std::printf("[test_push_controller] FAIL: %s\n", msg.c_str());
  }
}
}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  mjModel* m = synthetic::MakeModel();
  mjData* d = mj_makeData(m);

  const int root_id = mj_name2id(m, mjOBJ_BODY, "synthetic_root");
  Check(root_id >= 0, "body 'synthetic_root' must exist in the synthetic model");

  const std::array<double, 3> force = {0.0, 60.0, 0.0};
  const double duration_s = 0.2;
  biped::PushController push(root_id, force, duration_s);

  Check(!push.IsActive(), "PushController must start inactive");
  auto zero = push.CurrentForce();
  Check(zero[0] == 0.0 && zero[1] == 0.0 && zero[2] == 0.0,
        "CurrentForce() must be zero before any Trigger()");

  // Before any Trigger(), Apply() must not write a nonzero force.
  push.Apply(d, biped::kSimDt * biped::kSubstepsPerTick);
  Check(d->xfrc_applied[root_id * 6 + 1] == 0.0,
        "xfrc_applied must be zero before Trigger() is ever called");

  push.Trigger();
  Check(push.IsActive(), "PushController must be active immediately after Trigger()");
  auto cf = push.CurrentForce();
  Check(cf[0] == force[0] && cf[1] == force[1] && cf[2] == force[2],
        "CurrentForce() must equal the configured force while active");

  // Apply for less than the full duration: force must remain applied.
  const double dt = biped::kSimDt * biped::kSubstepsPerTick;  // 0.02s/tick
  const int ticks_within_duration = static_cast<int>(duration_s / dt) - 1;  // 9 ticks = 0.18s
  for (int i = 0; i < ticks_within_duration; ++i) {
    push.Apply(d, dt);
    Check(push.IsActive(), "push must remain active mid-duration (tick " + std::to_string(i) + ")");
    Check(d->xfrc_applied[root_id * 6 + 0] == force[0] &&
              d->xfrc_applied[root_id * 6 + 1] == force[1] &&
              d->xfrc_applied[root_id * 6 + 2] == force[2],
          "xfrc_applied must match the configured force while active (tick " +
              std::to_string(i) + ")");
  }

  // One more tick crosses the duration boundary: force must clear to zero
  // and IsActive() must become false.
  push.Apply(d, dt);
  Check(!push.IsActive(), "push must deactivate once elapsed_s reaches duration_s");
  Check(d->xfrc_applied[root_id * 6 + 0] == 0.0 && d->xfrc_applied[root_id * 6 + 1] == 0.0 &&
            d->xfrc_applied[root_id * 6 + 2] == 0.0,
        "xfrc_applied must be cleared to zero once the push deactivates");
  auto cf_after = push.CurrentForce();
  Check(cf_after[0] == 0.0 && cf_after[1] == 0.0 && cf_after[2] == 0.0,
        "CurrentForce() must be zero after deactivation");

  // Applying further ticks with no active push must keep xfrc_applied at
  // zero (not stale/leftover values).
  push.Apply(d, dt);
  Check(d->xfrc_applied[root_id * 6 + 1] == 0.0,
        "xfrc_applied must stay zero on subsequent ticks with no active push");

  // Re-triggering resets the timer and reapplies the force.
  push.Trigger();
  push.Apply(d, dt);
  Check(push.IsActive() && d->xfrc_applied[root_id * 6 + 1] == force[1],
        "re-Trigger() must reactivate the push and reapply the force");

  std::printf("[test_push_controller] %d checks, %d failures\n", g_checks, g_failures);
  std::printf("[test_push_controller] %s\n", g_failures == 0 ? "PASS" : "FAIL");
  mj_deleteData(d);
  mj_deleteModel(m);
  return g_failures == 0 ? 0 : 1;
}
