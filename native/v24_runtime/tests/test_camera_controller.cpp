// Unit tests for camera_controller.h: follow/top/free mode cycling and
// mjvCamera configuration. Uses the synthetic test model only to obtain a valid
// body id (see synthetic_model.h); no GL context needed.
//
// Usage: test_camera_controller
#include <cstdio>
#include <string>

#include <mujoco/mujoco.h>

#include "synthetic_model.h"

#include "biped_native/camera_controller.h"

namespace {
int g_checks = 0;
int g_failures = 0;

void Check(bool cond, const std::string& msg) {
  ++g_checks;
  if (!cond) {
    ++g_failures;
    std::printf("[test_camera_controller] FAIL: %s\n", msg.c_str());
  }
}
}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  mjModel* m = synthetic::MakeModel();
  const int root_id = mj_name2id(m, mjOBJ_BODY, "synthetic_root");
  Check(root_id >= 0, "body 'synthetic_root' must exist in the synthetic model");

  biped::CameraController camera(root_id);

  // Constructed state: kFollow (per the constructor's ApplyMode(kFollow) call).
  Check(camera.Mode() == biped::CameraMode::kFollow, "initial mode must be kFollow");
  Check(camera.Camera().type == mjCAMERA_TRACKING, "kFollow must use mjCAMERA_TRACKING");
  Check(camera.Camera().trackbodyid == root_id, "kFollow must track the root body id");

  // Cycle kFollow -> kTop.
  camera.CycleMode();
  Check(camera.Mode() == biped::CameraMode::kTop, "1st cycle must reach kTop");
  Check(camera.Camera().type == mjCAMERA_TRACKING, "kTop must use mjCAMERA_TRACKING");
  Check(camera.Camera().trackbodyid == root_id, "kTop must track the root body id");
  Check(camera.Camera().elevation < -85.0, "kTop must look nearly straight down");

  // Cycle kTop -> kFree.
  camera.CycleMode();
  Check(camera.Mode() == biped::CameraMode::kFree, "2nd cycle must reach kFree");
  Check(camera.Camera().type == mjCAMERA_FREE, "kFree must use mjCAMERA_FREE");
  Check(camera.Camera().trackbodyid == -1, "kFree must not track any body");

  // Cycle kFree -> kFollow (wraps around).
  camera.CycleMode();
  Check(camera.Mode() == biped::CameraMode::kFollow, "3rd cycle must wrap back to kFollow");
  Check(camera.Camera().type == mjCAMERA_TRACKING, "wrapped kFollow must use mjCAMERA_TRACKING");

  // A 4th cycle must reproduce the exact same kTop parameters as the 1st
  // (deterministic, no hidden state drift across repeated cycles).
  camera.CycleMode();
  const double top_elevation_first_pass = camera.Camera().elevation;
  const double top_distance_first_pass = camera.Camera().distance;
  camera.CycleMode();
  camera.CycleMode();
  camera.CycleMode();  // back to kTop again (4 cycles later)
  Check(camera.Mode() == biped::CameraMode::kTop, "mode sequence must be periodic with period 3");
  Check(camera.Camera().elevation == top_elevation_first_pass,
        "repeated kTop visits must be parameter-identical (no drift)");
  Check(camera.Camera().distance == top_distance_first_pass,
        "repeated kTop visits must be parameter-identical (no drift)");

  std::printf("[test_camera_controller] %d checks, %d failures\n", g_checks, g_failures);
  std::printf("[test_camera_controller] %s\n", g_failures == 0 ? "PASS" : "FAIL");
  mj_deleteModel(m);
  return g_failures == 0 ? 0 : 1;
}
