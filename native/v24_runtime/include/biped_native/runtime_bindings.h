// External runtime bindings: the boundary between this public runtime code and
// the (private) robot model it is run against.
//
// Nothing in this repository describes a robot. The values below must be
// supplied by an authorized user in a JSON file kept outside git; see
// native/v24_runtime/README.md for the schema and bindings/ for a template.
// This mirrors the Python-side `ModelSpec` (docs/model_access.md).
#pragma once

#include <array>
#include <string>
#include <vector>

#include "biped_native/runtime_layout.h"

namespace biped {

struct RuntimeBindings {
  // Ordered joint names: the order defines observation/action channel order.
  std::array<std::string, kNumJoints> joint_names;

  // Model sensor names (resolved to addresses at runtime).
  std::string sensor_orientation;  // unit quaternion, wxyz (4 values)
  std::string sensor_gravity;      // up-vector in body frame (3)
  std::string sensor_local_linvel; // body-frame linear velocity (3)
  std::string sensor_gyro;         // angular velocity (3)
  std::string sensor_root_pos;     // world root position (3)

  // Contact bindings and the body that external pushes / the camera act on.
  std::string floor_geom;
  std::string left_foot_geom;
  std::string right_foot_geom;
  std::string root_body;

  // Per-channel residual scale of the position-target law.
  std::array<float, kActionSize> action_scale{};

  // Full qpos used to start/reset an episode (length must equal model nq).
  std::vector<double> initial_qpos;

  // Root-height settings (metres): reward-gate target and fall threshold.
  double target_height = 0.0;
  double min_root_height = 0.0;
};

// Throws std::runtime_error with a precise message for missing fields, wrong
// sizes, non-finite or non-positive values, or unfilled template placeholders.
RuntimeBindings ParseRuntimeBindings(const std::string& json_text);
RuntimeBindings LoadRuntimeBindings(const std::string& path);

}  // namespace biped
