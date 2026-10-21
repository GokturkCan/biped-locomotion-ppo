#include "biped_native/control_law.h"

#include <algorithm>

namespace biped {

std::array<float, kActionSize> MapActionToTarget(
    const std::array<float, kActionSize>& action,
    const std::array<float, kNumJoints>& bridge_q_ref_base,
    const std::array<float, kNumJoints>& action_scale,
    const std::array<float, kNumJoints>& joint_lowers,
    const std::array<float, kNumJoints>& joint_uppers) {
  std::array<float, kActionSize> target{};
  for (int i = 0; i < kActionSize; ++i) {
    const float raw = bridge_q_ref_base[i] + action_scale[i] * action[i];
    target[i] = std::min(std::max(raw, joint_lowers[i]), joint_uppers[i]);
  }
  return target;
}

}  // namespace biped
