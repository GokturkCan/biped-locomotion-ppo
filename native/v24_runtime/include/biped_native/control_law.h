// Action -> joint position target mapping of the V24 residual-position law:
//   target = clip(q_reference + action_scale * action, joint_lower, joint_upper)
// q_reference is the settled reset pose (fixed per episode). action_scale comes
// from RuntimeBindings and the joint limits from the loaded model; none are
// compiled in. No further residual multiplier is applied.
#pragma once

#include <array>

#include "biped_native/runtime_layout.h"

namespace biped {

std::array<float, kActionSize> MapActionToTarget(
    const std::array<float, kActionSize>& action,
    const std::array<float, kNumJoints>& bridge_q_ref_base,
    const std::array<float, kNumJoints>& action_scale,
    const std::array<float, kNumJoints>& joint_lowers,
    const std::array<float, kNumJoints>& joint_uppers);

}  // namespace biped
