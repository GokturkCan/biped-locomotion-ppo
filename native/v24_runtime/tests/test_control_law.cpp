// Tests for control_law.cpp: target = clip(reference + scale * action, lo, hi).
//
// Usage: test_control_law
#include <array>

#include "biped_native/control_law.h"
#include "check.h"

int main() {
  using namespace biped;
  std::array<float, kActionSize> action{}, ref{}, scale{}, lo{}, hi{};
  for (int i = 0; i < kActionSize; ++i) {
    action[i] = (i % 3 == 0) ? 1.0f : (i % 3 == 1 ? -1.0f : 0.0f);
    ref[i] = 0.1f * i;
    scale[i] = 0.05f;
    lo[i] = ref[i] - 0.03f;
    hi[i] = ref[i] + 0.02f;
  }
  const auto t = MapActionToTarget(action, ref, scale, lo, hi);
  testing::CheckNear(t[0], ref[0] + 0.02, 1e-6, "positive residual clipped to upper limit");
  testing::CheckNear(t[1], ref[1] - 0.03, 1e-6, "negative residual clipped to lower limit");
  testing::CheckNear(t[2], ref[2], 1e-6, "zero action returns the reference");

  scale.fill(0.01f);
  const auto u = MapActionToTarget(action, ref, scale, lo, hi);
  testing::CheckNear(u[0], ref[0] + 0.01, 1e-6, "unclipped positive residual = scale*action");
  testing::CheckNear(u[1], ref[1] - 0.01, 1e-6, "unclipped negative residual = scale*action");
  return testing::Finish("test_control_law");
}
