// Contract test for onnx_actor.cpp against a SYNTHETIC ONNX actor produced by
// tools/make_synthetic_actor.py (random weights, no trained policy). It checks
// shape contract enforcement, batching consistency and agreement with the
// reference outputs that the Python generator computed with a NumPy forward pass.
//
// Usage: test_onnx_actor <synthetic_actor.onnx> <synthetic_io.bin>
// The .bin layout: int32 n, then n*57 float32 observations, then n*12 float32 actions.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "biped_native/onnx_actor.h"
#include "biped_native/runtime_layout.h"
#include "check.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <synthetic_actor.onnx> <synthetic_io.bin>\n", argv[0]);
    return 2;
  }
  using namespace biped;
  std::ifstream f(argv[2], std::ios::binary);
  int32_t n = 0;
  f.read(reinterpret_cast<char*>(&n), sizeof(n));
  std::vector<float> obs(static_cast<size_t>(n) * kObsSize), ref(static_cast<size_t>(n) * kActionSize);
  f.read(reinterpret_cast<char*>(obs.data()), obs.size() * sizeof(float));
  f.read(reinterpret_cast<char*>(ref.data()), ref.size() * sizeof(float));
  testing::Check(static_cast<bool>(f) && n > 0, "io fixture readable");

  OnnxActor actor(argv[1]);
  testing::Check(!actor.InputName().empty() && !actor.OutputName().empty(), "io names resolved");
  testing::Check(actor.Providers().size() == 1 && actor.Providers()[0] == "CPUExecutionProvider",
                 "CPU provider only");

  const std::vector<float> batch = actor.RunBatch(obs, n);
  double max_diff = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) max_diff = std::fmax(max_diff, std::fabs(batch[i] - ref[i]));
  testing::Check(max_diff < 1e-4, "ONNX Runtime matches NumPy reference (max |diff| = " +
                                      std::to_string(max_diff) + ")");

  double max_single = 0.0;
  for (int r = 0; r < n; ++r) {
    std::array<float, kObsSize> o{};
    std::copy(obs.begin() + r * kObsSize, obs.begin() + (r + 1) * kObsSize, o.begin());
    const auto a = actor.Run(o);
    for (int j = 0; j < kActionSize; ++j) {
      max_single = std::fmax(max_single, std::fabs(a[j] - batch[r * kActionSize + j]));
    }
  }
  testing::Check(max_single < 1e-5, "single-row Run equals batched RunBatch");

  bool bounded = true;
  for (float v : batch) bounded &= (v >= -1.0f && v <= 1.0f);
  testing::Check(bounded, "tanh head output within [-1, 1]");

  bool threw = false;
  try {
    actor.RunBatch(std::vector<float>(kObsSize - 1), 1);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  testing::Check(threw, "wrong batch size rejected");
  return testing::Finish("test_onnx_actor");
}
